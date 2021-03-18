#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys
import h5py
import yaml

import click
from IPython.core import ultratb

import numpy as np
import pymaster as nmt
from pathlib import Path
import h5py
import began
import pandas as pd
from began.visualization import mplot, plot
from began import stats

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger('began')

@click.command()
@click.option('--cfg_path', 'cfg_path', required=True,
                type=click.Path(exists=True), help='path to config file of map cutting')
@click.option('--model_cfg_path', 'model_cfg_path', required=True,
                type=click.Path(exists=True), help='path to config file of map cutting')
@click.option('--model_path', 'model_path', required=True,
                type=click.Path(exists=True), help='path to config file of map cutting')
@click.option('--input_path', 'input_path', required=True,
                type=click.Path(exists=True), help='path to input file from which to load maps')
@click.option('--output_dir', 'output_dir', type=click.Path(exists=True), required=True)
@click.option('--plot_dir', 'plot_dir', type=click.Path(exists=True), required=True)
@click.option('--batch_size', 'batch_size', type=int, default=-1, help='Size of batch to calculate')
@click.option('--aposize', 'aposize', type=float, default=2., help='scale of mask apodization')
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(cfg_path: Path, model_cfg_path: Path, model_path: Path, input_path: Path, output_dir: Path, plot_dir: Path, batch_size: int, aposize: float, seed: int, log_level: int):
    # initialize random seed in numpy
    np.random.seed(seed)
    plot_dir = Path(plot_dir).absolute()
    output_dir = Path(output_dir).absolute()
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # get samples from the model
    _logger.info("""
    Data being loaded: {:s}
    Output directory: {:s}
    Batch size used: {:d}
    """.format(input_path, str(output_dir), batch_size))

    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(model_cfg_path) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    ang_x = config['patch']['ang_x'] * np.pi / 180. # angular size of patch in degrees
    ang_y = config['patch']['ang_y'] * np.pi / 180. # angular size of patch in degrees
    npix_x = config['pixelization']['xres'] # pixelization in x dimension
    npix_y = config['pixelization']['yres'] # pixelization in y dimension
    
    _logger.info("""Creating mask with the specifications:
    x resolution: {:d} pixels
    y resolution: {:d} pixels
    x linear size: {:.03f} degrees
    y linear size: {:.03f} degrees
    apodization scale: {:.03f} degrees
    """.format(npix_x, npix_y, ang_x * 180. / np.pi, ang_y * 180. / np.pi, aposize))

    # read in data
    with h5py.File(input_path, 'r') as f:
        map_batch = f['cut_maps'][:batch_size, :, :, 0]
    assert map_batch.ndim == 3
    _logger.debug(repr(map_batch.shape))


    # read in model
    cvae = began.CVAE(model_cfg['architecture']['lat_dim'], model_cfg['architecture']['kernel_size'])
    cvae.load_weights(model_path)
    vae_batch = cvae.decode(np.random.randn(batch_size, model_cfg['architecture']['lat_dim']))

    ###########################################################################
    #
    # ONE-POINT STATISTICS
    #
    ###########################################################################


    ###########################################################################
    #
    # TWO-POINT STATISTICS
    #
    ###########################################################################
    # create flat mask with taper at edges
    mask = stats.build_flat_mask(npix_x, npix_y, ang_x, ang_y, aposize)
    fig, ax = plot(mask, xlabel="x", ylabel="y", title="Apodized mask", extent=(-10, 10, -10, 10))
    fig.savefig(plot_dir / "apodized_mask.pdf", bbox_inches='tight')

    # evaluate the power spectrum of the sampled maps
    nmtbin = stats.dimensions_to_nmtbin(npix_x, npix_y, ang_x, ang_y, is_Dell=True)
    _logger.info("""
    Binning parameters:
        nbands: {:d}
        effective ells: {:s}
        """.format(int(nmtbin.get_n_bands()), np.array2string(nmtbin.get_effective_ells())))
    autospectra = stats.batch_00_autospectrum(map_batch, ang_x, ang_y, mask, nmtbin)

    _logger.debug("Batch autospectrum shape: ", repr(autospectra.shape))
    _logger.debug("Batch autospectrum:", np.array2string(autospectra))

    bpws = []
    batch_num = []
    multipoles = []
    for i, spectrum in enumerate(autospectra):
        for ell, bpw in zip(nmtbin.get_effective_ells(), spectrum[0]):
            multipoles.append(ell)
            bpws.append(bpw)
            batch_num.append(i)

    df = pd.DataFrame({'bandpowers': bpws, 'multipoles': multipoles, 'batch_number': batch_num})
    df.to_hdf(output_dir / "metrics.h5", "train_spectra")

    autospectra = stats.batch_00_autospectrum(vae_batch, ang_x, ang_y, mask, nmtbin)
    
    bpws = []
    batch_num = []
    multipoles = []
    for i, spectrum in enumerate(autospectra):
        for ell, bpw in zip(nmtbin.get_effective_ells(), spectrum[0]):
            multipoles.append(ell)
            bpws.append(bpw)
            batch_num.append(i)

    df = pd.DataFrame({'bandpowers': bpws, 'multipoles': multipoles, 'batch_number': batch_num})
    df.to_hdf(output_dir / "metrics.h5", "vae_spectra")

    ###########################################################################
    #
    # HIGHER-ORDER STATISTICS
    #
    ###########################################################################

    #with h5py.File(output_dir / "metrics.h5", "a") as handle:
    #    g = handle.create_group("batch_auto")
    #    g.create_dataset("bandpower", data=nmtbin.get_effective_ells())
    #    g.create_dataset("autospectra", data=auto_spectra)
    # calculate the Frechet distance from the distribution of power spectra in the training set

    # save summary statistics in hdf5 file



if __name__ == '__main__':
    main()
