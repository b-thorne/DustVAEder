#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys
import yaml
import h5py

import click
from IPython.core import ultratb

import healpy as hp 
import numpy as np
import astropy.units as u
from astropy.io import fits
from began.tools import get_patch_centers, FlatCutter




# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)

@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('--input_path', 'input_path', required=True, 
                type=click.Path(exists=True), help='path to input data')
@click.option('--output_path', 'output_path', required=True,
                type=click.Path(), help='path to output file')
@click.option('-p', 'polarization', default=False)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(cfg_path: Path, input_path: Path, output_path: Path, polarization: bool, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("""
    Configuration: {:s}
    Input {:s}
    Output: {:s}""".format(cfg_path, input_path, output_path))

    # define a configuration dictionary
    cfg = {}

    # read configuration file
    with open(cfg_path) as f:
        cfg.update(yaml.load(f, Loader=yaml.FullLoader))

    cfg['gal_cut'] *= u.deg
    cfg['step_size'] *= u.deg
    cfg['ang_x'] *= u.deg 
    cfg['ang_y'] *= u.deg

    # read map and infer nside
    if polarization:
        cfg['polarization'] = True
        field = (0, 1, 2)
    else:
        cfg['polarization'] = False
        field = 0

    input_map = hp.read_map(input_path, field=field, dtype=np.float64, verbose=False)
    logging.debug("Input map fits header: \n {:s}".format(repr(fits.open(input_path)[1].header)))

    logging.info(
        """Cutting map with tiling: 
        gal_cut: {:.01f}  
        step_size: {:.01f}""".format(cfg['gal_cut'], cfg['step_size']))

    centers = get_patch_centers(cfg['gal_cut'], cfg['step_size'])
   
    logging.debug("Number of patches: {:d}".format(len(centers)))

    logging.info(
        """Patch parameters:
        Linear size in degrees: {:.01f} degrees by {:.01f} degrees
        Number of pixels: {:.01f} by {:.01f} 
        """.format(cfg['ang_x'], cfg['ang_y'], cfg['xres'], cfg['yres']))

    # cut out maps at each of the patch centers
    fc = FlatCutter(cfg['ang_x'], cfg['ang_y'], cfg['xres'], cfg['yres'])
    cut_maps = np.array([fc.rotate_to_pole_and_interpolate(lon, lat, input_map) for (lon, lat) in centers])

    # save maps and add new axis at end corresponding to channel
    with h5py.File(output_path, "a") as f:
        a = f.require_dataset("cut_maps", shape=cut_maps.shape, dtype=cut_maps.dtype)
        a[...] = cut_maps
        a.attrs.update(cfg)

if __name__ == '__main__':
    main()  