""" This file plots summary statistics of the trained VQVAE model.
This includes:

1. Example plots of predicted maps from the training set.
2. Power spectrum distribution of testing set compared to power
spectrum of predicted maps.
3. Pixel value histograms of test and predicted sets.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tree
import os
import sys

from pathlib import Path
import xarray as xa
from xhistogram.xarray import histogram

from began import rotate, flip
from began import Encoder, Decoder, VQVAEModel, ResidualStack
from began import make_square_mask, make_flat_bins, apply_nmt_flat
import pymaster as nmt

plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5, 5]

if __name__ == '__main__':

    PLOT_DIR = Path('/home/bthorne/projects/gan/began/reports/figures/slides/temp/planck')

    model = tf.saved_model.load("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/vqvae-saved")

    data_dir = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data")
    test_fpath = data_dir / "mhd_ntest-0300.cdf"
    test_images = xa.open_dataarray(test_fpath).sel(pol=['t']).transpose('batch', ..., 'pol').values.astype(np.float32)

    data_dir = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data")
    test_fpath = data_dir / "mhd_cl_ntest-0300.cdf"
    test_xr_cl = xa.open_dataarray(test_fpath)

    planck_maps_xr = xa.open_dataarray("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data/planck_cutouts_30deg.cdf")

    dims = ['batch', 'x', 'y', 'pol']    
    coords = {
        'batch': np.arange(planck_maps_xr.batch.size),
        'x': np.arange(256),
        'y': np.arange(256),
        'pol': ['t']
    }
    planck_reco_xr = xa.DataArray(model.inference(planck_maps_xr.values.astype(np.float32))['x_recon'].numpy(), coords=coords, dims=dims, name='planck_recon')
    resi_xr = planck_maps_xr - planck_reco_xr
    resi_xr.name = 'residuals'

    vmin = 0.
    vmax = 2.
    cmap = 'cividis'
    batch_idx = [10, 50, 100]
    fg = planck_reco_xr.sel(batch=batch_idx, pol='t').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin, vmax=vmax)
    fg.fig.savefig(PLOT_DIR / 'vqvae-temp-planck_recon.png', bbox_inches='tight')
    fg = planck_maps_xr.sel(batch=batch_idx, pol='t').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin, vmax=vmax)
    fg.fig.savefig(PLOT_DIR / 'vqvae-temp-planck_maps.png', bbox_inches='tight')
    fg = resi_xr.sel(batch=batch_idx, pol='t').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=-vmax/10., vmax=vmax/10.)
    fg.fig.savefig(PLOT_DIR / 'vqvae-temp-planck_residuals.png', bbox_inches='tight')


    bins = np.linspace(vmin, vmax, 100)
    planck_maps_xr_hist = histogram(planck_maps_xr.sel(pol='t'), bins=[bins], dim=['x', 'y'])
    planck_reco_xr_hist = histogram(planck_reco_xr.sel(pol='t'), bins=[bins], dim=['x', 'y'])

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(planck_maps_xr_hist.cut_maps_bin, planck_maps_xr_hist.quantile(0.25, 'batch'), planck_maps_xr_hist.quantile(0.75, 'batch'), label='Planck images', alpha=0.4)
    ax.fill_between(planck_reco_xr_hist.planck_recon_bin, planck_reco_xr_hist.quantile(0.25, 'batch'), planck_reco_xr_hist.quantile(0.75, 'batch'), label='Reconstructions', alpha=0.4)
    ax.legend(frameon=False)
    ax.set_label(r"Density")
    ax.set_xlabel(r"$T~{\rm [arb. units]}$")
    ax.set_xlim(0.3, 1.75)
    ax.set_title("Interquartile range of pixel histogram")
    fig.savefig(PLOT_DIR / "vqvae-temp-planck_hist.png", bbox_to_inches='tight')

    bins = np.linspace(-0.1, 0.1, 300)
    resi_xr_hist = histogram(resi_xr.sel(pol='t'), bins=[bins], dim=['x', 'y'])
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(resi_xr_hist.residuals_bin, resi_xr_hist.quantile(0.25, 'batch'), resi_xr_hist.quantile(0.75, 'batch'), alpha=0.5)
    ax.set_ylabel(r"Frequency")
    ax.set_xlabel(r"$\delta T~{\rm [arb. units]}$")
    ax.set_title("Planck fit residuals histogram")
    ax.axvline(x=0, color='k', alpha=0.5)
    fig.savefig(PLOT_DIR / "vqvae-temp-planck_hist_residuals.png", bbox_inches='tight')

    nx = 256
    ny = 256
    theta = 29.
    ang = np.radians(theta)
    mask = make_square_mask(nx, ny, ang)
    binning = make_flat_bins(ang, nx, 8)

    f0 = nmt.NmtFieldFlat(ang, ang, mask, np.random.randn(1, nx, nx))

    wsp00 = nmt.NmtWorkspaceFlat()
    wsp00.compute_coupling_matrix(f0, f0, binning)

    planck_reco_xr_cl = apply_nmt_flat(planck_reco_xr, mask, ang, binning, wsp00=wsp00)
    planck_maps_xr_cl = apply_nmt_flat(planck_maps_xr, mask, ang, binning, wsp00=wsp00)

    test_quantiles = planck_maps_xr_cl.quantile([0.25, 0.75], 'batch').sel(field='tt')
    recon_quantiles = planck_reco_xr_cl.quantile([0.27, 0.75], 'batch').sel(field='tt')
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(test_quantiles.bandpowers, test_quantiles.isel(quantile=0), test_quantiles.isel(quantile=1), alpha=0.6, label='Test set')
    ax.fill_between(recon_quantiles.bandpowers, recon_quantiles.isel(quantile=0), recon_quantiles.isel(quantile=1), alpha=0.6, label='Reconstruction')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_ylim(1e-10, 1e-6)
    ax.set_xlim(230, 1000)
    ax.set_ylabel(r'$C_\ell^{\rm TT}~[{\rm arb. units}]$')
    ax.set_xlabel(r'$\ell_b$')
    fig.savefig(PLOT_DIR / 'vqvae-temp-planck_cl.png', bbox_to_inches='tight')