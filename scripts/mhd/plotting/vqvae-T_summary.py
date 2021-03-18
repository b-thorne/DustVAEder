""" This file plots summary statistics of the trained VQVAE model.
This includes:

1. Example plots of predicted maps from the training set.
2. Power spectrum distribution of testing set compared to power
spectrum of predicted maps.
3. Pixel value histograms of test and predicted sets.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tree
import xarray as xa
from xhistogram.xarray import histogram

from began import (
    Decoder,
    Encoder,
    ResidualStack,
    VQVAEModel,
    apply_nmt_flat,
    flip,
    make_flat_bins,
    make_square_mask,
    rotate
)

plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5, 5]

if __name__ == '__main__':

    PLOT_DIR = Path('/home/bthorne/projects/gan/began/reports/figures/slides/temp/test')

    # Read in model
    model = tf.saved_model.load("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/vqvae-saved")

    # Read in test maps
    data_dir = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/data")
    test_fpath = data_dir / "mhd_ntest-0300.cdf"
    test_xr= xa.open_dataarray(test_fpath).sel(pol=['t'])
    test_xr.name = 'test'

    # Read in test power spectra
    test_fpath = data_dir / "mhd_cl_ntest-0300.cdf"
    test_xr_cl = xa.open_dataarray(test_fpath)

    # Make predictions on test set and compute their power spectra
    reco_xr = test_xr.copy(data=model.inference(test_xr.transpose('batch', ..., 'pol').values.astype(np.float32))['x_recon'])
    reco_xr.name = 'predictions'
    reco_xr_cl = apply_nmt_flat(reco_xr)

    # Calculate residuals
    resi_xr = test_xr - reco_xr
    resi_xr.name = 'residuals'


    # Make map-level plots of test data and reconstruction
    batch_idx = np.random.randint(0, test_xr.batch.size, 3)
    cmap = 'cividis'
    vmin = 0
    vmax = 0.3
    fg = test_xr.sel(batch=batch_idx, pol='t').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin, vmax=vmax)
    fg.fig.savefig(PLOT_DIR / "vqvae-T-test_maps.png", bbox_inches='tight')
    fg = reco_xr.sel(batch=batch_idx, pol='t').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin, vmax=vmax)
    fg.fig.savefig(PLOT_DIR / "vqvae-T-test_reco.png", bbox_inches='tight')
    fg = resi_xr.sel(batch=batch_idx, pol='t').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=-0.05, vmax=0.05)
    fg.fig.savefig(PLOT_DIR / "vqvae-T-test_resi.png", bbox_inches='tight')

    # Make histograms of map level residuals
    bins = np.linspace(-0.01, 0.01, 50)
    resi_xr_hist = histogram(resi_xr, bins=[bins], dim=['x', 'y']).sel(pol='t')
    resi_xr_hist

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(resi_xr_hist.residuals_bin, resi_xr_hist.quantile(0.25, 'batch'), resi_xr_hist.quantile(0.75, 'batch'), alpha=0.5)
    ax.set_ylabel(r"Count")
    ax.set_xlabel(r"$\delta T~{\rm [arb. units]}$")
    ax.set_title("Histogram of test set residuals")
    ax.set_xticks([-0.01, -0.005, 0, 0.005, 0.01])
    ax.axvline(x=0, color='k', alpha=0.5)
    fig.savefig(PLOT_DIR / "vqvae-T-hist_resi.png", bbox_inches='tight')

    resi_xr_pct = resi_xr / test_xr
    resi_xr_pct.name = 'residuals_pct'
    bins = np.linspace(-0.1, 0.1, 200)
    resi_xr_hist_pct = histogram(resi_xr_pct, bins=[bins], dim=['x', 'y']).sel(pol='t')

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(resi_xr_hist_pct.residuals_pct_bin, resi_xr_hist_pct.quantile(0.25, 'batch'), resi_xr_hist_pct.quantile(0.75, 'batch'), alpha=0.5)
    ax.set_ylabel(r"Count")
    ax.set_xlabel(r"$\delta T / T$")
    ax.set_title("Histogram of test set residuals")
    ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax.axvline(x=0, color='k', alpha=0.5)
    fig.savefig(PLOT_DIR / "vqvae-T-hist_resi_pct.png", bbox_inches='tight')

    bins = np.linspace(vmin, vmax, 100)
    test_xr_hist = histogram(test_xr.sel(pol='t'), bins=[bins], dim=['x', 'y'])
    reco_xr_hist = histogram(reco_xr.sel(pol='t'), bins=[bins], dim=['x', 'y'])

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(reco_xr_hist.predictions_bin, reco_xr_hist.quantile(0.25, 'batch'), reco_xr_hist.quantile(0.75, 'batch'), label='Test images', alpha=0.4)
    ax.fill_between(test_xr_hist.test_bin, test_xr_hist.quantile(0.25, 'batch'), test_xr_hist.quantile(0.75, 'batch'), label='Reconstructions', alpha=0.4)
    ax.legend(frameon=False)
    ax.set_ylabel(r"Count")
    ax.set_xlabel(r"$T~{\rm [arb. units]}$")
    ax.set_title("Histogram of test set and predictions")
    ax.axvline(x=0, color='k', alpha=0.5)
    fig.savefig(PLOT_DIR / "vqvae-T-hist_test.png", bbox_inches='tight')

    # Make plots of power spectra of test images and reconstructions
    test_quantiles = test_xr_cl.quantile([0.25, 0.75], 'batch').sel(field='tt')
    recon_quantiles = reco_xr_cl.quantile([0.27, 0.75], 'batch').sel(field='tt')
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(test_quantiles.bandpowers, test_quantiles.isel(quantile=0), test_quantiles.isel(quantile=1), alpha=0.6, label='Test set')
    ax.fill_between(recon_quantiles.bandpowers, recon_quantiles.isel(quantile=0), recon_quantiles.isel(quantile=1), alpha=0.6, label='Reconstruction')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_ylim(1e-13, 1e-7)
    #ax.set_xlim(200, 1500)
    ax.set_ylabel(r'$C_\ell^{\rm TT}~[{\rm arb. units}]$')
    ax.set_xlabel(r'$\ell_b$')
    fig.savefig(PLOT_DIR / 'vqvae-temp-cl.png', bbox_inches='tight')
