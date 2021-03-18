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
import sys
import os

from pathlib import Path
import xarray as xa
from xhistogram.xarray import histogram

from began import rotate, flip
from began import Encoder, Decoder, VQVAEModel, ResidualStack
from began import make_square_mask, make_flat_bins, apply_nmt_flat
import pymaster as nmt

import matplotlib as mpl 
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5, 5]

if __name__ == '__main__':

    PLOT_DIR = Path('/home/bthorne/projects/gan/began/reports/figures/slides/pol/test')

    # Read in model
    model = tf.saved_model.load("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_pol/pol-vqvae-1024")

    # Read in test images
    data_dir = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/data")
    test_fpath = data_dir / "mhd_ntest-0300.cdf"
    test_xr = xa.open_dataarray(test_fpath).sel(pol=['q', 'u']).isel(batch=np.arange(100)).transpose('batch', ..., 'pol')
    test_xr.name = 'test'
    
    # Read in test power spectra
    test_fpath = data_dir / "mhd_cl_ntest-0300.cdf"
    test_xr_cl = apply_nmt_flat(test_xr )

    # Calculate reconstructions
    reco_xr = test_xr.copy(data=model.inference(test_xr.values.astype(np.float32))['x_recon'])
    reco_xr.name = 'predictions'
    reco_xr_cl = apply_nmt_flat(reco_xr)

    # Calculate residuals
    resi_xr = test_xr - reco_xr
    resi_xr.name = 'residuals'

    # Plot a selection of map level reconstructions
    batch_idx = np.random.randint(0, resi_xr.batch.size, 3)
    cmap = 'cividis'
    vmin = -0.025
    vmax = 0.025
    fg = test_xr.sel(batch=batch_idx, pol='q').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin, vmax=vmax)
    fg.fig.savefig(PLOT_DIR / 'vqvae-P-test_q.png', bbox_inches='tight')
    fg = reco_xr.sel(batch=batch_idx, pol='q').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin, vmax=vmax)
    fg.fig.savefig(PLOT_DIR / 'vqvae-P-pred_q.png', bbox_inches='tight')
    fg = resi_xr.sel(batch=batch_idx, pol='q').plot.pcolormesh('x', 'y', col_wrap=3, col='batch', cmap=cmap, vmin=vmin/10., vmax=vmax/10.)
    fg.fig.savefig(PLOT_DIR / 'vqvae-P-resi_q.png', bbox_inches='tight')

    # Plot histograms of Q-residuals
    bins = np.linspace(-0.002, 0.002, 50)
    resi_xr_hist = histogram(resi_xr, bins=[bins], dim=['x', 'y']).sel(pol='q')
    resi_xr_hist

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(resi_xr_hist.residuals_bin, resi_xr_hist.quantile(0.25, 'batch'), resi_xr_hist.quantile(0.75, 'batch'), label='Bootstrapped test images', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), frameon=False)
    ax.set_ylabel(r"Count")
    ax.set_xlabel(r"$\delta Q~{\rm [arb. units]}$")
    ax.set_title("Interquartile range of pixel histogram")
    ax.axvline(x=0, color='k', alpha=0.5)
    fig.savefig(PLOT_DIR / 'vqvae-P-resi_hist_q.png', bbox_inches='tight')
    
    bins = np.linspace(-0.002, 0.002, 100)
    test_xr_hist = histogram(test_xr.sel(pol='q'), bins=[bins], dim=['x', 'y'])
    reco_xr_hist = histogram(reco_xr.sel(pol='q'), bins=[bins], dim=['x', 'y'])

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(reco_xr_hist.predictions_bin, reco_xr_hist.quantile(0.25, 'batch'), reco_xr_hist.quantile(0.75, 'batch'), label='Test images', alpha=0.4)
    ax.fill_between(test_xr_hist.test_bin, test_xr_hist.quantile(0.25, 'batch'), test_xr_hist.quantile(0.75, 'batch'), label='Reconstructions', alpha=0.4)
    ax.legend(frameon=False)
    ax.set_ylabel(r"Count")
    ax.set_xlabel(r"$\delta Q~{\rm [arb. units]}$")
    ax.set_title("Interquartile range of pixel histogram")
    ax.axvline(x=0, color='k', alpha=0.5)
    fig.savefig(PLOT_DIR / "vqvae-P-test_hist_q.png", bbox_inches='tight')

    resi_xr_pct = resi_xr / np.abs(test_xr)
    resi_xr_pct.name = 'residuals_pct'
    bins = np.linspace(-0.1, 0.1, 100)
    resi_xr_hist = histogram(resi_xr_pct, bins=[bins], dim=['x', 'y']).sel(pol='q')

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(resi_xr_hist.residuals_pct_bin, resi_xr_hist.quantile(0.25, 'batch'), resi_xr_hist.quantile(0.75, 'batch'), alpha=0.5)
    ax.set_ylabel(r"Count")
    ax.set_xlabel(r"$\delta Q / |Q|$")
    ax.set_title("Histogram of test set residuals - Q")
    ax.axvline(x=0, color='k', alpha=0.5)
    ax.set_xticks([-0.1, 0., 0.1])
    fig.savefig(PLOT_DIR / 'vqvae-P-resi_hist_pct_q.png', bbox_inches='tight')
    
    # Plot power spectrum level reconstructions
    test_quantiles = test_xr_cl.quantile([0.25, 0.75], 'batch').sel(field='ee')
    recon_quantiles = reco_xr_cl.quantile([0.27, 0.75], 'batch').sel(field='ee')
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(test_quantiles.bandpowers, test_quantiles.isel(quantile=0), test_quantiles.isel(quantile=1), alpha=0.6, label='Test set')
    ax.fill_between(recon_quantiles.bandpowers, recon_quantiles.isel(quantile=0), recon_quantiles.isel(quantile=1), alpha=0.6, label='Reconstruction')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(frameon=False)
    #ax.set_ylim(1e-13, 1e-9)
    #ax.set_xlim(200, 1000)
    ax.set_ylabel(r'$C_\ell^{\rm EE}~[{\rm arb. units}]$')
    ax.set_xlabel(r'$\ell_b$')
    ax.set_title("Test set predictions of EE")
    fig.savefig(PLOT_DIR / 'vqvae-P-test_ee.png', bbox_inches='tight')

    test_quantiles = test_xr_cl.quantile([0.25, 0.75], 'batch').sel(field='bb')
    recon_quantiles = reco_xr_cl.quantile([0.27, 0.75], 'batch').sel(field='bb')
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(test_quantiles.bandpowers, test_quantiles.isel(quantile=0), test_quantiles.isel(quantile=1), alpha=0.6, label='Test set')
    ax.fill_between(recon_quantiles.bandpowers, recon_quantiles.isel(quantile=0), recon_quantiles.isel(quantile=1), alpha=0.6, label='Reconstruction')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(frameon=False)
    #ax.set_ylim(1e-13, 1e-9)
    #ax.set_xlim(200, 1000)
    ax.set_ylabel(r'$C_\ell^{\rm BB}~[{\rm arb. units}]$')
    ax.set_xlabel(r'$\ell_b$')
    ax.set_title("Test set predictions of BB")
    fig.savefig(PLOT_DIR / 'vqvae-P-test_bb.png', bbox_inches='tight')