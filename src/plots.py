import tensorflow as tf 
import numpy as np

import os, sys

from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import cosmoplotian
import cosmoplotian.colormaps
import matplotlib as mpl
#string_cmap = "div yel grn"
string_cmap = "RdYlBu"
cmap = mpl.cm.get_cmap(string_cmap)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmap(0.2), "k", "red"]) 
#plt.rcParams['text.usetex'] = True
from matplotlib import patheffects
from matplotlib import text

def outline_text(ax):
    """Add a white outline to all text to make it stand out from the
    background.
    """
    effects = [patheffects.withStroke(linewidth=2, foreground='w')]
    for artist in ax.findobj(text.Text):
        artist.set_path_effects(effects)

def make_prior_sample_plot(arr1, title=""):
    # assume direct output of model prediction with shape (n_batch, x, y, n_channel)
    assert arr1.ndim == 4

    # assume only one channel
    assert arr1.shape[-1] == 1

    # assume 4 samples
    assert arr1.shape[0] == 3
    
    fig, axes = plt.subplots(ncols=3, nrows=1, dpi=150, figsize=(5, 5./3.), sharex=True, sharey=True)

    kwds = {
        'extent': [-4, 4, -4, 4],
        'aspect': 'auto',
        'interpolation': 'nearest',
        'origin': 'lower',
        'cmap': 'RdYlBu',
        'vmin': -3,
        'vmax': 3
    }
    
    fig.suptitle(title)
    axes[0].imshow(arr1[0, :, :, 0], **kwds)
    axes[1].imshow(arr1[1, :, :, 0], **kwds)
    img = axes[2].imshow(arr1[2, :, :, 0], **kwds)

    for ax in axes.flatten():
        ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
        ax.yaxis.set_ticks([-4, -2, 0, 2, 4])
        ax.grid(linestyle='--', linewidth=0.5, color='k', alpha=0.5)
        ax.tick_params(axis='both', direction='in')
        ax.tick_params(axis='x', rotation=90)
        outline_text(ax)

    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.89, wspace=0.06, hspace=0.06)

    cbax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(img, cax=cbax)

    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_label("Dimensionless log scaled emission")
    return fig, axes

def make_prediction_plot_with_residuals(arr1, arr2, title, vlims=[-3, 3]):
    # assume direct output of model prediction with shape (n_batch, x, y, n_channel)
    assert arr1.ndim == 4
    assert arr2.ndim == 4
    # assume only one channel
    #assert arr1.shape[-1] == 1
    #assert arr2.shape[-1] == 1
    # assume 4 samples
    #assert arr1.shape[0] == 3
    #assert arr2.shape[0] == 3
    
    residuals = arr1 - arr2
    
    fig, axes = plt.subplots(ncols=3, nrows=3, dpi=150, figsize=(5, 5), sharex=True, sharey=True)

    kwds = {
        'extent': [-4, 4, -4, 4],
        'aspect': 'auto',
        'interpolation': 'nearest',
        'origin': 'lower',
        'cmap': 'RdYlBu',
        'vmin': vlims[0],
        'vmax': vlims[-1]
    }
    
    fig.suptitle(title)
    axes[0, 0].imshow(arr1[0, :, :, 0], **kwds)
    axes[0, 0].set_ylabel("$y~{\\rm [^\\circ]}$")
    axes[0, 1].imshow(arr1[1, :, :, 0], **kwds)
    axes[0, 2].imshow(arr1[2, :, :, 0], **kwds)
    axes[1, 0].imshow(arr2[0, :, :, 0], **kwds)
    axes[1, 0].set_ylabel("$y~{\\rm [^\\circ]}$")
    axes[2, 0].set_ylabel("$y~{\\rm [^\\circ]}$")
    
    img = axes[1, 1].imshow(arr2[1, :, :, 0], **kwds)
    img = axes[1, 2].imshow(arr2[2, :, :, 0], **kwds)
    
    axes[2, 0].imshow(residuals[0, :, :, 0], **kwds)
    axes[2, 0].set_xlabel("$x~{\\rm [^\\circ]}$")
    axes[2, 1].imshow(residuals[1, :, :, 0], **kwds)
    axes[2, 1].set_xlabel("$x~{\\rm [^\\circ]}$")
    axes[2, 2].imshow(residuals[2, :, :, 0], **kwds)
    axes[2, 2].set_xlabel("$x~{\\rm [^\\circ]}$")
    
    for ax in axes.flatten():
        ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
        ax.yaxis.set_ticks([-4, -2, 0, 2, 4])
        ax.grid(linestyle='--', linewidth=0.5, color='k', alpha=0.5)
        ax.tick_params(axis='both', direction='in')
        ax.tick_params(axis='x', rotation=90)
        outline_text(ax)

    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.89, wspace=0.06, hspace=0.06)

    cbax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(img, cax=cbax)

    #cbar.set_ticks(np.linspace(*vlims, sum([abs(f) for f in vlims])+1))
    cbar.set_label(r"{\rm Dimensionless log scaled emission}")
    return fig, axes

def make_semantic_sequence_plot(arr, title):
    # assume direct output of model prediction with shape (n_batch, x, y, n_channel)
    assert arr.ndim == 4
    # assume only one channel
    assert arr.shape[-1] == 1
    # assume 4 samples
    assert arr.shape[0] == 12
    
    #fig, axes = plt.subplots(ncols=4, nrows=3, dpi=150, figsize=(6, 3), sharex=True, sharey=True)
    nrows = 3
    ncols = 4
    fig = plt.figure(constrained_layout=True, figsize=(ncols * 1.55, nrows * 1.5))

    imshow_kwds = {
        'extent': [-4, 4, -4, 4],
        'aspect': 'auto',
        'interpolation': 'nearest',
        'origin': 'lower',
        'cmap': 'div yel grn',
        'vmin': -3,
        'vmax': 3
    }


    height_ratios = [1] * nrows
    width_ratios = [1] * ncols + [0.1]

    msgs = [
        r"$\mathbf{x}_1~(\lambda=0)$",
        r"$\lambda=1/11$",
        r"$\lambda=2/11$",
        r"$\lambda=3/11$",
        r"$\lambda=4/11$",
        r"$\lambda=5/11$",
        r"$\lambda=6/11$",
        r"$\lambda=7/11$",
        r"$\lambda=8/11$",
        r"$\lambda=9/11$",
        r"$\lambda=10/11$",
        r"$\mathbf{x}_2~(\lambda=1)$"
    ]
    
    spec = fig.add_gridspec(ncols=ncols + 1, nrows=nrows, width_ratios=width_ratios, height_ratios=height_ratios)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(spec[i, j])
            ax.set_box_aspect(1)
            img = ax.imshow(arr[k, :, :, 0], **imshow_kwds)
            ax.grid(linestyle='--', linewidth=0.5, color='k', alpha=0.5)
            ax.tick_params(axis='both', direction='in')
            ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
            ax.yaxis.set_ticks([-4, -2, 0, 2, 4])
            ax.set_title(msgs[k])
            if i == nrows - 1:
                ax.set_xlabel(r"$x~{\rm [^\circ]}$")
                
            if j == 0:
                ax.set_ylabel(r"$y~{\rm [^\circ]}$")  
            k += 1
            if j != 0:
                ax.yaxis.set_ticklabels([])
            if i != 2:
                ax.xaxis.set_ticklabels([])
    cbax = fig.add_subplot(spec[:, -1])
    cbar = fig.colorbar(img, cax=cbax)

    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_label(r"{\rm Dimensionless log-scaled emission}")
    return fig, ax

def make_missing_window_plot(arr, title):
    # assume direct output of model prediction with shape (n_batch, x, y, n_channel)
    assert arr.ndim == 4
    # assume only one channel
    assert arr.shape[-1] == 1
    # assume 4 samples
    assert arr.shape[0] == 9
    
    fig, axes = plt.subplots(ncols=3, nrows=3, dpi=150, figsize=(7, 6), sharex=True, sharey=True)

    kwds = {
        'extent': [-4, 4, -4, 4],
        'aspect': 'auto',
        'interpolation': 'nearest',
        'origin': 'lower',
        'cmap': 'div yel grn',
        'vmin': -3,
        'vmax': 3
    }
    fig.suptitle(title)
    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='gray')
    
    axes[0, 0].imshow(arr[0, :, :, 0], **kwds)
    axes[0, 0].set_ylabel(r"$y~{\rm [^\circ]}$")
    axes[0, 1].imshow(arr[1, :, :, 0], **kwds)
    axes[0, 2].imshow(arr[2, :, :, 0], **kwds)
    
    axes[1, 0].imshow(arr[3, :, :, 0], **kwds)
    axes[1, 0].set_ylabel(r"$y~{\rm [^\circ]}$")
    axes[1, 1].imshow(arr[4, :, :, 0], **kwds)
    axes[1, 2].imshow(arr[5, :, :, 0], **kwds)
    
    axes[2, 0].imshow(arr[6, :, :, 0], **kwds)
    axes[2, 0].set_ylabel(r"$y~{\rm [^\circ]}$")
    axes[2, 0].set_xlabel(r"$x~{\rm [^\circ]}$")
    axes[2, 1].imshow(arr[7, :, :, 0], **kwds)
    axes[2, 1].set_xlabel(r"$x~{\rm [^\circ]}$")
    img = axes[2, 2].imshow(arr[8, :, :, 0], **kwds)
    axes[2, 2].set_xlabel(r"$x~{\rm [^\circ]}$")

    for ax in axes.flatten():
        ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
        ax.yaxis.set_ticks([-4, -2, 0, 2, 4])
        ax.grid(linestyle='--', linewidth=0.5, color='k', alpha=0.5)
        ax.tick_params(axis='both', direction='in')
        ax.tick_params(axis='x', rotation=90)
        outline_text(ax)

    msgs = [
        r"$x_1$",
        r"$x_2$",
        r"$x_3$",
        r"$A_1 x_1 + n$",
        r"$A_2 x_2 + n$",
        r"$A_3 x_3 + n$",
        r"$g_\phi(z_1^{\rm MAP})$",
        r"$g_\phi(z_2^{\rm MAP})$",
        r"$g_\phi(z_3^{\rm MAP})$",
    ]
    for ax, msg in zip(axes.flatten(), msgs):
        ax.annotate(msg, xy=(0.03, 0.91), xycoords='axes fraction')
        outline_text(ax)
        
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.89, wspace=0.06, hspace=0.06)

    cbax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(img, cax=cbax)

    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_label(r"{\rm Dimensionless log-scaled emission}")
    return fig, ax

def history_plot(history, keys):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for k in keys:
        l1, = ax.plot(history[k], label=k)
        ax.plot(history['val_'+k], color=l1.get_color(), linestyle="--")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    return fig, ax