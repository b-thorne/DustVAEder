""" This file plots summary statistics of the MHD training
data. Thisincludes:

1. Example plots of cut-outs displayed by direction and zmin.
2. Power spectrum distribution
3. Pixel value histograms of both temperature and polarization.
4. Non-Gaussian statistics of the temperature.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import seaborn as sns
import xarray as xa
from xhistogram.xarray import histogram

from began import apply_per_image_standardization

plt.style.use("seaborn-white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = [5, 5]

if __name__ == "__main__":

    PLOT_DIR = Path("/home/bthorne/projects/gan/began/reports/figures/slides/train")

    data_fpath = "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data/mhd.zarr"
    data = xa.open_zarr(data_fpath)["data"]
    data.load()
    data.attrs["units"] = r"$T~{\rm MJy/sr}$"

    #
    # Map Plotting
    #
    cbar_kw = {"label": r"$T~[{\rm arb. units}]$"}
    fg = data.sel(
        time=np.random.randint(100, 500, 3), pol="t", direc=["up", "dn"], zmin=200
    ).plot(
        x="x",
        y="y",
        col="time",
        row="direc",
        cmap="cividis",
        cbar_kwargs=cbar_kw,
        robust=True,
    )
    fg.fig.savefig(PLOT_DIR / "direc-time-t.png", bbox_inches="tight")

    cbar_kw = {"label": r"$Q~[{\rm arb. units}]$"}
    fg = data.sel(
        time=np.random.randint(100, 500, 3), pol="q", direc=["up", "dn"], zmin=200
    ).plot(
        x="x",
        y="y",
        col="time",
        row="direc",
        cmap="cividis",
        cbar_kwargs=cbar_kw,
        robust=True,
    )
    fg.fig.savefig(PLOT_DIR / "direc-time-q.png", bbox_inches="tight")

    cbar_kw = {"label": r"$U~[{\rm arb. units}]$"}
    fg = data.sel(
        time=np.random.randint(100, 500, 3), pol="u", direc=["up", "dn"], zmin=200
    ).plot(
        x="x",
        y="y",
        col="time",
        row="direc",
        cmap="cividis",
        cbar_kwargs=cbar_kw,
        robust=True,
    )
    fg.fig.savefig(PLOT_DIR / "direc-time-u.png", bbox_inches="tight")

    cbar_kw = {"label": r"$X~[{\rm arb. units}]$"}
    plot_tqu = apply_per_image_standardization(
        data.sel(time=200, pol=["t", "q", "u"], direc=["up", "dn"], zmin=200)
    )
    fg = plot_tqu.plot(
        x="x",
        y="y",
        col="pol",
        row="direc",
        cmap="cividis",
        cbar_kwargs=cbar_kw,
        robust=True,
    )
    fg.fig.savefig(PLOT_DIR / "pol-vs-direc.png", bbox_inches="tight")

    #
    # Histogram Plotting
    #
    data_fpath = "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data/mhd_ntrain-3150.cdf"
    data = xa.open_dataarray(data_fpath)

    bins = np.linspace(-0.01, 0.3, 100)
    bins = np.logspace(-3, 0, 100)
    hist = histogram(data.sel(pol="t"), bins=[bins], dim=["x", "y"])

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        hist.data_bin,
        hist.quantile(0.25, "batch"),
        hist.quantile(0.75, "batch"),
        alpha=0.5,
    )
    ax.set_ylabel(r"Frequency")
    ax.set_xlabel(r"$T~{\rm [arb. units]}$")
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-hist_t.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    sns
    ax.fill_between(
        hist.data_bin,
        hist.quantile(0.25, "batch"),
        hist.quantile(0.75, "batch"),
        alpha=0.5,
    )
    ax.set_ylabel(r"Frequency")
    ax.set_xlabel(r"$T~{\rm [arb. units]}$")
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-hist_t.png", bbox_inches="tight")

    bins = np.linspace(-0.01, 0.025)
    hist = histogram(data.sel(pol="q"), bins=[bins], dim=["x", "y"])
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        hist.data_bin,
        hist.quantile(0.25, "batch"),
        hist.quantile(0.75, "batch"),
        alpha=0.5,
    )
    ax.set_ylabel(r"Frequency")
    ax.set_xlabel(r"$Q~{\rm [arb. units]}$")
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-hist_q.png", bbox_inches="tight")

    bins = np.linspace(-0.02, 0.02)
    hist = histogram(data.sel(pol="u"), bins=[bins], dim=["x", "y"])
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        hist.data_bin,
        hist.quantile(0.25, "batch"),
        hist.quantile(0.75, "batch"),
        alpha=0.5,
    )
    ax.set_ylabel(r"Frequency")
    ax.set_xlabel(r"$U~{\rm [arb. units]}$")
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-hist_u.png", bbox_inches="tight")

    #
    # Plotting Power Spectra
    #
    data_fpath = "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data/mhd_cl_ntrain-3150.cdf"
    train_cl = xa.open_dataarray(data_fpath)
    cl_stats = train_cl.quantile([0.25, 0.75], "batch")
    ratio = train_cl.sel(field="bb") / train_cl.sel(field="ee")

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        cl_stats.bandpowers,
        cl_stats.sel(field="tt", quantile=0.25),
        cl_stats.sel(field="tt", quantile=0.75),
        alpha=0.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell_b$")
    ax.set_ylabel(r"$C_{\ell_b}^{\rm TT}$")
    # ax.set_xlim(150, None)
    ax.set_ylim(None, 1e-6)
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-cl_tt.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        cl_stats.bandpowers,
        cl_stats.sel(field="ee", quantile=0.25),
        cl_stats.sel(field="ee", quantile=0.75),
        alpha=0.5,
        label="EE",
    )
    ax.fill_between(
        cl_stats.bandpowers,
        cl_stats.sel(field="bb", quantile=0.25),
        cl_stats.sel(field="bb", quantile=0.75),
        alpha=0.5,
        label="BB",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell_b$")
    ax.set_ylabel(r"$C_{\ell_b}$")
    # ax.set_xlim(150, None)
    ax.set_ylim(None, 1e-8)
    ax.legend(frameon=False)
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-cl_eebb.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        cl_stats.bandpowers,
        ratio.quantile(0.25, "batch"),
        ratio.quantile(0.75, "batch"),
    )
    ax.set_xlabel(r"$\ell_b$")
    ax.set_ylabel(r"$C^{\rm EE}_{\ell_b}/C_{\ell_b}^{\rm BB}$")
    # ax.set_xlim(150, None)
    ax.set_ylim(0, 1.1)
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-cl_eebb_ratio.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(
        cl_stats.bandpowers,
        cl_stats.sel(field="tt", quantile=0.25),
        cl_stats.sel(field="tt", quantile=0.75),
        alpha=0.5,
        label="TT",
    )
    ax.fill_between(
        cl_stats.bandpowers,
        cl_stats.sel(field="ee", quantile=0.25),
        cl_stats.sel(field="ee", quantile=0.75),
        alpha=0.5,
        label="EE",
    )
    ax.fill_between(
        cl_stats.bandpowers,
        cl_stats.sel(field="bb", quantile=0.25),
        cl_stats.sel(field="bb", quantile=0.75),
        alpha=0.5,
        label="BB",
    )
    ax.set_xlabel(r"$\ell_b$")
    ax.set_ylabel(r"$C_{\ell_b}$")
    # ax.set_xlim(150, 1500)
    ax.set_ylim(1e-14, 1e-6)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("Training set power spectra - interquartile range")
    fig.savefig(PLOT_DIR / "mhd_ntrain-3150-cl_tteebb_ratio.png", bbox_inches="tight")

