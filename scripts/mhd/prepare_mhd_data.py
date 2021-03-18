#!/home/bthorne/projects/gan/began/gpu-env/bin/python
# SBATCH --job-name="vae-mhd-temperature"
# SBATCH -p gpu-shared
# SBATCH --gres=gpu:p100:1
# SBATCH --ntasks-per-node=7
# SBATCH --mem=25G
# SBATCH -t 24:00:00
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import click
import numpy as np
import xarray as xa
from astropy.io import fits

import began
from began import apply_nmt_flat, apply_per_image_standardization

_logger = logging.getLogger(__name__)


def log(arr):
    return np.log10(arr)


def read_fits(fpath):
    hdu = fits.open(fpath)
    arr = np.array([hdu[i].data for i in range(1, 4)])
    return arr


@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


@click.command()
@click.option("--seed", "seed", type=int, default=1234321)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(seed: int, log_level: int):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    DATA_DIR = Path(
        "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/data/flat-maps"
    )

    def fpath(timestep, direction, cutoff):
        return DATA_DIR / f"R8_4pc_newacc.{timestep:04d}.{direction}.zmin{cutoff}.fits"

    WRITE_DIR = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/data")

    cdf_record = WRITE_DIR / "mhd.cdf"

    if cdf_record.exists():
        data = xa.open_dataarray(str(WRITE_DIR / "mhd.cdf"))

    else:
        # simulation vertical height
        z_len = 8.0
        z_int = z_len / 2.0
        # width of simulation volume
        y = 1.0
        # linear angle subtended by patch side as seen
        # by observer
        theta = np.degrees(2.0 * np.arctan(y / 2.0 / z_int))
        ang_x_deg = theta
        ang_x_rad = np.radians(theta)
        ang_y_deg = theta
        ang_y_rad = np.radians(theta)
        print(f"{ang_x_deg:.02f} by {ang_y_deg:.02f} patch")
        # metadata
        timesteps = np.arange(100, 675)
        directions = ["up", "dn"]
        cutoffs = [100, 200, 300]
        res_x = 256
        res_y = 256
        # set up coordinates
        x = np.linspace(-ang_x_deg / 2.0, ang_x_deg / 2.0, res_x)
        y = np.linspace(-ang_y_deg / 2.0, ang_y_deg / 2.0, res_y)
        for timestep in timesteps:
            for direction in directions:
                for cutoff in cutoffs:
                    assert fpath(timestep, direction, cutoff).exists()
        arr = np.empty((len(timesteps), len(directions), len(cutoffs), 3, 256, 256))
        # read in all the data, there are thousands of files, this takes ~ 20 seconds
        for t, timestep in enumerate(timesteps):
            if t % 10 == 0:
                print(t)
            for d, direction in enumerate(directions):
                for c, cutoff in enumerate(cutoffs):
                    arr[t, d, c] = read_fits(fpath(timestep, direction, cutoff))
        data = xa.DataArray(
            arr,
            coords={
                "time": timesteps,
                "direc": directions,
                "zmin": cutoffs,
                "pol": ["t", "q", "u"],
                "x": x,
                "y": y,
            },
            dims=["time", "direc", "zmin", "pol", "x", "y"],
        )
        data.attrs["res_x"] = res_x
        data.attrs["res_y"] = res_y
        data.attrs["ang_x"] = ang_x_rad
        data.attrs["ang_y"] = ang_y_rad
        data.to_netcdf(str(WRITE_DIR / "mhd.cdf"))

    # Stack the zmin, direc, and time dimensions, and move pol to the last dimension
    images = data.stack(batch=("zmin", "direc", "time")).transpose("batch", ..., "pol")

    # sample random indices to create test and train split by sampling batch dimension
    # without replacement
    np.random.seed(seed)
    ntest = 300
    ntrain = images.batch.size - ntest
    indices = set(range(images.batch.size))
    test_idx = set(np.random.choice(images.batch.size, size=ntest, replace=False))
    train_idx = indices - test_idx

    # check to make sure no elements are assigned to both test and train
    try:
        assert not bool((test_idx & train_idx))
    except AssertionError:
        raise AssertionError("Test and train indices share at least one item")

    test_idx = sorted(test_idx)
    train_idx = sorted(train_idx)

    try:
        assert len(test_idx) == ntest
        assert len(train_idx) == ntrain
    except AssertionError:
        raise AssertionError("Size of test/train split different to expected.")

    images.attrs["units"] = r"$T~{\rm MJy/sr}$"

    # take the test and train subsets, unstack for rest of operations
    test = images.isel(batch=test_idx).reset_index("batch")
    train = images.isel(batch=train_idx).reset_index("batch")

    # set up power spectrum estimation
    with working_directory(WRITE_DIR):
        _logger.debug("Working in {WRITE_DIR}")
        # -------------------
        # Split and save RAW data
        # -------------------
        test.to_netcdf(f"mhd_ntest-{ntest:04d}.cdf")

        test_cl = apply_nmt_flat(test)
        test_cl.to_netcdf(f"mhd_cl_ntest-{ntest:04d}.cdf")

        train.to_netcdf(f"mhd_ntrain-{ntrain:04d}.cdf")

        train_cl = apply_nmt_flat(train)
        train_cl.to_netcdf(f"mhd_cl_ntrain-{ntrain:04d}.cdf")
        sys.exit()
        # -------------------
        # Apply STANDARDIZATION of mean 0 and std dev 1, split, and save
        # -------------------
        test_std = apply_per_image_standardization(test)
        test_std.to_netcdf(f"mhd_std_ntest-{ntest:04d}.cdf")

        test_std_cl = apply_nmt_flat(test_std)
        test_std_cl.to_netcdf(f"mhd_std_cl_ntest-{ntest:04d}.cdf")

        train_std = apply_per_image_standardization(train)
        train_std.to_netcdf(f"mhd_std_ntrain-{ntrain:04d}.cdf")

        train_std_cl = apply_nmt_flat(train_std)
        train_std_cl.to_netcdf(f"mhd_std_cl_ntrain-{ntrain:04d}.cdf")

        # -------------------
        # Apply log base 10, split, and save
        # -------------------
        test = log(test)
        test.to_netcdf(f"mhd_log_ntest-{ntest:04d}.cdf")

        test_cl = apply_nmt_flat(test)
        test_cl.to_netcdf(f"mhd_log_cl_ntest-{ntest:04d}.cdf")

        train = log(train)
        train.to_netcdf(f"mhd_log_ntrain-{ntrain:04d}.cdf")

        train_cl = apply_nmt_flat(train)
        train_cl.to_netcdf(f"mhd_log_cl_ntrain-{ntrain:04d}.cdf")

        # -------------------
        # Apply log base 10, standardize to interval 0 - 1, split, and save
        # -------------------
        trainx_normed = (train - train.min()) / (train.max() - train.min())
        train_normed.to_netcdf(f"mhd_log_interval01_ntrain-{ntrain:04d}.cdf")

        train_cl = apply_nmt_flat(train_normed)
        train_cl.to_netcdf(f"mhd_log_interval01_cl_ntrain-{ntrain:04d}.cdf")

        test_normed = (test - test.min()) / (test.max() - test.min())
        test_normed.to_netcdf(f"mhd_log_interval01_ntest-{ntest:04d}.cdf")

        test_cl = apply_nmt_flat(test_normed)
        test_cl.to_netcdf(f"mhd_log_interval01_cl_ntest-{ntest:04d}.cdf")

        # -------------------
        # Apply log base 10, standardize to mean 0 and std dev 1, split, and save
        # -------------------
        test = apply_per_image_standardization(test)
        test.to_netcdf(f"mhd_log_std_ntest-{ntest:04d}.cdf")

        test_cl = apply_nmt_flat(test)
        test_cl.to_netcdf(f"mhd_log_std_cl_ntest-{ntest:04d}.cdf")

        train = apply_per_image_standardization(train)
        train.to_netcdf(f"mhd_log_std_ntrain-{ntrain:04d}.cdf")

        train_cl = apply_nmt_flat(train)
        train_cl = train_cl.to_netcdf(f"mhd_log_std_cl_ntrain-{ntrain:04d}.cdf")


if __name__ == "__main__":
    main()
