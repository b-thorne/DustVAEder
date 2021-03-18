#!/home/bthorne/projects/gan/began/gpu-env/bin/python
#SBATCH --job-name="vae-mhd-temperature"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH -t 24:00:00
import matplotlib
matplotlib.use('Agg')
import logging
from pathlib import Path
import sys
import yaml
import click
import os

import numpy as np
import xarray as xa
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
import random
import pymaster as nmt

import began
from began.logging import setup_vae_run_logging
from began import apply_per_image_standardization, apply_nmt_flat, make_square_mask, make_flat_bins
from contextlib import contextmanager 

_logger = logging.getLogger(__name__)

def fpath(timestep, direction, cutoff):
    return DATA_DIR / f"R8_4pc_newacc.{timestep:04d}.{direction}.zmin{cutoff}.fits"

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
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(seed: int, log_level: int):    
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    DATA_DIR = Path("/oasis/scratch/comet/bthorne/temp_project/flat-maps")
    WRITE_DIR = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_pol/data")

    zarr_record = WRITE_DIR / "mhd.zarr"

    if zarr_record.exists():
        data = xa.open_zarr(str(DATA_DIR / "mhd.zarr"))
        
    else:
        timesteps = np.arange(100, 675)
        directions = ['up', 'dn']
        cutoffs = [100, 200, 300]
        x = np.arange(256)
        y = np.arange(256)
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
        data = xa.Dataset({
            'data': (('time', 'direc', 'zmin', 'pol', 'x', 'y'), arr) 
            }, coords={
                'time': timesteps,
                'direc': directions,
                'zmin': cutoffs,
                'pol': ['t', 'q', 'u'],
                'x': x,
                'y': y,
            })
        data.to_zarr(str(WRITE_DIR / "mhd.zarr"))
    
    # load from dask array
    data.load()
    # Extract the polarization dimensions, stack the zmin, direc, and time dimensions, 
    # and move batch to the first dimension, and pol to the last dimension
    pol = data['data'].sel(pol=['q', 'u']).stack(batch=['zmin', 'time', 'direc']).transpose('batch', ..., 'pol')
    pol = pol.isel(batch=np.random.randint(0, 3000, 200))
    # Apply a log normalization, by shifting the minimum pixel value to 1 first.
    # Then apply also a standardization
    log_pol = np.log10(pol - pol.values.min() + 1)
    log_normed_pol = apply_per_image_standardization(log_pol)

    # Set up power spectrum calculation
    nx = 256
    ang = np.radians(29.)
    apo_mask = make_square_mask(nx, nx, ang)
    b = make_flat_bins(ang, nx, 8)

    f2 = nmt.NmtFieldFlat(ang, ang, apo_mask, np.random.randn(2, nx, nx), purify_b=True, purify_e=True)
    wsp22 = nmt.NmtWorkspaceFlat()
    wsp22.compute_coupling_matrix(f2, f2, b)

    # Calculate power spectra of the three different normalizations (raw, log, log-std)
    pol_cl = apply_nmt_flat(pol, apo_mask, ang, b, wsp22=wsp22)
    log_pol_cl = apply_nmt_flat(log_pol, apo_mask, ang, b, wsp22=wsp22)
    log_normed_pol_cl = apply_nmt_flat(log_normed_pol, apo_mask, ang, b, wsp22=wsp22)

    # sample random indices to create test and train split
    ntest = 100
    indices = set(range(pol.batch.size))
    test_idx = sorted(random.sample(indices, ntest))
    train_idx = sorted(indices - set(test_idx))
    
    # In WRITE_DIR save the various normalized datasets, and their spectra
    with working_directory(WRITE_DIR):
        np.save("mhd_pol_test.npy", pol.isel(batch=test_idx).values)
        np.save("mhd_pol_train.npy", pol.isel(batch=train_idx).values)

        pol_cl.isel(batch=test_idx).reset_index('batch').to_netcdf("mhd_pol_test_cl.cdf")
        pol_cl.isel(batch=train_idx).reset_index('batch').to_netcdf("mhd_pol_train_cl.cdf")

        np.save("mhd_log_pol_test.npy", log_pol.isel(batch=test_idx).values)
        np.save("mhd_log_pol_train.npy", log_pol.isel(batch=train_idx).values)

        log_pol_cl.isel(batch=test_idx).reset_index('batch').to_netcdf("mhd_log_pol_cl_test.cdf")
        log_pol_cl.isel(batch=train_idx).reset_index('batch').to_netcdf("mhd_log_pol_cl_train.cdf")

        np.save("mhd_log_std_pol_test.npy", log_pol.isel(batch=test_idx).values)
        np.save("mhd_log_std_pol_train.npy", log_pol.isel(batch=train_idx).values)

        log_normed_pol_cl.isel(batch=test_idx).reset_index('batch').to_netcdf("mhd_log_pol_cl_test.cdf")
        log_normed_pol_cl.isel(batch=train_idx).reset_index('batch').to_netcdf("mhd_log_pol_cl_train.cdf")

if __name__ == '__main__':
    main()

