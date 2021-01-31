import tensorflow as tf 

from absl import flags
from absl import app
from absl import logging

import numpy as np
import healpy as hp 

from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager 
import os
import sys

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u 
from astropy.io import fits

from utils import FlatCutter, get_patch_centers, rotate

import requests

DATASET_DIRECTORY = "/global/cscratch1/sd/bthorne/dustvaeder/datasets"
GNILC_URL = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Dust-GNILC-F545_2048_R2.00.fits"
GNILC_FNAME = "COM_CompMap_Dust-GNILC-F545_2048_R2.00.fits"

STRING_TO_DATASET = {
    "MHD" : "/global/cscratch1/sd/bthorne/dustvaeder/datasets/MHD/R8_4pc/R8_4pc.npy",
    "GNILC" : "/global/cscratch1/sd/bthorne/dustvaeder/datasets/GNILC/GNILC.npy"
}

def load_npy_dataset(label):
    return np.load(STRING_TO_DATASET[label])

def load_dataset(label, batch_size=8):
    npy_dataset = load_npy_dataset(label)
    nsamples, npixx, npixy, npol = npy_dataset.shape
    input_shape = (npixx, npixy, npol)
    dataset = (tf.data.Dataset.from_tensor_slices(npy_dataset.astype(np.float32))
        .shuffle(nsamples)
        .map(tf.image.per_image_standardization)
        .map(tf.image.random_flip_left_right, num_parallel_calls=4)
        .map(tf.image.random_flip_up_down, num_parallel_calls=4)
        .map(rotate, num_parallel_calls=4)
        .batch(batch_size)
        .prefetch(1)
    )
    ntrain = int(0.7 * nsamples)
    nval = int(0.15 * nsamples)
    
    train_dataset = dataset.take(ntrain)
    val_dataset = dataset.skip(ntrain).take(nval)
    test_dataset = dataset.skip(ntrain + nval)
    dataset_info = {
        'ntrain': ntrain,
        'nval': nval, 
        'ntest': nval,
        'npixx': npixx, 
        'npixy': npixy, 
        'nchannels': npol, 
        'input_shape': (npixx, npixy, npol)
    }
    return train_dataset, val_dataset, test_dataset, dataset_info

def load_gnilc_dataset(batch_size):
    return

def fpath(data_dir, timestep, direction, cutoff):
    s = data_dir / f"R8_4pc_newacc.{timestep:04d}.{direction}.zmin{cutoff}.fits"
    return data_dir / f"R8_4pc_newacc.{timestep:04d}.{direction}.zmin{cutoff}.fits"

def read_fits(fpath):
    hdu = fits.open(fpath)
    arr = np.array([hdu[i].data for i in range(1, 4)])
    return arr

def prepare_MHD(source_dir, write_path, z_len=8., y=1., res_x=256, res_y=256):
    source_dir = Path("/global/cscratch1/sd/changgoo/flat-maps/R8_4pc/")
    write_path = Path(DATASET_DIRECTORY) / "MHD/R8_4pc/R8_4pc.npy"
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
    # metadata
    timesteps = np.arange(100, 675)
    directions = ["up", "dn"]
    cutoffs = [100, 200, 300]
    # set up coordinates
    x = np.linspace(-ang_x_deg / 2.0, ang_x_deg / 2.0, res_x)
    y = np.linspace(-ang_y_deg / 2.0, ang_y_deg / 2.0, res_y)
    logging.info("Reading ...")
    arr = np.empty((len(timesteps), len(directions), len(cutoffs), 3, 256, 256))
    for i, timestep in enumerate(tqdm(timesteps, desc="Preparing MHD dataset", leave=True, position=0)):
        for j, direction in enumerate(directions):
            for k, cutoff in enumerate(cutoffs):
                assert fpath(source_dir, timestep, direction, cutoff).exists()
                arr[i, j, k] = read_fits(fpath(source_dir, timestep, direction, cutoff))
    arr = np.moveaxis(arr, 3, -1) # move polarization axis to last place
    arr = arr.reshape((-1, 256, 256, 3)) # flatten first three dimensions
    logging.info("Saving ...")
    np.save(write_path, arr)
    return

def prepare_GNILC(res=256, gal_cut=5, step_size=4, ang_x=8, ang_y=8):
    source_path = Path(DATASET_DIRECTORY) / "GNILC" / GNILC_FNAME
    write_path = Path(DATASET_DIRECTORY) / "GNILC" / "GNILC.npy"
    logging.info(f"Source file: {source_path}")

    if not source_path.exists():
        req = requests.get(GNILC_URL, stream=True, allow_redirects=True)
        with open(source_path, 'wb+') as f:
            for chunk in req.iter_content(8096):
                if chunk:
                    f.write(chunk)
    HDU = fits.open(source_path)
    I_GNILC = hp.reorder(HDU[1].data['I'].astype(np.float32), inp=HDU[1].header['ORDERING'], out='RING')

    centers = get_patch_centers(gal_cut * u.deg, step_size * u.deg)
    lons, lats = zip(*centers)
    lons = np.array([lon.value for lon in lons])
    lats = np.array([lat.value for lat in lats])    

    cutter = FlatCutter(ang_x * u.deg, ang_y * u.deg, res, res)

    cut_maps = []
    for (lon, lat) in tqdm(centers, desc="Preparing GNILC dataset"):
        cut_maps.append(cutter.rotate_to_pole_and_interpolate(lon, lat, I_GNILC))
    cut_maps = np.array(cut_maps)
    logging.info("Saving ... ")
    np.save(write_path, cut_maps)
    return

def main(argv):
    del argv

    if FLAGS.dataset == "MHD":
        prepare_MHD()
    
    if FLAGS.dataset == "GNILC":
        prepare_GNILC()
    
    return

FLAGS = flags.FLAGS

if __name__ == "__main__":
    flags.DEFINE_enum(
        "dataset", 
        "MHD", 
        ["MHD", "GNILC"], 
        "Which datset to prepare.")
    app.run(main)