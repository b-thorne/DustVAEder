#!/home/bthorne/projects/gan/began/gpu-env/bin/python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys
import yaml

import click
from IPython.core import ultratb

import healpy as hp 
import numpy as np
import astropy.units as u
from astropy.io import fits
from began.tools import get_patch_centers, FlatCutter

import xarray as xa

_logger = logging.getLogger(__name__)

@click.command()
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    gal_cut = 40. * u.deg
    step_size = 10. * u.deg
    ang_x = 30 * u.deg
    ang_y = 30 * u.deg
    xres = 256
    yres = 256
    input_path = "/home/bthorne/projects/gan/began/data/raw/planck/COM_CompMap_Dust-GNILC-F545_2048_R2.00.fits"
    fullres_output_path = "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data/planck_cutouts_30deg.cdf"
    smoothed_output_path = "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data/planck_cutouts_1degsmo_30deg.cdf"

    centers = get_patch_centers(gal_cut, step_size)
    
    input_map = hp.read_map(input_path, field=0, dtype=np.float32, verbose=False)
 

   
    # cut out maps at each of the patch centers
    fc = FlatCutter(ang_x, ang_y, xres, yres)
    cut_maps = np.array([fc.rotate_to_pole_and_interpolate(lon, lat, input_map) for (lon, lat) in centers])

    dims = ['batch', 'x', 'y', 'pol']    
    coords = {
        'batch': np.arange(cut_maps.shape[0]),
        'x': np.arange(256),
        'y': np.arange(256),
        'pol': ['t']
    }
    cut_maps = xa.DataArray(cut_maps, coords=coords, dims=dims, name='cut_maps')
    cut_maps.to_netcdf(fullres_output_path, 'w')
  
    input_map = hp.smoothing(input_map, fwhm=np.pi/180. )
    fc = FlatCutter(ang_x, ang_y, xres, yres)
    cut_maps = np.array([fc.rotate_to_pole_and_interpolate(lon, lat, input_map) for (lon, lat) in centers])
    cut_maps = xa.DataArray(cut_maps, coords=coords, dims=dims, name='cut_maps')
    cut_maps.to_netcdf(smoothed_output_path, 'w')


if __name__ == '__main__':
    main()  