import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read in training data from file
    DATA_DIR = os.path.realpath("/home/bthorne/projects/gan/began/data/processed")
    FILE_NAME = "HFI_SkyMap_545-field-Int_2048_R3.00_full.hdf5"
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

    with h5py.File(FILE_PATH, 'r') as f:
        group = f['Intensity/Planck/Renormalized']
        ntrain = group.attrs['NPATCHES']
        res = group.attrs['RES']
        train_data = group[...]
        lonras = group.attrs['LONRAS_HPIX']
        latras = group.attrs['LATRAS_HPIX']
        
    print("Training data consists of {:d} samples of a {:d} by {:d} image.".format(ntrain, res, res))
    print("Training data array has shape:", train_data.shape)
 
    # Plotting single longitude slice of Galaxy
    fig, axes = plt.subplots(int(np.sqrt(ntrain)), 1, sharex=True)
    plt.subplots_adjust(hspace=0)
    vmin = np.min(train_data)
    vmax = np.max(train_data) / 1000.
    for i, ax in enumerate(axes.flatten()[::-1]):
        ax.set_title("{:d}".format(i))
        ax.imshow(train_data[i], extent=[lonras[i][1], lonras[i][0], latras[i][0], latras[i][1]], vmin=vmin, vmax=vmax, origin='lower')
    
    # Plotting single latitude slice of
    fig, axes = plt.subplots(1, int(np.sqrt(ntrain)), sharey=True)
    plt.subplots_adjust(wspace=0)
    vmin = np.min(train_data)
    vmax = np.max(train_data) / 1000.
    for i, ax in enumerate(axes.flatten()[::-1]):
        i = 6 + i * int(np.sqrt(ntrain))
        ax.set_title("{:d}".format(i))
        ax.imshow(train_data[i], extent=[lonras[i][1], lonras[i][0], latras[i][0], latras[i][1]], vmin=vmin, vmax=vmax, origin='lower')
    plt.show()
