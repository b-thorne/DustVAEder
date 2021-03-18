#!/home/bthorne/projects/gan/began/gpu-env/bin/python
#SBATCH --job-name="vae-mhd-temp-summarize"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=25G
#SBATCH -t 24:00:00
import matplotlib
matplotlib.use('Agg')
import logging
from pathlib import Path
import sys
import yaml
import click

import numpy as np
import xarray as xa
import tensorflow as tf

import began
from began.logging import setup_vae_run_logging
from began import compute_apply_gradients


_logger = logging.getLogger(__name__)

def check_directories(root_dir, name):
    root_dir = Path(root_dir).absolute()
    try:
        assert root_dir.exists()
    except AssertionError:
        raise AssertionError("Root dir must exist")
    directories = [root_dir]
    for sub in ['models', 'plots', 'data']:
        subdir = root_dir / sub / name
        subdir.mkdir(exist_ok=True)
        directories.append(subdir)
    return directories

def load_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg['data'], cfg['training'], cfg['architecture'], Path(cfg['name'])

@click.command()
@click.option('--root_dir', 'root_dir', required=True,
              type=click.Path(exists=True), help='Path to directory containing data and configurations for this run')
@click.option('--cfg_path', 'cfg_path', required=True,
              type=click.Path(exists=True), help='Path to yaml configuration file')
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(root_dir: Path, cfg_path: Path, seed: int, log_level: int):    
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Read configuration file containing hyperparameter specficcations.
    data_cfg, train_cfg, architecture_cfg, name = load_cfg(cfg_path)
    # Set up directory structure for saved models and plots.
    (root_dir, model_dir, plot_dir, data_dir) = check_directories(root_dir, name)
    model_path = model_dir / name.with_suffix('.h5')
    training_samples_path = data_dir / name.with_suffix('.zarr')

    # Read in MHD data and convert to testing and training
    # tensorflow datasets.
    data_dir = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data")
    train_fpath = data_dir / data_cfg['train_file']
    test_fpath = data_dir / data_cfg['test_file']

    train_images = xa.open_dataarray(train_fpath).transpose('batch', ..., 'pol').values
    dset = tf.data.Dataset.from_tensor_slices(train_images)
    dset = dset.shuffle(train_images.shape[0])
    dset = dset.batch(train_cfg['batch_size'])

    test_images = xa.open_dataarray(test_fpath).transpose('batch', ..., 'pol').values
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
    test_dataset = test_dataset.batch(train_cfg['batch_size'])

    # Define the optimizer to use.
    optimizer = tf.keras.optimizers.Adam(beta_1=train_cfg['beta_1'], learning_rate=train_cfg['learning_rate'])
    
    # Build the VAE model
    model = began.CVAE(architecture_cfg['lat_dim'], architecture_cfg['inference_kernels'], architecture_cfg['generative_kernels'])

    # Set up an xarray dataset to log samples of the network's output at each
    # epoch of training. We hardcode that nine points in the latent space are
    # sampled. The generated samples are saved after training, but the state of
    # the model is not saved.
    dims = ["epoch", "z", "x", "y", "pol"]
    coords = {
        "epoch": np.arange(train_cfg['epochs']),
        "z": np.arange(9),
        "x": np.arange(256),
        "y": np.arange(256),
        "pol": ['t'],
    }
    data = np.zeros((train_cfg['epochs'], 9, 256, 256, 1))
    samples_viz = xa.Dataset({'samples': (dims, data)}, coords=coords)
    z_viz = tf.random.normal(shape=[9, architecture_cfg['lat_dim']])

    # We now begin the training process. In each epoch, we loop over the 
    # training set in batches. After each batch, the gradients are computed
    # and updated. After 10 epochs, the loss is evaluated and logged, and
    # samples from the latent space evaluated with the current weights.
    for epoch in range(train_cfg['epochs']):
        _logger.info(f"Epoch: {epoch:05d} / {train_cfg['epochs']:05d}")
        for train_x in dset: 
            compute_apply_gradients(model, train_x, optimizer)

        # generate samples after training for an epoch
        samples_viz['samples'][dict(epoch=epoch)] = model.decode(z_viz)
        
        # after every 10 epochs save summary statistics
        if epoch % 10 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(began.vae.compute_loss(model, test_x))
            elbo = - loss.result()
            _logger.info(f"ELBO at epoch {epoch:05d} is {elbo:.03f}")

            #with summary_writer.as_default():
            #    tf.summary.scalar('elbo', elbo, step=epoch)
            facet_grid = samples_viz['samples'].sel(epoch=epoch, pol='t').plot(x='x', y='y', col='z', col_wrap=3, cmap='cividis', robust=True)
            facet_grid.fig.savefig(plot_dir / f"sample_epoch{epoch:04d}.png")
    model.save_weights(str(model_path))
    samples_viz.to_zarr(training_samples_path, 'w')

if __name__ == '__main__':
    main()

