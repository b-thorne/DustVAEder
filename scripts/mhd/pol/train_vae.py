#!/home/bthorne/projects/gan/began/gpu-env/bin/python
#SBATCH --job-name="vae-mhd-temperature"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH -t 08:00:00
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

def check_directories(root_dir, subdir, name):
    subdir = Path(root_dir).absolute() / subdir
    try:
        assert root_dir.exists()
    except AssertionError:
        raise AssertionError("Root dir must exist")
    directories = [root_dir]
    for sub in ['models', 'plots', 'data']:
        subsubdir = subdir / sub / name
        subsubdir.mkdir(exist_ok=True, parents=True)
        directories.append(subsubdir)
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
    
    root_dir = Path(root_dir).absolute()
    assert root_dir.exists()

    NX = 256
    NY = 256
    NCHANNELS = 2

    # Read configuration file containing hyperparameter specficcations.
    data_cfg, train_cfg, architecture_cfg, name = load_cfg(cfg_path)
    _logger.debug(f"\n data_cfg: {data_cfg} \n train_cfg: {train_cfg} \n architecture_cfg: {architecture_cfg}")
    # Set up directory structure for saved models and plots.
    _logger.debug(f"\n root_dir {root_dir}")
    (root_dir, model_dir, plot_dir, data_dir) = check_directories(root_dir, "mhd_pol", name)
    _logger.debug(f"\n root_dir {root_dir} \n model_dir {model_dir} \n plot_dir {plot_dir} \n data_dir {data_dir}")

    model_path = model_dir / name.with_suffix('.h5')
    training_samples_path = root_dir / name.with_suffix('.zarr')

    # Read in MHD data and convert to testing and training tensorflow datasets.
    train_images = xa.open_zarr(str(root_dir / "mhd.zarr"))
    train_images = np.array(train_images['data'].sel(zmin=data_cfg['zmin'], pol=['q', 'u']).stack(z=('direc', 'time')).transpose('z', ..., 'pol')).astype(np.float32)

    nsplit = train_cfg['train_test_split']
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images[:nsplit]).shuffle(nsplit).batch(train_cfg['batch_size']).map(tf.image.per_image_standardization)
    test_dataset = tf.data.Dataset.from_tensor_slices(train_images[nsplit:]).batch(train_cfg['batch_size'])

    # Define the optimizer to use.
    optimizer = tf.keras.optimizers.Adam(beta_1=train_cfg['beta_1'], learning_rate=train_cfg['learning_rate'])
    
    # Build the VAE model
    model = began.CVAE(architecture_cfg['lat_dim'], architecture_cfg['kernel_size'], channels=NCHANNELS)

    # Set up an xarray dataset to log samples of the network's output at each
    # epoch of training. We hardcode that nine points in the latent space are
    # sampled. The generated samples are saved after training, but the state of
    # the model is not saved.
    dims = ["epoch", "z", "x", "y", "pol"]
    coords = {
        "epoch": np.arange(train_cfg['epochs']),
        "z": np.arange(9),
        "x": np.arange(NY),
        "y": np.arange(NX),
        "pol": ['q', 'u'],
    }
    data = np.zeros((train_cfg['epochs'], 9, NX, NY, NCHANNELS))
    samples_viz = xa.Dataset({'samples': (dims, data)}, coords=coords)
    z_viz = tf.random.normal(shape=[9, architecture_cfg['lat_dim']])

    # We now begin the training process. In each epoch, we loop over the 
    # training set in batches. After each batch, the gradients are computed
    # and updated. After 10 epochs, the loss is evaluated and logged, and
    # samples from the latent space evaluated with the current weights.
    for epoch in range(train_cfg['epochs']):
        # do training in batches
        for train_x in train_dataset: 
            compute_apply_gradients(model, train_x, optimizer)

        # generate samples after training for an epoch
        samples_viz['samples'][dict(epoch=epoch)] = model.sample(z_viz)
        
        # after every 10 epochs save summary statistics
        if epoch % 10 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(began.vae.compute_loss(model, test_x))
            elbo = - loss.result()

            #with summary_writer.as_default():
            #    tf.summary.scalar('elbo', elbo, step=epoch)
            facet_grid = samples_viz['samples'].sel(epoch=epoch, pol='q').plot(x='x', y='y', col='z', col_wrap=3, cmap='cividis', robust=True)
            facet_grid.fig.savefig(plot_dir / f"sample_q_epoch{epoch:04d}.png")
            facet_grid = samples_viz['samples'].sel(epoch=epoch, pol='u').plot(x='x', y='y', col='z', col_wrap=3, cmap='cividis', robust=True)
            facet_grid.fig.savefig(plot_dir / f"sample_u_epoch{epoch:04d}.png")
    model.save_weights(str(model_path))
    samples_viz.to_zarr(training_samples_path, 'w')

if __name__ == '__main__':
    main()

