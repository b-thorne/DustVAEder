#!/home/bthorne/projects/gan/began/gpu-env/bin/python
#SBATCH --job-name="mhd-temp-vae-hp-optimization"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH -t 2-00:00:00

""" This script uses the keras-tuner package to search the
space of the began.CVAE hyperparameters. The hyperparameters
invluded in the search are:

- Size of latent dimension
- Size of inference net convolutional kernel
- Training batch size
- Learning rate

The number of epochs over which to run each trial is choosable, 
and the default is 400 epochs. The maximum number of trials is
also choosable, but is set to 20 by hand. 

This script optimizes the VAE trained on MHD data.
"""

import sys
import click
import logging
import yaml
import h5py
from pathlib import Path
import kerastuner as kt
import numpy as np
import tensorflow as tf
import xarray as xa
import began
from began.logging import setup_vae_run_logging

_logger = logging.getLogger(__name__)

def load_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg['data'], cfg['optimization']

@click.command()
@click.option('--opt_config_path', 'opt_config_path', required=True, type=click.Path(exists=True), help='path to configuration for optimization')
@click.option('--results_dir', 'results_dir', required=True, type=click.Path(exists=True), help='directory to place results')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(opt_config_path: Path, results_dir: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                        )
    # Read configuration file containing hyperparameter specficcations.
    data_cfg, optimization_cfg, = load_cfg(opt_config_path)

    # Read in MHD data and convert to testing and training
    # tensorflow datasets.
    train_images = xa.open_zarr("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd.zarr")
    train_images = np.array(train_images['data'].sel(pol='t').stack(z=('zmin', 'direc', 'time')).transpose('z', ...)).astype(np.float32)[..., None]
    nsplit = data_cfg['test_split']
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images[nsplit:]).shuffle(train_images.shape[0]-nsplit)

    # initialize the type of keras-tuner oracle to use (we use Bayesian Optimization)
    objective = kt.Objective('loss', 'min')
    oracle = kt.oracles.BayesianOptimization(objective=objective, max_trials=optimization_cfg['max_trials'])
    # initialize instance of keras-tuner
    lat_dim_range = optimization_cfg['lat_dim_range']
    kernel_size_range = optimization_cfg['kernel_size_range']
    build_model = model_builder(lat_dim_range, kernel_size_range)

    tuner = HPTuner(
        oracle=oracle,
        hypermodel=build_model,
        directory=results_dir,
        project_name=data_cfg['project_name']
        )
    
    # do the hyperparameter search
    tuner.search(dataset=train_dataset, epochs=optimization_cfg['epochs'])
    
    # print results of hyperparameter search
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)

def model_builder(lat_dim_range, kernel_size_range):
    """ This function wraps a model building function. This allows
    us to run the optimization in differenct ranges of hyperparameters.
    """
    def wrapper(hp):
        lat_dim = hp.Int('lat_dim', *lat_dim_range)
        kernel_size = hp.Choice('kernel_size', kernel_size_range)
        return began.CVAE(lat_dim, kernel_size)
    return wrapper

class HPTuner(kt.Tuner):

    def run_trial(self, trial, dataset, epochs):
        hp = trial.hyperparameters

        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        train_ds = dataset.batch(hp.Int('batch_size', 8, 32, default=8)).map(tf.image.per_image_standardization)
        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log', default=0.0002)
        optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=lr)
        epoch_loss_metric = tf.keras.metrics.Mean()

        @tf.function
        def run_train_step(data):
            with tf.GradientTape() as tape:
                loss = began.vae.compute_loss(model, data)
                gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss

        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        for epoch in range(epochs):
            print('Epoch: {}'.format(epoch))
            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch, data in enumerate(train_ds):
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = float(run_train_step(data))
                self.on_batch_end(trial, model, batch, logs={'loss': batch_loss})

                if batch % 30 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print('Batch: {}, Average Loss: {}'.format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={'loss': epoch_loss})
            epoch_loss_metric.reset_states()

if __name__ == '__main__':
    main()