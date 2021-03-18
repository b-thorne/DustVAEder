#!/home/bthorne/projects/gan/began/envs-gpu/bin/python
#SBATCH --job-name="vae-hp-optimization"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH -t 24:00:00

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
import began
from began.logging import setup_vae_run_logging

_logger = logging.getLogger(__name__)

@click.command()
@click.option('--train_path', 'train_path', required=True, type=click.Path(exists=True), help='path to training data')
@click.option('--results_dir', 'results_dir', required=True, type=click.Path(exists=True), help='directory to place results')
@click.option('--epochs', 'epochs', type=int, default=400)               
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(train_path: Path, results_dir: Path, epochs: int, seed: int, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                        )
    
    # initialize random seed in numpy
    np.random.seed(seed)
    # initialize random seed in tensorflow
    tf.random.set_seed(seed)
    
    # make paths absolute
    train_path = Path(train_path).absolute()
    results_dir = Path(results_dir).absolute()

    # read in training data, make into dataset, shuffle, and standardize to mean=0,std=1. 
    with h5py.File(train_path, 'r') as f:
        dset = f["cut_maps"]
        train_images = dset[...].astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).map(tf.image.per_image_standardization)
    test_dataset = dataset.take(100)
    train_dataset = dataset.skip(100)

    # initialize the type of keras-tuner oracle to use (we use Bayesian Optimization)
    oracle = kt.oracles.Hyperband(objective=kt.Objective('loss', 'min'), max_trials=1)
    # initialize instance of keras-tuner
    tuner = HPTuner(
        oracle=oracle,
        hypermodel=build_model,
        directory=results_dir,
        project_name='vae_custom_training_hp'
        )
    
    # do the hyperparameter search
    tuner.search(train_ds=train_dataset, epochs=epochs)
    
    # print results of hyperparameter search
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)

def build_model(hp):
    """Builds a convolutional model."""
    lat_dim = hp.Int('lat_dim', 32, 512)
    kernel_size = hp.Choice('kernel_size', values=[4, 5])
    return began.CVAE(lat_dim, kernel_size)

class HPTuner(kt.Tuner):

    def run_trial(self, trial, train_ds, epochs):
        hp = trial.hyperparameters

        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        train_ds = train_ds.batch(hp.Int('batch_size', 8, 32, default=8))
        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float('learning_rate', 5e-5, 1e-3, sampling='log', default=2e-4)
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