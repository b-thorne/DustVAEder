#!/home/bthorne/projects/gan/began/envs-gpu/bin/python
#SBATCH --job-name="vae-hp-optimization"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH -t 24:00:00
import logging
from pathlib import Path
import sys
import time
import yaml

import click
from IPython.core import ultratb

import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py
import began
from began.logging import setup_vae_run_logging
from began.visualization import mplot, plot

_logger = logging.getLogger(__name__)

@click.command()
@click.option('--train_cfg_path', 'train_cfg_path', required=True,
              type=click.Path(exists=True), help='path to training config file of network')
@click.option('--model_cfg_path', 'model_cfg_path', required=True,
              type=click.Path(exists=True), help='path to model config file of network')
@click.option('--train_path', 'train_path', required=True, 
                type=click.Path(exists=True), help='path to training data')
@click.option('--model_path', 'model_path', required=True,
                type=click.Path(), help='path to output file in which to save model')
@click.option('--plot_dir', 'plot_dir', type=click.Path(exists=True), required=True)
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(train_cfg_path: Path, model_cfg_path: Path, train_path: Path, model_path: Path, plot_dir: Path, seed: int, log_level: int):
    # initialize random seed in numpy
    np.random.seed(seed)
    # initialize random seed in tensorflow
    tf.random.set_seed(seed)
    
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    plot_dir = Path(plot_dir).absolute()

    cfg = {}

    with open(train_cfg_path) as f:
        cfg.update(yaml.load(f, Loader=yaml.FullLoader))
    
    with open(model_cfg_path) as f:
        cfg.update(yaml.load(f, Loader=yaml.FullLoader))

    logging.info(
        """Dependencies:
            Training data path: {:s}
            Model save path: {:s}
            Plotting directory: {:s}
    """.format(train_path, model_path, str(plot_dir)))

    logging.info("""Working with GPU: {:s}""".format(str(tf.test.is_gpu_available())))

    logging.info("""
    Network parameters:
        Size of latent dimension: {:d}
        Batch size: {:d}
        Epochs: {:d}
    """.format(cfg['lat_dim'], cfg['batch_size'], cfg['epochs']))
    # set up logging
    summary_writer = setup_vae_run_logging(cfg['lat_dim'], cfg['batch_size'], cfg['epochs'])

    # Batch and shuffle the data
    with h5py.File(train_path, 'r') as f:
        dset = f["cut_maps"]
        train_images = dset[...].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(cfg['batch_size']).map(tf.image.per_image_standardization)
    test_dataset = dataset.take(100)
    train_dataset = dataset.skip(100)

    optimizer = tf.keras.optimizers.Adam(beta_1=cfg['beta_1'], learning_rate=cfg['learning_rate'])
    model = began.CVAE(cfg['lat_dim'], cfg['kernel_size'])

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    
    random_vector_for_generation = tf.random.normal(shape=[9, cfg['lat_dim']])
    for epoch in range(1, cfg['epochs'] + 1):
        print("Epoch: ", epoch)
        start_time = time.time()
        for step, train_x in enumerate(train_dataset):
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(began.vae.compute_loss(model, test_x))
            elbo = -loss.result()
            print("\t loss: ", elbo)

            with summary_writer.as_default():
                tf.summary.scalar('elbo', elbo, step=epoch)
            title = 'Epoch: {:03d}, Test set ELBO: {:04.02f}, time elapse for current epoch {:02.02f}'.format(epoch, elbo, end_time - start_time)
            generate_and_save_images(model, epoch, random_vector_for_generation, plot_dir, title)

    model.save_weights(model_path)

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = began.vae.compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_and_save_images(model, epoch, test_input, plot_dir, title=None):
    predictions = model.sample(test_input)
    mean = np.mean(predictions)
    std = np.std(predictions)
    fig, _ = mplot(predictions[..., 0], extent=(-10, 10, -10, 10), title=title, cbar_range=[mean - 2*std, mean+2*std])
    fig.savefig(plot_dir / 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')

if __name__ == '__main__':
    main()

