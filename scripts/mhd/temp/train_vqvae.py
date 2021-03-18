#!/home/bthorne/projects/gan/began/gpu-env/bin/python
#SBATCH --job-name="vqvae-T"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=bthorne
#SBATCH -t 12:00:00
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
import sonnet as snt

import began
from began.logging import setup_vae_run_logging
from began import Encoder, Decoder, VQVAEModel
from began.augmentation import rotate, flip

_logger = logging.getLogger(__name__)

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
    # Read in the MHD data
    data_dir = Path("/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/data")
    
    train_fpath = data_dir / "mhd_ntrain-3150.cdf"
    test_fpath = data_dir / "mhd_ntest-0300.cdf"

    train_images = xa.open_dataarray(train_fpath).sel(pol=['t']).transpose('batch', ..., 'pol').values.astype(np.float32)
    test_images = xa.open_dataarray(test_fpath).sel(pol=['t']).transpose('batch', ..., 'pol').values.astype(np.float32)

    channels = train_images.shape[-1]

    train_data_variance = np.var(train_images)

    # Vector Quantized VAE hyperparameters
    batch_size = 32
    image_size = 256

    # 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
    # 10k steps gives reasonable accuracy with VQVAE on Cifar10. 
    num_training_updates = 100000

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    # These hyper-parameters define the size of the model (number of parameters and layers).
    # The hyper-parameters in the paper were (For ImageNet):
    # batch_size = 128
    # image_size = 128
    # num_hiddens = 128
    # num_residual_hiddens = 32
    # num_residual_layers = 2

    # This value is not that important, usually 64 works.
    # This will not change the capacity in the information-bottleneck.
    embedding_dim = 64

    # The higher this value, the higher the capacity in the information bottleneck.
    #num_embeddings = 512
    num_embeddings = 1024

    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost = 0.25

    # Use EMA updates for the codebook (instead of the Adam optimizer).
    # This typically converges faster, and makes the model less dependent on choice
    # of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
    # developed afterwards). See Appendix of the paper for more details.
    vq_use_ema = True

    # This is only used for EMA updates.
    decay = 0.99

    learning_rate = 3e-4

    # Setup checkpointing
    checkpoint_root = "/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/checkpoints"
    checkpoint_name = f"temp-vqvae-embed{num_embeddings:04d}"
    save_prefix = os.path.join(checkpoint_root, checkpoint_name)

    # Saved model path
    saved_model_path = f"/oasis/scratch/comet/bthorne/temp_project/began_scratch/mhd/mhd_temp/temp-vqvae-{num_embeddings:04d}"

    # Setup training dataset
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_images.shape[0])
                 .map(rotate, num_parallel_calls=4)
                 .map(flip, num_parallel_calls=4)
                 .repeat(-1)
                 .batch(batch_size)
                 .prefetch(-1))

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .repeat(1)  # 1 epoch
        .batch(batch_size)
        .prefetch(-1))

    # # Build modules.
    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, channels=channels)
    pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
        kernel_shape=(1, 1),
        stride=(1, 1),
        name="to_vq")


    if vq_use_ema:
        vq_vae = snt.nets.VectorQuantizerEMA(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay)
    else:
        vq_vae = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)
        
    model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                    data_variance=train_data_variance)

    chkpt = tf.train.Checkpoint(module=model)


    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    latest = tf.train.latest_checkpoint(checkpoint_root)
    if latest is not None:
        chkpt.restore(latest)

        # Read in MHD data and convert to testing and training
        # tensorflow datasets.

    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            model_output = model(data, is_training=True)
        trainable_variables = model.trainable_variables
        grads = tape.gradient(model_output['loss'], trainable_variables)
        optimizer.apply(grads, trainable_variables)
        return model_output

    for step_index, data in enumerate(train_dataset):
        train_results = train_step(data)
        train_losses.append(train_results['loss'])
        train_recon_errors.append(train_results['recon_error'])
        train_perplexities.append(train_results['vq_output']['perplexity'])
        train_vqvae_loss.append(train_results['vq_output']['loss'])

        if (step_index + 1) % 100 == 0:
            print('%d train loss: %f ' % (step_index + 1,
                                    np.mean(train_losses[-100:])) +
            ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
            ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
            ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))

        if step_index and not step_index % 1000:
            chkpt.save(save_prefix)
            
        if step_index == num_training_updates:
            break
    chkpt.save(save_prefix)

    @tf.function(input_signature=[tf.TensorSpec([None, 256, 256, 1])])
    def inference(x):
        return model(x, is_training=False)

    to_save = snt.Module()
    to_save.inference = inference
    to_save.all_variables = list(model.variables)
    tf.saved_model.save(to_save, saved_model_path)

if __name__ == '__main__':
    main()

