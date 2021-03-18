import h5py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from began import build_adversarial_model, build_discriminator, build_generator, training_schedule
from pathlib import Path
from datetime import datetime
import os

if __name__ == '__main__':
    # set up directories for logging tensorflow training
    try:
        LOGDIR = Path(os.environ['TF_LOGDIR'])
    except KeyError:
        raise KeyError('TF_LOGDIR environment variable must be set for logging directory.')

    try:
        assert LOGDIR.exists()
    except AssertionError:
        raise AssertionError(r"$TF_LOGDIR does not exist, create before proceeding.")

    LOGDIR = LOGDIR / 'scalars' / datetime.now().strftime("%Y%m%d-%H%M%S")

    FILE_WRITER = tf.summary.create_file_writer(str(LOGDIR / "metrics"))
    FILE_WRITER.set_as_default()

    # Project directory
    PROJ_DIR = Path("/home/bthorne/projects/gan/began")
    # Model directory
    MODEL_PATH = PROJ_DIR / "model" / "mnist_dcgan_NTRAIN1000.h5"

    # Network architecture
    DEPTH = 32
    IMG_DIM = 28
    CHANNELS = 1
    KERNELS = [5, 5, 5]
    STRIDES = [2, 2, 2]
    FILTERS = [DEPTH * 2 ** i for i in range(len(KERNELS))]
    LATENT_DIM = 64

    # Derived parameters
    SHAPE = (IMG_DIM, IMG_DIM, CHANNELS)

    # Training parameters
    BUFFER_SIZE = 60000
    BATCH_SIZE = 32

    # Build inidividual and joint models.
    DIS = build_discriminator(FILTERS, KERNELS, STRIDES, SHAPE)
    GEN = build_generator(DIS, FILTERS, KERNELS, STRIDES, LATENT_DIM, SHAPE)
    ADV = build_adversarial_model(DIS, GEN)
    print(GEN.summary())
    # Load raw training data
    (X_TRAIN, _), (_, _) = mnist.load_data()

    # Apply preprocessing to scale data
    X_TRAIN = X_TRAIN.reshape(X_TRAIN.shape[0], 28, 28, 1).astype('float32')
    X_TRAIN = (X_TRAIN - 127.5) / 127.5
    TRAIN_DSET = tf.data.Dataset.from_tensor_slices(X_TRAIN).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    trained_model = training_schedule(DIS, GEN, ADV, TRAIN_DSET,
        LATENT_DIM, BUFFER_SIZE, BATCH_SIZE, callback=True)
    trained_model.save(os.fspath(MODEL_PATH))