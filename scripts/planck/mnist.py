import h5py
from tensorflow.keras.datasets import mnist
from src import build_adversarial_model, build_discriminator, build_generator, training_schedule
from pathlib import Path
import os

if __name__ == '__main__':
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
    TRAIN_STEPS = 5000
    BATCH_SIZE = 32

    # Build inidividual and joint models.
    DIS = build_discriminator(FILTERS, KERNELS, STRIDES, SHAPE)
    GEN = build_generator(DIS, FILTERS, KERNELS, STRIDES, LATENT_DIM, SHAPE)
    ADV = build_adversarial_model(DIS, GEN)
    print(GEN.summary())
    # Load raw training data
    (X_TRAIN, _), (_, _) = mnist.load_data()

    # Apply preprocessing to scale data
    X_TRAIN = X_TRAIN[..., None] / 255. * 2. - 1.

    trained_model = training_schedule(DIS, GEN, ADV, X_TRAIN[:1000],
        LATENT_DIM, TRAIN_STEPS, BATCH_SIZE)
    trained_model.save(os.fspath(MODEL_PATH))