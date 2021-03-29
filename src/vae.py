import os

from absl import app
from absl import flags
from absl import logging

import gin

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import GlorotUniform as glorot_uniform

from pathlib import Path
import pickle

from datasets import load_dataset
from utils import rotate
import plots 

try:
    import pymaster as nmt
except ImportError:
    print("Continuing without NaMaster")

tfkl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

FLAGS = flags.FLAGS
PROJECT_DIR = Path(os.path.dirname(__file__))
RESULTS_DIR = "/global/cscratch1/sd/bthorne/dustvaeder/results"

@gin.configurable("ConvBatchLayer", denylist=["x"])
def ConvBatchLayer(x, filters, kernel_size, strides, activation=tf.nn.relu, padding='same', bn_axis=-1, kernel_initializer=glorot_uniform, momentum=0.9):
    x = tfkl.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=None, kernel_initializer=kernel_initializer())(x)
    x = tfkl.BatchNormalization(momentum=momentum, axis=bn_axis)(x)
    x = tfkl.Activation(activation)(x)
    return x


def TransConvBatchLayer(x, filters, kernel_size, strides, bn_axis=-1, activation=tf.nn.relu, momentum=0.9):
    x = tfkl.BatchNormalization(momentum=momentum, axis=bn_axis)(x)
    x = tfkl.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(x)
    return x 


@gin.configurable("conv_block")
def ConvBlock(x, filters=256, strides=2, kernel_size=4):
    x = ConvBatchLayer(x, filters, kernel_size, strides)
    x = ConvBatchLayer(x, filters // 2, kernel_size, strides)
    x = ConvBatchLayer(x, filters // 2 // 2, kernel_size, strides)
    return x


@gin.configurable("transconv_block")
def TransConvBlock(x, nfilters_top=128, kernel_size=4, nchannels=1):
    x = TransConvBatchLayer(x, nfilters_top, kernel_size, (2, 2))
    x = TransConvBatchLayer(x, nfilters_top // 2, kernel_size, (2, 2))
    x = TransConvBatchLayer(x, nfilters_top // 2 // 2, kernel_size, (2, 2))
    x = TransConvBatchLayer(x, nfilters_top // 2 // 2 // 2, kernel_size, (2, 2))
    x = TransConvBatchLayer(x, 2 * nchannels, 1, (1, 1), activation=None) # this must have two filters in order for the dimensions to work. Else need a Dense layer.
    return x


@gin.configurable("transconv_decoder")
def TransConvDecoder(latent_dimension, input_shape, nfilters_top=128, kernel_size=4):
    inputs = tfk.Input(shape=latent_dimension)
    npix_top = int(input_shape[0] / 2 ** 4)
    x = tfkl.Dense(npix_top ** 2 * 32)(inputs)
    x = tfkl.Reshape([npix_top, npix_top, 32])(x)
    x = TransConvBlock(x, nfilters_top=nfilters_top, nchannels=input_shape[-1])
    x = tfkl.Flatten()(x)
    x = tfpl.IndependentNormal(input_shape)(x)
    return tfk.Model(inputs=inputs, outputs=x, name="transconv_decoder")


@gin.configurable("conv_encoder")
def ConvEncoder(input_shape, num_filters_top=256, kernel_size=4, latent_dimension=256):
    inputs = tfk.Input(shape=input_shape)
    x = ConvBlock(inputs, filters=num_filters_top)
    x = tfkl.Flatten()(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(latent_dimension), activation=None)(x)
    x = tfpl.IndependentNormal(latent_dimension)(x)
    return tfk.Model(inputs=inputs, outputs=x, name="conv_encoder")

@gin.configurable("conv_encoder_ae")
def ConvEncoderAE(input_shape, num_filters_top=256, kernel_size=4, latent_dimension=256):
    inputs = tfk.Input(shape=input_shape)
    x = ConvBlock(inputs, filters=num_filters_top)
    x = tfkl.Flatten()(x)
    x = tfkl.Dense(latent_dimension, activation=None)(x)
    return tfk.Model(inputs=inputs, outputs=x, name="conv_encoder")

def ResNetIdentityBlock(x, filters, kernel_size):
  x_shortcut = x

  f1, f2, f3 = filters

  x = ConvBatchLayer(x, f1, 1, 1)
  x = ConvBatchLayer(x, f2, kernel_size, 1)
  x = ConvBatchLayer(x, f3, 1, 1)
  
  x = tfkl.Add()([x, x_shortcut])
  x = tfkl.Activation('relu')(x)
  return x

def ResNetConvBlock(x, filters, kernel_size, s=2):
    x_shortcut = x

    f1, f2, f3 = filters

    x = ConvBatchLayer(x, f1, 1, s, padding='valid')
    x = ConvBatchLayer(x, f2, kernel_size, 1, padding='same')
    x = ConvBatchLayer(x, f3, 1, 1, padding='valid')

    x_shortcut = tfkl.Conv2D(f3, 1, s, padding='valid', kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = tfkl.BatchNormalization()(x_shortcut)

    x = tfkl.Add()([x, x_shortcut])
    x = tfkl.Activation('relu')(x)
    return x

def ResNetEncoder(input_shape, latent_dimension):
    x_input = tfkl.Input(input_shape)

    x = tfkl.Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(x_input)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation('relu')(x)
    x = tfkl.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = ResNetConvBlock(x, [64, 64, 256], 3, s=1)
    x = ResNetIdentityBlock(x, [64, 64, 256], 3)
    x = ResNetIdentityBlock(x, [64, 64, 256], 3)

    x = ResNetConvBlock(x, [128, 128, 512], 3, s=2)
    x = ResNetIdentityBlock(x, [128, 128, 512], 3)
    x = ResNetIdentityBlock(x, [128, 128, 512], 3)
    x = ResNetIdentityBlock(x, [128, 128, 512], 3)

    x = ResNetConvBlock(x, [256, 256, 1024], 3, s=1)
    x = ResNetIdentityBlock(x, [256, 256, 1024], 3)
    x = ResNetIdentityBlock(x, [256, 256, 1024], 3)
    x = ResNetIdentityBlock(x, [256, 256, 1024], 3)
    x = ResNetIdentityBlock(x, [256, 256, 1024], 3)

    x = ResNetConvBlock(x, [512, 512, 2048], 3, s=2)
    x = ResNetIdentityBlock(x, [512, 512, 2048], 3)
    x = ResNetIdentityBlock(x, [512, 512, 2048], 3)

    x = tfkl.AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    x = tfkl.Flatten()(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(latent_dimension), activation=None)(x) 
    x = tfpl.IndependentNormal(latent_dimension)(x)

    return tfk.Model(inputs=x_input, outputs=x, name='ResNet50')


def IAF(latent_dimension):
    # Implementation of Inverse Autoregressive Flow similar to that outlined in
    # https://arxiv.org/abs/1606.04934
    tfd.TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
            bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                    params=2, hidden_units=[512, 512])
                    )
            )
                    )
    return

@gin.configurable("VAE")
class VAE(tf.keras.Model):
    def __init__(
            self,
            input_shape=(256, 256, 1), 
            latent_dimension=256,
            beta=1.,
            encoder_type="conv", 
            decoder_type="transconv", 
            name="VAE"):
        super(VAE, self).__init__(name=name)

        self.neg_elbo_tracker = tfk.metrics.Mean(name="neg elbo")
        self.reconstruction_loss_tracker = tfk.metrics.Mean(name="reconstruction loss")
        self.kl_loss_tracker = tfk.metrics.Mean(name="kl loss")

        self.beta = beta

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dimension), scale=1), reinterpreted_batch_ndims=1)

        if encoder_type == "conv":
            self._encoder = ConvEncoder(input_shape, latent_dimension=latent_dimension)
        if encoder_type == "ResNet":
            self._encoder = ResNetEncoder(input_shape, latent_dimension)

        if decoder_type == "transconv": 
            self._decoder = TransConvDecoder(latent_dimension, input_shape)

        return 

    def call(self, x):
        encoded_dist = self._encoder(x)
        return self._decoder(encoded_dist.sample())
    
    def sample(self, size):
        return self._decoder(self.prior.sample(size))

    @property
    def metrics(self):
        return [
            self.neg_elbo_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def calculate_loss(self, x):
        encoding_dist = self._encoder(x)
        sampled_z = encoding_dist.sample()
        sampled_decoding_dist = self._decoder(sampled_z)
        kl_loss = tf.reduce_sum(tfd.kl_divergence(encoding_dist, self.prior))
        reconstruction_loss = sampled_decoding_dist.log_prob(x)
        neg_elbo = - reconstruction_loss + self.beta * kl_loss
        return kl_loss, reconstruction_loss, neg_elbo

    def train_step(self, x):
        with tf.GradientTape() as tape:
            kl_loss, reconstruction_loss, neg_elbo = self.calculate_loss(x)

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.neg_elbo_tracker.update_state(neg_elbo)

        grads = tape.gradient(neg_elbo, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'neg_elbo': self.neg_elbo_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }

    def test_step(self, x):
        kl_loss, reconstruction_loss, neg_elbo = self.calculate_loss(x)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.neg_elbo_tracker.update_state(neg_elbo)            
        return {
            'neg_elbo': self.neg_elbo_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }


@gin.configurable("AE")
class AE(tf.keras.Model):
    def __init__(
            self,
            input_shape=(256, 256, 1), 
            latent_dimension=256,
            num_filters_top=256,
            nfilters_top=512, 
            encoder_type="conv", 
            decoder_type="transconv",
            name="AE"):
        super(AE, self).__init__(name=name)

        self.mse_loss_tracker = tfk.metrics.Mean(name="mse")

        if encoder_type == "conv":
            self._encoder = ConvEncoderAE(input_shape, num_filters_top=num_filters_top, latent_dimension=latent_dimension)
        if encoder_type == "ResNet":
            self._encoder = ResNetEncoder(input_shape, latent_dimension)

        if decoder_type == "transconv": 
            self._decoder = TransConvDecoder(latent_dimension, input_shape, nfilters_top=nfilters_top)

        return 

    def call(self, x):
        encoded_vector = self._encoder(x)
        return self._decoder(encoded_vector)

    @property
    def metrics(self):
        return [
            self.mse_loss_tracker,
        ]

    def calculate_loss(self, x):
        encoded_vector = self._encoder(x)
        decoded_image = self._decoder(encoded_vector)
        mse_loss = tfk.losses.MSE(decoded_image, x)
        return mse_loss

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mse_loss = self.calculate_loss(x)

        self.mse_loss_tracker.update_state(mse_loss)

        grads = tape.gradient(mse_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'mse_loss': self.mse_loss_tracker.result(),
        }

    def test_step(self, x):
        mse_loss = self.calculate_loss(x)
        self.mse_loss_tracker.update_state(mse_loss)           
        return {
            'mse_loss': self.mse_loss_tracker.result(),
        }


def make_21011181_plots(test_dataset, vae):
    
    dr = iter(test_dataset)
    pred_true = np.zeros((3, 256, 256, 3))
    for i in range(3):
        pred_true[i] = next(dr).numpy()

    pred = vae(pred_true).mean().numpy()

    fig, ax = plots.make_prediction_plot_with_residuals(pred_true, pred, "")
    fig.savefig("plot_w-res.pdf", bbox_inches="tight")

    prior_sample = vae.sample(3).mean().numpy()
    fig, ax = plots.make_prior_sample_plot(prior_sample)
    fig.savefig("prior_sample.pdf", bbox_inches="tight")


def Spectra(Lx=8, Ly=8, Nx=256, Ny=256, aposize=1.):
    Lx = np.deg2rad(Lx)
    Ly = np.deg2rad(Ly)
    mask = np.ones((Nx, Ny)).flatten()
    xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
    yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny

    f = 200.
    mask[np.where(xarr.flatten() < Lx / f)] = 0
    mask[np.where(xarr.flatten() > (f - 1) * Lx / f)] = 0
    mask[np.where(yarr.flatten() < Ly / f)] = 0
    mask[np.where(yarr.flatten() > (f - 1) * Ly / f)] = 0
    mask = mask.reshape([Ny, Nx])
    mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposize, apotype="C1")

    l0_bins = np.arange(Nx / 4) * 4 * np.pi / Lx
    lf_bins = (np.arange(Nx / 4) + 1) * 4 * np.pi / Lx

    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    f0 = nmt.NmtFieldFlat(Lx, Ly, mask, np.random.randn(1, Nx, Ny))
    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f0, b)
    return

def NmtTTSpectra(dataset, Lx, Ly, mask, binning):
    """ Get TT spectra of images (assumed shape (nsamples, xdim, ydim, nchannels))
    """
    N = dataset.shape[0]
    cl = np.zeros((N, binning.get_n_bands()))
    for i in range(N):
        f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [dataset[i, :, :, 0]])
        cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
        cl00_uncoupled = w00.decouple_cell(cl00_coupled)
        cl[i] = cl00_uncoupled
    return cl

@gin.configurable("Dataset")
def Dataset(label, batch_size, seed=1234):
    tf.random.set_seed(seed)
    return load_dataset(label, batch_size)

@gin.configurable("Eval", denylist=["test_dataset", "results_dir"])
def Eval(test_dataset, train_dataset, results_dir, data_dir, vae):
    eval_results = vae.evaluate(test_dataset)

    batch = next(iter(test_dataset))
    
    truths = batch[:3]
    predictions = vae(truths).mean().numpy()
    fig, ax = plots.make_prediction_plot_with_residuals(truths[..., :1], predictions[..., :1], "")
    fig.savefig(results_dir / "T_recon_w_res.pdf", bbox_inches="tight")

    if batch.shape[-1] == 3:
        fig, ax = plots.make_prediction_plot_with_residuals(truths[..., 1:2], predictions[..., 1:2], "", vlims=[-0.5, 0.5])
        fig.savefig(results_dir / "Q_recon_w_res.pdf", bbox_inches="tight")

        fig, ax = plots.make_prediction_plot_with_residuals(truths[..., 2:], predictions[..., 2:], "", vlims=[-0.5, 0.5])
        fig.savefig(results_dir / "U_recon_w_res.pdf", bbox_inches="tight")

    samples = vae._decoder(vae.prior.sample(3)).mean().numpy()
    fig, ax = plots.make_prior_sample_plot(samples[..., :1])
    fig.savefig(results_dir / "T_sample.pdf", bbox_inches='tight')
    if batch.shape[-1] == 3:
        fig, ax = plots.make_prior_sample_plot(samples[..., 1:2])
        fig.savefig(results_dir / "Q_sample.pdf", bbox_inches='tight')
        fig, ax = plots.make_prior_sample_plot(samples[..., 2:])
        fig.savefig(results_dir / "U_sample.pdf", bbox_inches='tight')

    history = pickle.load(open(results_dir / 'history.pkl', "rb"))
    fig, ax = plots.history_plot(history, ["reconstruction_loss", "kl_loss", "neg_elbo"])
    fig.savefig(results_dir / "training_history.pdf", bbox_inches='tight')

    # Now process all of the test and train set images. Too slow to do 
    # on the fly on CPU. Really need to figure out how to get NaMaster
    # to work in a kernel with GPU-based TF.
    predicted_test = []
    true_test = []
    for x in test_dataset:
        predicted_test.append(vae(x).mean().numpy())
        true_test.append(x.numpy())
    predicted_test = np.concatenate(predicted_test)
    true_test = np.concatenate(true_test)

    predicted_train = []
    true_train = []
    for x in train_dataset:
        predicted_train.append(vae(x).mean().numpy())
        true_train.append(x.numpy())
    predicted_train = np.concatenate(predicted_train)
    true_train = np.concatenate(true_train)

    np.save(data_dir / "test_true.npy", true_test)
    np.save(data_dir / "test_predicted.npy", predicted_test)

    np.save(data_dir / "train_true.npy", true_train)
    np.save(data_dir / "train_predicted.npy", predicted_train)
    return eval_results

def make_semantic_sequence_dataset(test_dataset, vae):
    testset = []
    for x in test_dataset:
        testset.append(x.numpy())
    testset = np.concatenate(testset)

    np.random.seed(192)
    i1 = np.random.randint(0, testset.shape[0])
    i2 = np.random.randint(0, testset.shape[0])
    test_map1 = testset[[i1]]
    test_map2 = testset[[i2]]
    z_mu_i1 = vae._encoder(test_map1)
    z_mu_i2 = vae._encoder(test_map2)
    z_interp = spherical_interpolation(z_mu_i1.mean().numpy()[0], z_mu_i2.mean().numpy()[0], num_samples=100)
    decoded_interpolated_vectors = vae._decoder(z_interp)
    semantic_sequence = np.concatenate((test_map1, decoded_interpolated_vectors.mean().numpy(), test_map2))
    np.save("/global/cscratch1/sd/bthorne/dustvaeder/semantic_interpolation_block.npy", semantic_sequence)
    return

def spherical_interpolation(z1, z2, num_samples=4):
    """ Function to interpolate between two Cartesian vectors, :math:`z1` and :math:`z2`,
    assuming a standard normal prior.
    """
    lamb = np.linspace(0, 1, num_samples)
    theta = np.dot(z1, z2) * np.pi / 180.
    return np.sin(theta * (1 - lamb[:, None])) / np.sin(theta) * z1[None, :] + np.sin(lamb[:, None] * theta) / np.sin(theta) * z2[None, :]

@gin.configurable("Train", denylist=["train_dataset", "val_dataset"])
def Train(train_dataset, val_dataset, model, optimizer=tfk.optimizers.Adam, epochs=2):
    model.compile(optimizer=optimizer)
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)   
    return model, history

@gin.configurable("Load", denylist=["saved_model_path"])
def Load(saved_model_path, vae, optimizer=tfk.optimizers.Adam):
    vae.compile(optimizer=optimizer)
    vae.load_weights(str(saved_model_path)).expect_partial()
    return vae

def main(argv):
    del argv # unused
    # parse the configuration file with hyperparameter settings
    gin.parse_config_file(FLAGS.gin_config)

    # make directory for results to be saved if it does not exist:
    # i) for data products to be stored on scratch
    results_dir_data = Path(RESULTS_DIR).absolute() / FLAGS.results_name
    results_dir_data.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving data products (including model) to {results_dir_data}")
    saved_model_path = str(results_dir_data / "vae")
    # ii) for corresponding plots and configs that need to be viewed
    results_dir_plot = Path(PROJECT_DIR).absolute().parent / "results" / FLAGS.results_name
    results_dir_plot.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving readable outputs to {results_dir_plot}")
    with open(results_dir_plot / "config.gin", "w") as f:
        f.write(gin.operative_config_str())

    # Get dataset
    train_dataset, val_dataset, test_dataset, dataset_info = Dataset()

    if FLAGS.mode == "standard":
        # perform training
        vae, history = Train(train_dataset, val_dataset)
        # save trained model
        vae.save_weights(saved_model_path, save_format='tf')
        with open(results_dir_plot / "history.pkl", 'wb') as f:
            pickle.dump(history.history, f) 
        # evaluate trained model 
        Eval(test_dataset, train_dataset, results_dir_plot, results_dir_data, vae)

    if FLAGS.mode == "train_vae":
        # perform training
        vae, history = Train(train_dataset, val_dataset)
        # save trained model
        vae.save_weights(saved_model_path, save_format='tf')
        with open(results_dir_plot / "history.pkl", 'wb') as f:
            pickle.dump(history.history, f) 

    if FLAGS.mode == "train_pae":
        # perform training
        ae, history = Train(train_dataset, val_dataset)
        # save trained model
        pae.save_weights(saved_model_path, save_format='tf')
        with open(results_dir_plot / "history.pkl", 'wb') as f:
            pickle.dump(history.history, f) 

    if FLAGS.mode == "eval":
        # evaluate pre-trained model
        vae = Load(saved_model_path, gin.REQUIRED)
        eval_results = Eval(test_dataset, train_dataset, results_dir_plot, results_dir_data, vae)

    if FLAGS.mode == "spectra":
        Spectra()

    if FLAGS.mode == "semantic":
        vae = Load(saved_model_path, gin.REQUIRED)
        make_semantic_sequence_dataset(test_dataset, vae)

    if FLAGS.mode == "2101.11181":
        # make plots for paper arXiv:2101.11181
        # load model
        vae = Load(saved_model_path, gin.REQUIRED)
        # make plots
        make_21011181_plots(test_dataset, vae)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "GNILC", ["GNILC", "MHD"], "Which dataset to use.")
    flags.DEFINE_enum("mode", "standard", ["standard", "training", "eval", "spectra", "semantic", "2101.11181"],  
        "Which mode to run in. Standard will train the model, and run the evaluations.")
    flags.DEFINE_string("results_name", "default", "Directory to be created for results.")
    flags.DEFINE_string("gin_config", "./configs/config.gin", "File containing the gin configuration.")

    app.run(main)