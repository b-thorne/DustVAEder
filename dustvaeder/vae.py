from absl import app
from absl import flags

import gin

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import GlorotUniform as glorot_uniform

from pathlib import Path

from datasets import load_dataset
from utils import rotate

tfkl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

FLAGS = flags.FLAGS

def ConvBatchLayer(x, filters, kernel_size, strides, activation='relu', padding='same', bn_axis=-1, kernel_initializer=glorot_uniform, momentum=0.9):
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

def ResNetIdentityBlock(x, filters, kernel_size):
  x_shortcut = x

  f1, f2, f3 = filters

  x = ConcBatchLayer(x, f1, 1, 1)
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
    X_input = tfkl.Input(input_shape)

    x = tfkl.Conv2D(64, (1, 1), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(x_input)
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

    x = tfkl.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    x = tfkl.Flatten()(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(latent_dimension), activation=None)(x) 
    x = tfpl.IndependentNormal(latent_dimension)(x)

    return tfk.Model(inputs=x_input, outputs=x, name='ResNet50')


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
            self._encoder = ConvEncoder(input_shape)
        if encoder_type == "ResNet":
            self._encoder = ResNetEncoder

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

def make_21011181_plots(test_dataset, vae):
    import plots 
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

@gin.configurable("Dataset")
def Dataset(label, batch_size):
    return load_dataset(label, batch_size)

@gin.configurable("eval", denylist=["test_dataset"])
def Eval(test_dataset, vae):
    eval_results = vae.evaluate(test_dataset)
    return eval_results

@gin.configurable("Train", denylist=["train_dataset", "val_dataset"])
def Train(train_dataset, val_dataset, vae, optimizer=tfk.optimizers.Adam, epochs=2):
    vae.compile(optimizer=optimizer)
    history = vae.fit(train_dataset, epochs=epochs, validation_data=val_dataset)   
    return vae

@gin.configurable("load", denylist=["saved_model_path"])
def Load(saved_model_path, vae, optimizer=tfk.optimizers.Adam, input_shape=(256, 256, 1)):
    vae = VAE(input_shape=input_shape)
    vae.compile(optimizer=optimizer)
    vae.load_weights(str(saved_model_path)).expect_partial()
    return vae

def main(argv):
    del argv # unused
    # parse the configuration file with hyperparameter settings
    gin.parse_config_file(FLAGS.gin_config)

    # make directory for results to be saved if it doe not exist
    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    saved_model_path = str(results_dir / "vae")

    # Get dataset
    train_dataset, val_dataset, test_dataset, dataset_info = Dataset()

    if FLAGS.mode == "standard":
        # perform training
        vae = Train(train_dataset, val_dataset)
        # save trained model
        vae.save_weights(saved_model_path, save_format='tf')
        # evaluate trained model 
        Eval(test_dataset, vae)

    if FLAGS.mode == "training":
        # perform training
        vae = Train(train_dataset, val_dataset)
        # save trained model
        vae.save_weights(saved_model_path, save_format='tf') 

    if FLAGS.mode == "eval":
        # evaluate pre-trained model
        vae = Load(saved_model_path, gin.REQUIRED)
        eval_results = Eval(test_dataset, vae)

    if FLAGS.mode == "2101.11181":
        # make plots for paper arXiv:2101.11181
        # load model
        vae = Load(saved_model_path, gin.REQUIRED)
        # make plots
        make_21011181_plots(test_dataset, vae)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "GNILC", ["GNILC", "MHD"], "Which dataset to use.")
    flags.DEFINE_enum("mode", "standard", ["standard", "training", "eval", "2101.11181"],  
        "Which mode to run in. Standard will train the model, and run the evaluations.")
    flags.DEFINE_string("results_dir", "/global/cscratch1/sd/bthorne/dustvaeder/results/default", "Directory to be created for results.")
    flags.DEFINE_string("gin_config", "./configs/config.gin", "File containing the gin configuration.")
    app.run(main)