from absl import app
from absl import flags

import gin

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import sonnet as snt

from pathlib import Path

from datasets import load_dataset
from utils import rotate

tfkl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

FLAGS = flags.FLAGS

def ConvBatchLayer(x, filters, kernel_size, strides, activation=tf.nn.relu, momentum=0.9):
    x = tfkl.Conv2D(filters, kernel_size, strides=strides, padding='same', activation=activation)(x)
    return tfkl.BatchNormalization(momentum=momentum)(x)


def TransConvBatchLayer(x, filters, kernel_size, strides, activation=tf.nn.relu, momentum=0.9):
    x = tfkl.BatchNormalization(momentum=momentum)(x)
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

def make_21011181_plots(vae, test_dataset):
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

def Eval(vae, test_dataset):
    eval_results = vae.evaluate(test_dataset)
    return eval_results

def Train(vae, optimizer, train_dataset, val_dataset, saved_model_path):
    epochs = 20
    vae.compile(optimizer=optimizer)
    print(vae._encoder.summary())
    print(vae._decoder.summary())
    history = vae.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    vae.save_weights(saved_model_path, save_format='tf')    
    return vae

def Load(saved_model_path, optimizer, input_shape):
    vae = VAE(input_shape=input_shape)
    vae.compile(optimizer=optimizer)
    vae.load_weights(str(saved_model_path)).expect_partial()
    return vae

def vqvae():
    num_training_updates = 100000

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
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
    vq_use_ema = True
    decay = 0.99    
    vq_vae = snt.nets.VectorQuantizerEMA(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay)
    train_data_variance = 1
    model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                    data_variance=train_data_variance)
    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []
    return 



def main(argv):
    del argv # unused
    BATCH_SIZE = 1
    input_shape = (256, 256, 1)
    num_filters_top = 256
    epochs = FLAGS.epochs
    beta_1 = 0.5
    learning_rate = 1e-4
    beta_disentangle = 5
    optimizer = tfk.optimizers.Adam(beta_1=beta_1, learning_rate=learning_rate)

    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    saved_model_path = str(results_dir / "vae")

    train_dataset, val_dataset, test_dataset, dataset_info = load_dataset(FLAGS.dataset, batch_size=BATCH_SIZE)

    if FLAGS.mode == "standard":
        vae = VAE(input_shape=dataset_info['input_shape'])
        vae = Train(vae, optimizer, train_dataset, val_dataset, saved_model_path)
        Eval(vae, test_dataset)

    if FLAGS.mode == "training":
        _ = Train(vae, optimizer, train_dataset, val_dataset, saved_model_path)

    if FLAGS.mode == "eval":
        vae = Load(saved_model_path, optimizer, dataset_info['input_shape'])
        eval_results = Eval(vae, test_dataset)

    if FLAGS.mode == "2101.11181":
        vae = Load(saved_model_path, optimizer, dataset_info['input_shape'])
        make_21011181_plots(vae, test_dataset)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "GNILC", ["GNILC", "MHD"], "Which dataset to use")
    flags.DEFINE_enum("mode", "standard", ["standard", "training", "eval", "2101.11181"],  
        "Which mode to run in. Standard will train the model, and run the evaluations.")
    flags.DEFINE_string("results_dir", "/global/cscratch1/sd/bthorne/dustvaeder/results/default", "Directory to be created for results.")
    flags.DEFINE_integer("epochs", 10, "Number of epochs to train for")
    app.run(main)