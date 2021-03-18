""" This submodule contains a VQ-VAE model, based on an implementation
using the Sonnet library (an API based on tensorflow). It is heavily 
based on the public code for the model described in https://arxiv.org/abs/1906.00446,
which can be found at: 
https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
"""

import tensorflow as tf 
import numpy as np

__all__ = ['ResidualStack', 'Encoder', 'Decoder', 'VQVAEModel']

class ResidualStack(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
        name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = snt.Conv2D(
                output_channels=num_residual_hiddens,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="res3x3_%d" % i)
            conv1 = snt.Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))
        
    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


class Encoder(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = snt.Conv2D(
            output_channels=self._num_hiddens // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")
        self._enc_2 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")
        self._enc_3 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def __call__(self, x):
        h = tf.nn.relu(self._enc_1(x))
        h = tf.nn.relu(self._enc_2(h))
        h = tf.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


class Decoder(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                name=None, channels=1):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = snt.Conv2DTranspose(
            output_channels=self._num_hiddens // 2,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")
        self._dec_3 = snt.Conv2DTranspose(
            output_channels=channels,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")
        
    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon
        

class VQVAEModel(snt.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1, 
                data_variance, name=None):
        super(VQVAEModel, self).__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def __call__(self, inputs, is_training):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, is_training=is_training)
        x_recon = self._decoder(vq_output['quantize'])
        recon_error = tf.reduce_mean((x_recon - inputs) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output': vq_output,
        }