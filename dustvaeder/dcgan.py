""" Implementation of the deep convolutional generative adversarial network.

A GAN consists of two logically separate networks, a discriminator and a
generator. These train antagonistically, and it makes sense to keep them
separate from a code point of view. 
"""
import tensorflow as tf
import tensorflow.keras.layers as lyr
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

def build_discriminator(filters, kernels, strides, input_shape):
    """ Function to create and return a `keras` model for the discriminator.
    """ 
    model = Sequential(name='Discriminator')
    model.add(
        lyr.Conv2D(
            filters[0], 
            kernels[0], 
            strides=strides[0], 
            input_shape=input_shape, 
            padding="same", 
            kernel_initializer="he_normal",
            name="Conv2D_D1"
            ))

    model.add(lyr.LeakyReLU(alpha=0.2, name="LRelu_D1"))

    for i, (flter, kernel, stride) in enumerate(zip(filters[1:], kernels[1:], strides[1:])):
        model.add(
            lyr.Conv2D(
                flter, 
                kernel, 
                strides=stride, 
                padding='same', 
                kernel_initializer='he_normal',
                name="Conv2D_D{:d}".format(i + 2)
                ))
        
        model.add(lyr.BatchNormalization(momentum=0.9, name="BNorm_D{:d}".format(i + 1)))
        
        model.add(lyr.LeakyReLU(alpha=0.2, name="LRelu_D{:d}".format(i + 2)))

    # Finally flatten and add a single densely connected layer, followed by an activation.
    model.add(lyr.Flatten(name="Flatten"))
    model.add(lyr.Dense(1, kernel_initializer='he_normal', name='Dense_D2'))
    model.add(lyr.Activation('sigmoid', name='Sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model


def build_generator(discriminator_model, filters, kernels, strides,
    latent_dim, input_shape):
    """ Function to create and reutrn a `keras` model for the generator.
    """
    #Get the size of the initial features from the size of the final feature layer in 
    #the discriminator
    dim1 = discriminator_model.get_layer('Flatten').input_shape[1]
    dim2 = discriminator_model.get_layer('Flatten').input_shape[2]
    
    # Get shape of input image and number of channels
    img_dim = input_shape[0]
    try:
        assert img_dim == input_shape[1]
    except AssertionError:
        raise AssertionError("Square image required, check input_shape")
    channels = input_shape[2]

    # Generator is sequential model
    model = Sequential(name="Generator")

    # First layer of generator is densely connected
    model.add(
        lyr.Dense(
            dim1 * dim2 * filters[-1], 
            input_dim=latent_dim,
            kernel_initializer='he_normal',
            name='Dense_G'
            ))
    
    model.add(lyr.Reshape((dim1, dim2, filters[-1]), name='Reshape'))
    
    model.add(
        lyr.BatchNormalization(
            momentum=0.9, 
            name='BNorm_G1'
            ))
    
    model.add(lyr.LeakyReLU(alpha=0.2, name='LRelu_G1'))

    # Iterate over layers defined by the number of kernels and strides
    # Depth scale decreased from second last element to first element.
    for i, (flter, kernel, stride) in enumerate(zip(filters[-2::-1], kernels[:-1], strides[:-1])):
        model.add(
            lyr.UpSampling2D(
                stride,
                name='UpSample_{:d}'.format(i + 1), 
                interpolation='bilinear'
                ))

        model.add(
            lyr.Conv2D(
                flter, 
                kernel, 
                strides=1, 
                padding='same',
                kernel_initializer='he_normal',
                name='Conv2D_G{:d}'.format(i + 1)
                ))

        model.add(
            lyr.BatchNormalization(
                momentum=0.9, 
                name='BN_G{:d}'.format(i + 2)
                ))
        
        model.add(lyr.LeakyReLU(alpha=0.2, name='LRelu_G{:d}'.format(i + 2)))
            
    model.add(
        lyr.UpSampling2D(
            strides[-1],
            name='UpSample_{:d}'.format(i + 2), 
            interpolation='bilinear'
            ))

    model.add(
        lyr.Conv2D(
            channels, 
            kernels[-1], 
            strides=1, 
            padding='same',
            kernel_initializer='he_normal',
            name='Conv2D_G{:d}'.format(i + 2)
            ))
    
    model.add(lyr.Activation('tanh', name='Tanh'))
    # If the output of the last layer is larger than the input for the discriminator crop
    # the image
    c_r = int((model.get_layer('Tanh').output_shape[1] - img_dim) / 2)
    c_c = int((model.get_layer('Tanh').output_shape[2] - img_dim) / 2)
    
    model.add(
        lyr.Cropping2D(cropping=((c_c, c_c), (c_r, c_r)), name='Crop2D'))
    return model


def build_adversarial_model(discriminator, generator):
    """ Function to create and return a `keras` model for the combined 
    generator + discriminator adversarial  network.
    """
    for layer in discriminator.layers:
        layer.trainable = False
    model = Sequential(name="Adversarial Model")
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model


def training_schedule(discriminator, generator, adversarial_model, training_dataset,
    latent_dim=32, epochs=50, callback=False):
    """ Function to execute a training schedule for the GAN. 

    Each iteration of the GAN training process consists of two steps:

    i) Train the discriminator on a combination of real and fake data,  updating
    its weights. 

    ii) Hold the discriminator weights constant, and train the adversarial
    network. To train the adversarial network we give it a random vectors paired
    with a valid target. This trains the generator part of the GAN to produce
    images that the discriminator will classify as valid.

    Parameters
    ----------
    nbatch: int
        Size of batches to train on. This is only really useful if the training
        data can not fit in memory. If training data fits in memory, just use all
        at once.
    epochs: int
        Number of epochs (passes through whole data set) to train.
    """
    image_lat = np.random.randn(1, latent_dim)
    for epoch in epochs:
        print("Epoch: ", epoch)
        for step, image_batch in enumerate(training_dataset):
            if callback:
                tf.summary.experimental.set_step(step)
            batch_size = len(image_batch)
            # First train the discriminator with correct labels
            # Randomly select batch from training samples
            y_real = np.random.binomial(1, 0.99, size=[batch_size, 1])
            y_fake = np.random.binomial(1, 0.01, size=[batch_size, 1])

            # Use `generator` to create fake images.
            noise = np.random.normal(loc=0., scale=1., size=[batch_size, latent_dim])
            fake_images = generator.predict(noise)

            # Train the discriminator on real and fake images.
            real_loss = discriminator.train_on_batch(image_batch, y_real)
            fake_loss = discriminator.train_on_batch(fake_images, y_fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            # Now train the adversarial network.
            # Create new fake images, and label as if they are from the training set.
            # Lie indicates that we are tricking the adversarial network by
            # telling it the target is valid, when in reality the discriminator
            # is being fed fake images by the generator.
            y_lie = np.ones([batch_size, 1])
            noise = np.random.normal(loc=0., scale=1., size=[batch_size, latent_dim])
            a_loss = adversarial_model.train_on_batch(noise, y_lie)
            if callback:
                tf.summary.image('random_draw', generator.predict(image_lat))
                tf.summary.scalar('aloss', a_loss)
                tf.summary.scalar('dloss', d_loss)
            print("Step number {:05d}, GAN loss is {:.03f}".format(step, a_loss))
    return adversarial_model