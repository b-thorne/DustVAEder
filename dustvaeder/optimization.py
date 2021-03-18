import tensorflow as tf

def get_l2_loss_and_gradient_function(ma, generator, sigma=None):
    """ Function to return the value and gradient of the
    L2 distance between a given map, `ma`, and a generator
    `generator` as a function of the latent space of the
    generator.

    Parameters:
    -----------
    ma: ndarray
        Image to which generator is to be compared. Of shape
        (NBATCH, xdim, ydim, channels)
    generator: Model object
        Tensorflow model object.

    Returns
    -------
    function
        Function taking vector of latent space parameters
        as input and returning vector of scalars representing
        the L2 distance.
    """
    def L2_loss_and_gradient(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            if sigma is not None:
                generated_image = gaussian_filter(generator(x), sigma)
            else:
                generated_image = generator(x)
            loss = tf.linalg.norm(generated_image - ma)
        jac = tape.gradient(loss, x)
        loss_value = tf.reshape(loss, [1])
        return loss_value, jac
    return L2_loss_and_gradient