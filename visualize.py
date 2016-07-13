import os
import numpy as np
import cv2 as cv
from chainer import serializers
from chainer import Variable
from VGGVisualizer import VGG
import imgutil


"""
TODO
- Speed up the unpooling loop with indexes loop
- Suport GPU
"""


def sample_im():
    """Return a preprocessed (averaged and resized to VGG) sample image."""
    mean = np.array([103.939, 116.779, 123.68])
    im = cv.imread('images/cat.jpg').astype(np.float32)
    im -= mean
    im = cv.resize(im, (224, 224)).transpose((2, 0, 1))
    im = im[np.newaxis, :, :, :]
    return im


def get_activations(model, x, layer):
    """Compute the activations for each feature map for the given layer for
    this particular image. Note that the input x should be a mini-batch
    of size one, i.e. a single image.
    """
    a = model.activations(Variable(x), layer=layer+1)  # To 1-indexed
    a = a.data[0]  # Assume batch with a single image
    return post_process_activations(a)


def post_process_activations(a):
    a -= a.min()
    if a.max() > 0:
        a *= 255.0 / a.max()
    else:
        a *= 255.0
    return a


def save_activations(model, x, layer, dst_root):
    """Save feature map activations for the given image as images on disk."""

    # Create the target directory if it doesn't already exist
    dst_dir = os.path.join(dst_root, 'layer_{}/'.format(layer+1))
    dst_dir = os.path.dirname(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print('Computing activations for layer {}...'.format(layer+1))
    activations = get_activations(model, x, layer)

    # Save each activation as its own image to later tile them all into
    # a single image for a better overview
    filename_len = len(str(len(activations)))
    for i, activation in enumerate(activations):
        im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
        filename = os.path.join(dst_dir,
                                '{num:0{width}}.jpg'  # Pad with zeros
                                .format(num=i, width=filename_len))

        print('Saving image {}...'.format(filename))
        imgutil.save_im(filename, im)

    tiled_filename = os.path.join(dst_root, 'layer_{}.jpg'.format(layer+1))
    print('Saving image {}...'.format(filename))
    imgutil.tile_ims(tiled_filename, dst_dir)


if __name__ == '__main__':
    print('Preparing the model...')
    model = VGG()
    serializers.load_hdf5('VGG.model', model)

    # Visualize each of the 5 convolutional layers in VGG
    for layer in range(5):
        save_activations(model, sample_im(), layer, 'activations')

    print('Done')
