import argparse
import os
import cupy
import numpy as np
# import cv2 as cv
from chainer import serializers, cuda
from chainer import Variable
from models.VGG import VGG
from utils import imgutil

from PIL import Image

"""
TODO
- Speed up the unpooling loop with indexes loop
- Suport GPU
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--image-filename', type=str, default='images/cat.jpg')
    parser.add_argument('--model-filename', type=str, default='VGG.model')
    return parser.parse_args()


def read_im(filename):

    """Return a preprocessed (averaged and resized to VGG) sample image."""

    mean = np.array([103.939, 116.779, 123.68])
    im = Image.open(filename)
    im = im.resize((224, 224))

    im = np.array(im, dtype=np.float32)
    im -= mean
    im = im.transpose((2, 0, 1))
    im = im[np.newaxis, :, :, :]

    return im


def get_activations(model, x, layer):

    """
    Compute the activations for each feature map for the given layer for
    this particular image. Note that the input x should be a mini-batch
    of size one, i.e. a single image.
    """

    if model._device_id >= 0:  # GPU
        x = cupy.array(x)

    a = model.activations(Variable(x), layer=layer)

    if model._device_id >= 0:
        a = cupy.asnumpy(a)

    a = post_process_activations(a)
    return a


def post_process_activations(a):
    # Center at 0 with std 0.1
    a -= a.mean()
    a /= (a.std() + 1e-5)
    a *= 0.1

    # Clip to [0, 1]
    a += 0.5
    a = np.clip(a, 0, 1)

    # To RGB
    a *= 255
    a = np.clip(a, 0, 255).astype('uint8')

    return a


def save_activations(model, x, layer, dst_root):
    """Save feature map activations for the given image as images on disk."""

    # Create the target directory if it doesn't already exist
    dst_dir = os.path.join(dst_root, 'layer_{}/'.format(layer+1))
    dst_dir = os.path.dirname(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print('Computing activations for layer {}...'.format(layer+1))
    activations = get_activations(model, x, layer+1)

    # Save each activation as its own image to later tile them all into
    # a single image for a better overview
    filename_len = len(str(len(activations)))
    for i, activation in enumerate(activations):
        im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
        # im = activation
        print('printing', im.shape)
        filename = os.path.join(dst_dir,
                                '{num:0{width}}.jpg'  # Pad with zeros
                                .format(num=i, width=filename_len))

        print('Saving image {}...'.format(filename))
        imgutil.save_im(filename, im)

    tiled_filename = os.path.join(dst_root, 'layer_{}.jpg'.format(layer+1))
    print('Saving image {}...'.format(filename))
    imgutil.tile_ims(tiled_filename, dst_dir)


def main(args):
    gpu = args.gpu
    model_filename = args.model_filename
    image_filename = args.image_filename

    print('Loading the model... ')

    model = VGG()
    serializers.load_hdf5(model_filename, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    sample_im = read_im(image_filename)

    # Visualize each of the 5 convolutional layers in VGG
    for layer in range(5):
        save_activations(model, sample_im, layer, 'activations')

if __name__ == '__main__':
    args = parse_args()
    main(args)
