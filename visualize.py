import argparse
import os
import cupy
import numpy as np
from chainer import serializers, cuda
from chainer import Variable
from models.VGG import VGG
from utils import imgutil
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--out-dirname', type=str, default='results')
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


def get_activations(model, x, layer_idx):

    """Compute the activations for each feature map for the given layer for
    this particular image. Note that the input x should be a mini-batch
    of size one, i.e. a single image.
    """

    if model._device_id >= 0:  # GPU
        x = cupy.array(x)

    a = model.activations(Variable(x), layer_idx)

    if model._device_id >= 0:
        a = cupy.asnumpy(a)

    # Center at 0 with std 0.1
    a -= a.mean()
    a /= (a.std() + 1e-5)
    a *= 0.1

    # Clip to [0, 1]
    a += 0.5
    a = np.clip(a, 0, 1)

    # To RGB
    a *= 255
    a = a.astype('uint8')

    return a


def main(args):
    gpu = args.gpu
    out_dirname = args.out_dirname
    model_filename = args.model_filename
    image_filename = args.image_filename

    print('Loading VGG model... ')

    model = VGG()
    serializers.load_hdf5(model_filename, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    sample_im = read_im(image_filename)

    # Visualize each of the 5 convolutional layers in VGG
    for layer_idx in range(5):
        # Create the target directory if it doesn't already exist
        dst_dir = os.path.join(out_dirname, 'layer_{}/'.format(layer_idx))
        dst_dir = os.path.dirname(dst_dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        print('Computing activations for layer {}...'.format(layer_idx))
        activations = get_activations(model, sample_im, layer_idx)

        filename_len = len(str(len(activations)))
        for i, activation in enumerate(activations):
            im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
            filename = os.path.join(dst_dir,
                                    '{num:0{width}}.jpg'.format(num=i, width=filename_len))

            imgutil.save_im(filename, im)

        tiled_filename = os.path.join(out_dirname, 'layer_{}.jpg'.format(layer_idx))
        imgutil.tile_ims(tiled_filename, dst_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
