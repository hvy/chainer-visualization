import argparse
import os
import math
import cupy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from chainer import Variable, serializers, cuda
from lib.models import VGG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out-dirname', type=str, default='results')
    parser.add_argument('--image-filename', type=str, default='images/cat.jpg')
    parser.add_argument('--model-filename', type=str, default='VGG.model')
    return parser.parse_args()


def save_im(filename, im):
    im = np.rollaxis(im, 0, 3)  # (c, h, w) -> (h, w, c)
    im = Image.fromarray(im)
    im.save(filename)


def save_ims(filename, ims, dpi=100, scale=0.5):
    n, c, h, w = ims.shape

    rows = int(math.ceil(math.sqrt(n)))
    cols = int(round(math.sqrt(n)))

    fig, axes = plt.subplots(rows, cols, figsize=(w*cols/dpi*scale, h*rows/dpi*scale), dpi=dpi)

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(ims[i].transpose((1, 2, 0)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
    plt.savefig(filename, dpi=dpi, bbox_inces='tight', transparent=True)
    plt.clf()
    plt.close()


def tile_ims(filename, directory):

    """Load all images in the given directory and tile them into one."""

    ims = [mpimg.imread(os.path.join(directory, f)) for f in sorted(os.listdir(directory))]
    ims = np.array(ims)
    ims = ims.transpose((0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h ,w)
    save_ims(filename, ims)


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


def visualize_layer_activations(model, im, layer_idx):

    """Compute the activations for each feature map for the given layer for
    this particular image. Note that the input x should be a mini-batch
    of size one, i.e. a single image.
    """

    if model._device_id is not None and model._device_id >= 0:  # Using GPU
        im = cupy.array(im)

    activations = model.activations(Variable(im), layer_idx)

    if isinstance(activations, cupy.ndarray):
        activations = cupy.asnumpy(activations)

    # Rescale to [0, 255]
    activations -= activations.min()
    activations /= activations.max()
    activations *= 255

    return activations.astype(np.uint8)


def visualize(out_dirname, im, model):

    def create_subdir(layer_idx):
        dirname = os.path.join(out_dirname, 'conv{}/'.format(layer_idx+1))
        dirname = os.path.dirname(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    def save_to_ims(activations, dirname):
        filename_len = len(str(len(activations)))

        # Save each feature map activation as its own image
        for i, activation in enumerate(activations):
            filename = '{num:0{width}}.png'.format(num=i, width=filename_len)
            filename = os.path.join(dirname, filename)
            save_im(filename, activation)

        # Save an image of all feature map activations in this layer
        tiled_filename = os.path.join(out_dirname, 'conv{}.png'.format(layer_idx+1))
        tile_ims(tiled_filename, dirname)

    # Visualize each of the 5 convolutional layers in VGG
    for layer_idx in range(5):
        print('Visualizing activations for conv{}...'.format(layer_idx+1))
        activations = visualize_layer_activations(model, im.copy(), layer_idx)
        dirname = create_subdir(layer_idx)
        save_to_ims(activations, dirname)


def main(args):
    gpu = args.gpu
    out_dirname = args.out_dirname
    model_filename = args.model_filename
    image_filename = args.image_filename

    model = VGG()
    serializers.load_hdf5(model_filename, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    im = read_im(image_filename)

    visualize(out_dirname, im, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
