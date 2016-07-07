import numpy as np
import cv2 as cv
import chainer
from chainer import serializers
from chainer import Variable

from VGG import VGG, input_dimensions
import VGGVisualizer


if __name__ == '__main__':
    print('Starting...')

    mean = np.array([103.939, 116.779, 123.68])
    img = cv.imread('images/cat.jpg').astype(np.float32)
    img -= mean
    img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    vgg = VGG()
    serializers.load_hdf5('VGG.model', vgg)
    vgg = VGGVisualizer.from_VGG(vgg)

    reconstruction = vgg(Variable(img), None)

    n, c, h, w = reconstruction.data.shape

    # Assume a single image in batch and get it
    img = reconstruction.data[0]
    img = np.rollaxis(img, 0, 3)
    img += mean

    cv.imwrite('cat_reconstructed.jpg', img)

    print('Done')
