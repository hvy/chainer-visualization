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


    # Visualize fst filter
    """
    imgs = ()
    for i in range(64):
        fil = vgg.conv1_1.W.data[i]
        fil = np.rollaxis(fil, 0, 3)
        min_val = fil.min()
        fil -= min_val
        max_val = fil.max()
        fil *= ( 255.0 / max_val)
        imgs += (fil,)
        imgs += (np.zeros((3, 3, 3)),)

    vis = np.concatenate(imgs, axis=0)
    cv.imwrite('filters_conv1_1_new.jpg', vis)
    """

    vgg = VGGVisualizer.from_VGG(vgg)

    reconstruction = vgg(Variable(img), None)

    n, c, h, w = reconstruction.data.shape

    # Assume a single image in batch and get it
    img = reconstruction.data[0]
    print('Max: {}'.format(img.max()))
    print('Min: {}'.format(img.min()))
    img -= img.min()
    if img.max() > 0:
        img *= 255.0 / img.max()
    else:
        img *= 255.0

    print('img.shape: {}'.format(img.shape))
    img = np.rollaxis(img, 0, 3)
    # img += mean

    # cv.imwrite('cat_reconstructed.jpg', img)
    cv.imwrite('new_dog.jpg', img)

    print('Done')
