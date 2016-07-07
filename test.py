import numpy as np
import cv2 as cv
import chainer
from chainer import serializers
from chainer import Variable

from VGG import VGG, input_dimensions


if __name__ == '__main__':
    print('Starting...')

    mean = np.array([103.939, 116.779, 123.68])
    img = cv.imread('images/cat.jpg').astype(np.float32)
    img -= mean
    img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    vgg = VGG()
    serializers.load_hdf5('VGG.model', vgg)

    pred = vgg(Variable(img), None)

    """
    words = open('data/synset_words.txt').readlines()
    words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
    words = np.asarray(words)

    top5 = np.argsort(pred)[0][::-1][:5]
    probs = np.sort(pred)[0][::-1][:5]
    for w, p in zip(words[top5], probs):
        print('{}\tprobability:{}'.format(w, p))
    """
    print('done!')


