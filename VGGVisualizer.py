import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
# from lib.chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from max_pooling_2d import max_pooling_2d
from unpooling_2d import unpooling_2d

F.max_pooling_2d = max_pooling_2d
F.unpooling_2d = unpooling_2d

input_dimensions = (244, 244)


def from_VGG(vgg):
    vgg_visualizer = VGGVisualizer()

    vgg_visualizer.conv1_1 = vgg.conv1_1
    vgg_visualizer.conv1_2 = vgg.conv1_2

    vgg_visualizer.deconv1_2.W = vgg.conv1_2.W
    # vgg_visualizer.deconv1_2.b = vgg.conv1_2.b
    vgg_visualizer.deconv1_1.W = vgg.conv1_1.W
    # vgg_visualizer.deconv1_1.b = vgg.conv1_1.b

    vgg_visualizer.conv2_1 = vgg.conv2_1
    vgg_visualizer.conv2_2 = vgg.conv2_2

    vgg_visualizer.deconv2_2.W = vgg.conv2_2.W
    # vgg_visualizer.deconv2_2.b = vgg.conv2_2.b
    vgg_visualizer.deconv2_1.W = vgg.conv2_1.W
    # vgg_visualizer.deconv2_1.b = vgg.conv2_1.b

    vgg_visualizer.conv3_1 = vgg.conv3_1
    vgg_visualizer.conv3_2 = vgg.conv3_2
    vgg_visualizer.conv3_3 = vgg.conv3_3

    vgg_visualizer.deconv3_3.W = vgg.conv3_3.W
    # vgg_visualizer.deconv3_3.b = vgg.conv3_3.b
    vgg_visualizer.deconv3_2.W = vgg.conv3_2.W
    # vgg_visualizer.deconv3_2.b = vgg.conv3_2.b
    vgg_visualizer.deconv3_1.W = vgg.conv3_1.W
    # vgg_visualizer.deconv3_1.b = vgg.conv3_1.b

    vgg_visualizer.conv4_1 = vgg.conv4_1
    vgg_visualizer.conv4_2 = vgg.conv4_2
    vgg_visualizer.conv4_3 = vgg.conv4_3

    vgg_visualizer.deconv4_3.W = vgg.conv4_3.W
    # vgg_visualizer.deconv4_3.b = vgg.conv4_3.b
    vgg_visualizer.deconv4_2.W = vgg.conv4_2.W
    # vgg_visualizer.deconv4_2.b = vgg.conv4_2.b
    vgg_visualizer.deconv4_1.W = vgg.conv4_1.W
    # vgg_visualizer.deconv4_1.b = vgg.conv4_1.b

    vgg_visualizer.conv5_1 = vgg.conv5_1
    vgg_visualizer.conv5_2 = vgg.conv5_2
    vgg_visualizer.conv5_3 = vgg.conv5_3

    vgg_visualizer.deconv5_3.W = vgg.conv5_3.W
    # vgg_visualizer.deconv5_3.b = vgg.conv5_3.b
    vgg_visualizer.deconv5_2.W = vgg.conv5_2.W
    # vgg_visualizer.deconv5_2.b = vgg.conv5_2.b
    vgg_visualizer.deconv5_1.W = vgg.conv5_1.W
    # vgg_visualizer.deconv5_1.b = vgg.conv5_1.b

    print '=================================='
    print vgg_visualizer.deconv1_1.W.data.shape
    print vgg.conv1_1.W.data.shape
    print '=================================='
    print '=================================='
    print '------------------------------------------------------------'
    print vgg.conv1_1.W.data[0][0][0][0]
    print vgg_visualizer.conv1_1.W.data[0][0][0][0]
    print vgg_visualizer.deconv1_1.W.data[0][0][0][0]
    print '------------------------------------------------------------'
    return vgg_visualizer


class VGGVisualizer(chainer.Chain):
    def __init__(self):
        super(VGGVisualizer, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            deconv1_2=L.Deconvolution2D(64, 64, 3, stride=1, pad=1, nobias=True),
            deconv1_1=L.Deconvolution2D(64, 3, 3, stride=1, pad=1, nobias=True),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            deconv2_2=L.Deconvolution2D(128, 128, 3, stride=1, pad=1, nobias=True),
            deconv2_1=L.Deconvolution2D(128, 64, 3, stride=1, pad=1, nobias=True),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            deconv3_3=L.Deconvolution2D(256, 256, 3, stride=1, pad=1, nobias=True),
            deconv3_2=L.Deconvolution2D(256, 256, 3, stride=1, pad=1, nobias=True),
            deconv3_1=L.Deconvolution2D(256, 128, 3, stride=1, pad=1, nobias=True),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            deconv4_3=L.Deconvolution2D(512, 512, 3, stride=1, pad=1, nobias=True),
            deconv4_2=L.Deconvolution2D(512, 512, 3, stride=1, pad=1, nobias=True),
            deconv4_1=L.Deconvolution2D(512, 256, 3, stride=1, pad=1, nobias=True),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            deconv5_3=L.Deconvolution2D(512, 512, 3, stride=1, pad=1, nobias=True),
            deconv5_2=L.Deconvolution2D(512, 512, 3, stride=1, pad=1, nobias=True),
            deconv5_1=L.Deconvolution2D(512, 512, 3, stride=1, pad=1, nobias=True)
        )
        self.visualize = 1
        """
        fc6=L.Linear(25088, 4096),
        fc7=L.Linear(4096, 4096),
        fc8=L.Linear(4096, 1000)
        """

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))

        """
        print('--- Before pooling (subset) ---')
        print (h.data[0, 0, :6, :6])
        """
        outsize1 = h.data.shape[2:]
        print 'outsize1: ', outsize1
        h, indexes1 = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        outsize2 = h.data.shape[2:]
        print 'outsize2: ', outsize2
        h, indexes2 = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        outsize3 = h.data.shape[2:]
        print 'outsize3: ', outsize3
        h, indexes3 = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        outsize4 = h.data.shape[2:]
        print 'outsize4: ', outsize4
        h, indexes4 = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        outsize5 = h.data.shape[2:]
        print 'outsize5: ', outsize5
        h, indexes5 = F.max_pooling_2d(h, 2, stride=2)

        # Reconstruction
        h = F.unpooling_2d(h, indexes5, 2, stride=2, outsize=outsize5)
        h = self.deconv5_3(F.relu(h))
        h = self.deconv5_2(F.relu(h))
        h = self.deconv5_1(F.relu(h))
        """
        h = F.relu(self.deconv5_3(h))
        h = F.relu(self.deconv5_2(h))
        h = F.relu(self.deconv5_1(h))
        """

        h = F.unpooling_2d(h, indexes4, 2, stride=2, outsize=outsize4)
        h = self.deconv4_3(F.relu(h))
        h = self.deconv4_2(F.relu(h))
        h = self.deconv4_1(F.relu(h))
        """
        h = F.relu(self.deconv4_3(h))
        h = F.relu(self.deconv4_2(h))
        h = F.relu(self.deconv4_1(h))
        """

        h = F.unpooling_2d(h, indexes3, 2, stride=2, outsize=outsize3)
        h = self.deconv3_3(F.relu(h))
        h = self.deconv3_2(F.relu(h))
        h = self.deconv3_1(F.relu(h))
        """
        h = F.relu(self.deconv3_3(h))
        h = F.relu(self.deconv3_2(h))
        h = F.relu(self.deconv3_1(h))
        """

        h = F.unpooling_2d(h, indexes2, 2, stride=2, outsize=outsize2)
        h = self.deconv2_2(F.relu(h))
        h = self.deconv2_1(F.relu(h))

        h = F.unpooling_2d(h, indexes1, 2, stride=2, outsize=outsize1)
        h = self.deconv1_2(F.relu(h))
        h = self.deconv1_1(F.relu(h))

        # Return first layer visualizations
        return h

        print('--- After pooling (subset) ---')
        print h_prim.data[0, 0, :6, :6]


        """
        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.acc = F.accuracy(h, t)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred
        """
