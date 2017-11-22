import chainer
from chainer import cuda
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class VGG(chainer.Chain):

    """Input dimensions are (n, 3, 224, 224)."""

    def __init__(self):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.fc6 = L.Linear(25088, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

        # Keep track of the pooling indices inside each function instance
        self.conv_blocks = [
            [self.conv1_1, self.conv1_2],
            [self.conv2_1, self.conv2_2],
            [self.conv3_1, self.conv3_2, self.conv3_3],
            [self.conv4_1, self.conv4_2, self.conv4_3],
            [self.conv5_1, self.conv5_2, self.conv5_3]
        ]
        self.deconv_blocks = []
        self.mps = [F.MaxPooling2D(2, 2) for _ in self.conv_blocks]

    def __call__(self, x):

        """Return a softmax probability distribution over predicted classes."""

        # Convolutional layers
        hs, _ = self.feature_map_activations(x)
        h = hs[-1]

        # Fully connected layers
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return F.softmax(h)

    def feature_map_activations(self, x):

        """Forward pass through the convolutional layers of the VGG returning
        all of its intermediate feature map activations."""

        hs = []
        pre_pooling_sizes = []

        h = x
        for conv_block, mp in zip(self.conv_blocks, self.mps):
            for conv in conv_block:
                h = F.relu(conv(h))

            pre_pooling_sizes.append(h.data.shape[2:])

            # Disable cuDNN, else pooling indices will not be stored
            with chainer.using_config('use_cudnn', 'never'):
                h = mp.apply((h,))[0]
            hs.append(h)

        return hs, pre_pooling_sizes

    def activations(self, x, layer_idx):

        """Return filter activations projected back to the input space, i.e.
        images with shape (n_feature_maps, 3, 224, 224) for a particula layer.

        The layer index is expected to be 0-based.
        """

        if x.shape[0] != 1:
            raise TypeError('Visualization is only supported for a single image at a time')

        self.check_add_deconv_layers()
        hs, unpooling_sizes = self.feature_map_activations(x)
        hs = [h.data for h in hs]

        activation_maps = []
        n_activation_maps = hs[layer_idx].shape[1]

        xp = self.xp

        for i in range(n_activation_maps):  # For each channel
            h = hs[layer_idx].copy()

            condition = xp.zeros_like(h)
            condition[0][i] = 1  # Keep one feature map and zero all other

            h = Variable(xp.where(condition, h, xp.zeros_like(h)))

            for i in reversed(range(layer_idx+1)):
                p = self.mps[i]
                h = F.upsampling_2d(h, p.indexes, p.kh, p.sy, p.ph, unpooling_sizes[i])
                for deconv in reversed(self.deconv_blocks[i]):
                    h = deconv(F.relu(h))

            activation_maps.append(h.data)

        return xp.concatenate(activation_maps)

    def check_add_deconv_layers(self, nobias=True):

        """Add a deconvolutional layer for each convolutional layer already
        defined in the network."""

        if len(self.deconv_blocks) == len(self.conv_blocks):
            return

        for conv_block in self.conv_blocks:
            deconv_block = []
            for conv in conv_block:
                out_channels, in_channels, kh, kw = conv.W.data.shape

                if isinstance(conv.W.data, cuda.ndarray):
                    initialW = cuda.cupy.asnumpy(conv.W.data)
                else:
                    initialW = conv.W.data

                deconv = L.Deconvolution2D(out_channels, in_channels,
                                           (kh, kw), stride=conv.stride,
                                           pad=conv.pad,
                                           initialW=initialW,
                                           nobias=nobias)

                if isinstance(conv.W.data, cuda.ndarray):
                    deconv.to_gpu()

                self.add_link('de{}'.format(conv.name), deconv)
                deconv_block.append(deconv)

            self.deconv_blocks.append(deconv_block)
