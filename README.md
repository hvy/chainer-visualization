# CNN Activation Visualization

An implementation in Chainer of the neural network visualization by Zeiler and Fergus, [Visualizing and Understanding Convolutional Networks, 2013](https://arxiv.org/abs/1311.2901).

## Run

Download the pretrained VGG Chainer model following the README in [this](https://github.com/mitmul/chainer-imagenet-vgg) repository.

Then, run the `visualize.py` script as follows. The VGG model will be feeded with an image and the activations in each of the five convolutional layer will be projected back to the input space, i.e. the space of the original image of size (3, 224, 224). The projections will be stored in the specified output directory.

```bash
python visualize.py --image-filename images/cat.jpg --model-filename VGG.model --out-dirname results --gpu 0 
```

## Sample Outputs

### First layer of convolutions

![](samples/cat/conv1.jpg)

### Second layer of convolutions

![](samples/cat/conv2.jpg)

### Third layer of convolutions

![](samples/cat/conv3.jpg)

### Forth layer of convolutions

![](samples/cat/conv4.jpg)

### Fifth layer of convolutions

![](samples/cat/conv5.jpg)
