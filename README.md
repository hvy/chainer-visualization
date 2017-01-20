# CNN Activation Visualization

An implementation in Chainer of the neural network visualization by Zeiler and Fergus, [Visualizing and Understanding Convolutional Networks, 2013](https://arxiv.org/abs/1311.2901).

## Run

Download the pretrained VGG Chainer model following the README in [this](https://github.com/mitmul/chainer-imagenet-vgg) repository.

Then, run the `visualize.py` script as follows. The VGG model will be feeded with an image and the activations in each of the five convolutional layer will be projected back to the input space, i.e. the space of the original image of size (3, 224, 224). The projections will be stored in the specified output directory.

```bash
python visualize.py --image-filename images/cat.jpg --model-filename VGG.model --out-dirname results --gpu 0
```

You can visualize the activations for an image of arbitrary size since the image will be scaled to the size expected by the classifier.

## Samples

Activations visualized from the convolutional layers of VGG using an image of a cat. Higher resolution images are found in the `samples` directory.

### 1st Layer of Convolutions

![](samples/conv1.png)

### 2nd Layer of Convolutions

![](samples/conv2.png)

### 3rd Layer of Convolutions

![](samples/conv3.png)

### 4th Layer of Convolutions

![](samples/conv4.png)

### 5th Layer of Convolutions

![](samples/conv5.png)
