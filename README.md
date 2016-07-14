# CNN Activation Visualization

An implementation of the neural network visualization by Zeiler and Fergus using Chainer.

[Visualizing and Understanding Convolutional Networks, 2012](https://arxiv.org/pdf/1311.2901v3.pdf)

## Prerequisites

- Install Chainer
- Download the Chainer VGG model


## Run

```bash
python visualize.py
```

Simply run the `visualize.py` script. The VGG model will be feeded with one of the sample images. Feature map activations for each of the five convolutional layers in the VGG model will be stored in the `activations/` sub-directorty.

## Sample Outputs

### First layer convolutions

![](samples/conv1.jpg)

### Second layer convolutions

![](samples/conv2.jpg)

### Third layer convolutions

![](samples/conv3.jpg)

### Forth layer convolutions

![](samples/conv4.jpg)

### Fifth layer convolutions

![](samples/conv5.jpg)

## TODO

- [ ] Speed up indexed unpooling
- [ ] Support GPU
- [ ] Support Python 3.5
