# CNN Activation Visualization

An implementation of the neural network visualization by Zeiler and Fergus using Chainer.

[Visualizing and Understanding Convolutional Networks, 2012](https://arxiv.org/pdf/1311.2901v3.pdf)

## Prerequisites

- Install Chainer
- Download the Chainer VGG model

## TODO

- [ ] Speed up indexed unpooling
- [ ] Support GPU
- [ ] Support Python 3.5

## Run

```bash
python visualize.py
```

Simply run the `visualize.py` script. The VGG model will be feeded with one of the sample images. Feature map activations for each of the five convolutional layers in the VGG model will be stored in the `activations/` sub-directorty.

