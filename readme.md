# MNIST Autoencoder Training

## Overview

This project implements an autoencoder model for the MNIST dataset using PyTorch. The model consists of an encoder, a bottleneck layer, and a decoder to reconstruct input images. The dataset is loaded using PyTorch's `datasets.MNIST`, and training is performed with binary cross-entropy loss.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.11.9
- PyTorch
- torchvision
- tqdm
- numpy

You can install them using:

```bash
pip install torch torchvision tqdm numpy
```

## Dataset

The MNIST dataset is automatically downloaded and loaded using `torchvision.datasets.MNIST`. A subset of 2000 images is used for both training and testing.

## Model Architecture

The `MODEL` class defines a convolutional autoencoder with:

- Convolutional layers for encoding
- Fully connected bottleneck layers
- Transposed convolutional layers for decoding
- ReLU activations
- Sigmoid activation for output

## Training Process

1. The model is trained using the Binary Cross Entropy Loss (`BCELoss`).
2. The Adam optimizer is used to update weights.
3. Training and testing losses are logged and printed per epoch.
4. The model is saved every 10 epochs.

## Model Saving

Trained models are saved periodically to the specified output directory (`base_output`). Each model checkpoint is saved as `model_<epoch>.pth`.

## Contact

For any questions or contributions, feel free to reach out!
