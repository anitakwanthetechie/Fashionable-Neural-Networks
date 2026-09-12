# Fashionable Neural Networks 👕👗🥾

A mini deep learning project that builds, trains, and evaluates a multi-class image classification neural network on the **Fashion-MNIST** dataset using TensorFlow/Keras.

## Overview

Neural networks are one of the fundamental building blocks behind many of today's AI systems, from computer vision to natural language processing and large language models. This project explores that building block hands-on: training a fully-connected neural network to recognize 10 categories of clothing and footwear from small grayscale images, then inspecting how confident the model is in its own predictions.

The goal was simple: give the network an image it has never seen before, and see if it can figure out what the item is.

## Dataset

[Fashion-MNIST](https://keras.io/api/datasets/fashion_mnist/) is a drop-in replacement for the original MNIST digits dataset, consisting of:

- 70,000 grayscale images (60,000 train / 10,000 test), each 28×28 pixels
- 10 classes, one label per image

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

The dataset is loaded directly via `tf.keras.datasets.fashion_mnist`, so no manual download is required.

## Approach

1. **Load & explore** the data, confirm image/label shapes, and visualize a few random samples.
2. **Preprocess**: scale pixel values from `[0, 255]` to `[0, 1]`, and add a channel dimension (`28x28` → `28x28x1`) since the images are single-channel grayscale.
3. **Build** a fully-connected (dense) neural network.
4. **Train** with early stopping and automatic checkpointing of the best model.
5. **Evaluate** learning curves (accuracy/loss vs. epoch) for train vs. validation splits.
6. **Predict** on unseen test images and inspect the full per-class probability distribution, not just the top prediction.

### Model architecture

A sequential fully-connected network — no convolutional layers, by design, to keep the focus on core neural network mechanics (dense layers, batch normalization, dropout):

```
Input (28, 28, 1)
Flatten                         → 784
Dense(256) → BatchNorm → ReLU → Dropout(0.2)
Dense(128) → BatchNorm → ReLU → Dropout(0.2)
Dense(10, activation='softmax')  → 10 class probabilities
```

- **Total params:** 236,682 (235,914 trainable / 768 non-trainable)
- **Optimizer:** Adam (learning rate = 0.0001)
- **Loss:** sparse categorical cross-entropy
- **Batch size:** 64
- **Callbacks:**
  - `EarlyStopping` — monitors `val_accuracy`, stops after 5 epochs without improvement, restores best weights
  - `ModelCheckpoint` — saves the best-performing model to `best_fashion_mnist_model.keras`

## Results

Training ran for up to 50 epochs but stopped early at epoch 31 once validation accuracy plateaued. The best epoch reached:

- **~90% validation accuracy**
- **~0.30 validation loss**

Beyond the headline accuracy, this project also inspects the model's full softmax output — the probability it assigns to *every* class, not just the winning one. On one test image, the model correctly identified a bag with **99.99% confidence**, correctly assigning almost zero probability to all 9 other classes — a nice illustration of how a network can be highly and appropriately confident even from a low-resolution 28×28 image.

Learning curves for both accuracy and loss (train vs. validation) are plotted in the notebook to check for overfitting and confirm the model converged smoothly.

## Project structure

```
.
├── Fashion_NN.ipynb                        # Main notebook: data loading, model, training, evaluation
├── best_fashion_mnist_model.keras          # Saved best model checkpoint (generated when you run the notebook)
└── README.md
```

## What's next

This project intentionally uses a dense (fully-connected) network rather than a CNN, to focus on the fundamentals. A natural next step would be introducing convolutional layers to see how much accuracy improves on the same task.
