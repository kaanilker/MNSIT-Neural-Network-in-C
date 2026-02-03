# MNIST Digit Classifier from Scratch in C

This repository contains a standalone implementation of a feed-forward neural network designed to classify handwritten digits from the MNIST dataset. The project is written entirely in pure C and intentionally avoids high-level machine learning libraries to demonstrate the core mathematical principles behind neural networks.

## Overview

The project implements a complete neural network pipeline, including data loading, forward propagation, backpropagation, and evaluation. Every step is implemented manually to provide full transparency into how neural networks operate at a low level.

## Technical Specifications

The model uses a Multi-Layer Perceptron (MLP) architecture with configurable hidden layer size:

- **Input Layer:** 784 neurons corresponding to 28×28 grayscale images (flattened)
- **Hidden Layer:** 256 neurons (configurable via `hiddenLayer` parameter) using ReLU activation function
- **Output Layer:** 10 neurons representing digits 0–9, using Sigmoid activation to produce probability-like confidence values
- **Training Algorithm:** Stochastic Gradient Descent (SGD) with Backpropagation
- **Weight Initialization:** He initialization for ReLU layers, Xavier-like initialization for Sigmoid output layer

## Dataset Requirements

The following MNIST dataset files must be present in the project directory:

- `train-images.idx3-ubyte` — Training images (60,000 samples)
- `train-labels.idx1-ubyte` — Training labels  
- `t10k-images.idx3-ubyte` — Test images (10,000 samples)
- `t10k-labels.idx1-ubyte` — Test labels  

These files can be downloaded from the official MNIST dataset source at http://yann.lecun.com/exdb/mnist/

## Installation and Compilation

The implementation depends only on standard C libraries:

- `stdio.h`
- `stdlib.h`
- `math.h`
- `omp.h` (for potential parallel processing support)

Since mathematical functions are used, the math library must be linked explicitly.

**Compile (Linux / Unix):**

```bash
gcc mnist.c -o mnist -lm -fopenmp
```

**Compile (without OpenMP):**

```bash
gcc mnist.c -o mnist -lm
```

## Execution

After compilation, run the program using:

```bash
./mnist
```

## Hyperparameters

The model uses the following hyperparameters (configurable at compile time):

- **Epochs:** 10
- **Learning Rate:** 0.1
- **Batch Size:** 64 (defined but not currently used; updates occur per sample)
- **Hidden Layer Neurons:** 256

These can be modified by changing the `#define` directives in the code:

```c
#define epoch 10
#define learningRate 0.1
#define hiddenLayer 256
```

## Training and Evaluation

- The network trains for the specified number of epochs
- Progress is printed every 500 training samples
- After training completes, the model is evaluated on the 10,000 test images
- The last 10 test predictions are displayed with detailed output probabilities
- Final test accuracy is reported as a percentage

## Results and Performance

**Current Performance with Default Hyperparameters:**

- Training set: 60,000 images  
- Test set: 10,000 images
- Epochs: 10
- Learning Rate: 0.1
- Hidden Layer Neurons: 256
- **Test Accuracy: 97.88%**

This result was achieved with the default hyperparameters. Performance may vary slightly due to random weight initialization.

## Implementation Details

The project includes manual implementations of:

- **Binary file parsing** of MNIST `.idx3-ubyte` and `.idx1-ubyte` formats
- **Input normalization** (pixel values scaled from 0–255 to 0.0–1.0)
- **He initialization** for weights (ReLU-optimized)
- **Activation functions:** ReLU (hidden layer) and Sigmoid (output layer)
- **Forward propagation** with matrix-vector operations
- **Backpropagation** with proper gradient computation through ReLU and Sigmoid
- **Gradient descent updates** for weights and biases

## Code Structure

The implementation follows this flow:

1. **Activation Functions** - Sigmoid, ReLU, and their derivatives
2. **Data Loading** - MNIST binary file parsing and normalization
3. **Weight Initialization** - He initialization for optimal convergence
4. **Training Loop**
   - Forward Propagation
   - Error Computation
   - Backpropagation with Weight Updates
5. **Test Evaluation**
   - Forward Propagation on test set
   - Prediction via argmax
   - Accuracy calculation and detailed output for last 10 predictions

## Purpose

This project is intended for **educational purposes**. It prioritizes clarity and low-level understanding over performance or scalability, making it ideal for:

- Learning neural network internals without abstraction
- Understanding backpropagation mathematics
- Seeing how gradient descent works at the implementation level
- Benchmarking against higher-level frameworks

## License

This project is licensed under the GNU General Public License v2.0 (GPL-2.0).

You are free to:
- Use this software for any purpose
- Study how the program works and modify it
- Redistribute copies
- Distribute modified versions

Under the following terms:
- Any distributed modifications must also be licensed under GPL-2.0
- Source code must be made available when distributing the software
- Changes made to the code must be documented

See the [LICENSE](LICENSE) file for the full license text or visit https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
