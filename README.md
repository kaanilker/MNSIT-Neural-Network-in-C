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
- **Learning Rate Scheduling:** Dynamic learning rate decay with gamma-based reduction

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
gcc mnsit.c -o mnsit -lm -fopenmp
```

**Compile (without OpenMP):**

```bash
gcc mnsit.c -o mnsit -lm
```

## Execution

After compilation, run the program using:

```bash
./mnsit
```

## Hyperparameters

The model uses the following hyperparameters (configurable at compile time):

- **Epochs:** 10
- **Initial Learning Rate:** 0.1
- **Learning Rate Decay (Gamma):** 0.5
- **Batch Size:** 64 (defined but not currently used; updates occur per sample)
- **Hidden Layer Neurons:** 256

These can be modified by changing the `#define` directives in the code:

```c
float learningRate = 0.1;
#define epoch 10
#define gamma 0.5
#define batchSize 64
#define hiddenLayer 256
```

### Learning Rate Scheduling

The current implementation includes a basic learning rate decay mechanism:
- The learning rate is reduced by multiplying with `gamma` (0.5) at epoch 5
- This means the learning rate changes from 0.1 to 0.05 after the fifth epoch
- For more advanced scheduling (e.g., step decay every N epochs), the code can be modified to use modulo operations

## Training and Evaluation

- The network trains for the specified number of epochs
- Progress is printed every 10,000 training samples during each epoch
- After each epoch completes, the model is evaluated on the entire test set (10,000 images) and accuracy is reported
- After all training completes, the last 10 test predictions (images 9990-9999) are displayed with:
  - Image index
  - Predicted class
  - True label
  - Full output probability distribution across all 10 classes
  - Correctness indicator (EVET/HAYIR)

## Results and Performance

**Current Performance with Default Hyperparameters:**

- Training set: 60,000 images  
- Test set: 10,000 images
- Epochs: 10
- Initial Learning Rate: 0.1
- Learning Rate Decay: 0.5 at epoch 5
- Hidden Layer Neurons: 256
- Activation Functions: ReLU (hidden), Sigmoid (output)

Expected performance with these settings typically achieves 97-98% test accuracy. Performance may vary slightly due to random weight initialization.

## Implementation Details

The project includes manual implementations of:

- **Binary file parsing** of MNIST `.idx3-ubyte` and `.idx1-ubyte` formats
- **Input normalization** (pixel values scaled from 0–255 to 0.0–1.0)
- **He initialization** for weights (ReLU-optimized)
- **Activation functions:** ReLU (hidden layer) and Sigmoid (output layer) with their derivatives
- **Forward propagation** with matrix-vector operations
- **Backpropagation** with proper gradient computation through ReLU and Sigmoid
- **Gradient descent updates** for weights and biases
- **Learning rate scheduling** with gamma-based decay
- **Per-epoch evaluation** for monitoring training progress
- **Detailed prediction output** for the last 10 test samples

## Code Structure

The implementation follows this flow:

1. **Activation Functions** - Sigmoid, ReLU, and their derivatives (`sigmoidTurev`, `reluTurev`)
2. **Data Loading** - MNIST binary file parsing with header skipping and normalization
3. **Weight Initialization** - He initialization for optimal convergence with ReLU activation
4. **Training Loop**
   - Learning rate decay check (currently at epoch 5)
   - Forward Propagation through hidden and output layers
   - Error Computation using one-hot encoded targets
   - Backpropagation with gradient calculation
   - Weight and bias updates using calculated gradients
   - Progress logging every 10,000 samples
5. **Per-Epoch Test Evaluation**
   - Forward propagation on entire test set
   - Prediction via argmax (finding maximum output probability)
   - Accuracy calculation and reporting
6. **Final Detailed Test Output**
   - Last 10 predictions with full probability distributions
   - Comparison with ground truth labels

## Evaluation Metrics

The model reports:
- **Test Accuracy:** Percentage of correctly classified test images (displayed after each epoch)
- **Per-Sample Output:** For the last 10 test images, displays all 10 output probabilities and the predicted class

Accuracy is calculated as:
```
Accuracy = (Number of Correct Predictions / Total Test Samples) × 100
```

Error rate can be derived as:
```
Error Rate = 100 - Accuracy
```

## Purpose

This project is intended for **educational purposes**. It prioritizes clarity and low-level understanding over performance or scalability, making it ideal for:

- Learning neural network internals without abstraction
- Understanding backpropagation mathematics and gradient descent
- Seeing how learning rate scheduling affects convergence
- Observing per-epoch training dynamics and model improvement
- Understanding activation functions and their derivatives
- Benchmarking against higher-level frameworks
- Studying the impact of hyperparameter choices

## Potential Improvements

While the current implementation is functional and educational, possible enhancements include:

- Implementing periodic step-based learning rate decay (e.g., every N epochs)
- Adding mini-batch gradient descent instead of per-sample updates
- Implementing adaptive learning rate methods (e.g., Adam, RMSprop)
- Adding validation set for early stopping
- Implementing cross-entropy loss calculation for monitoring
- Adding data augmentation for improved generalization
- Parallelizing matrix operations with OpenMP
- Adding confusion matrix for detailed error analysis

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
