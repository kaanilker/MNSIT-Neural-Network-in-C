# MNIST Digit Classifier from Scratch in C

This repository contains a standalone implementation of a feed-forward neural network designed to classify handwritten digits from the MNIST dataset. The project is written entirely in pure C and intentionally avoids high-level machine learning libraries to demonstrate the core mathematical principles behind neural networks.

## Overview

The project implements a complete neural network pipeline, including data loading, forward propagation, backpropagation, and evaluation. Every step is implemented manually to provide full transparency into how neural networks operate at a low level. The codebase is organized into a modular structure across multiple source files, with shared function declarations in a dedicated header file.

## Technical Specifications

The model uses a Multi-Layer Perceptron (MLP) architecture with configurable hidden layer size:

- **Input Layer:** 784 neurons corresponding to 28×28 grayscale images (flattened)
- **Hidden Layer:** 256 neurons (configurable via `hiddenLayer` parameter) using ReLU activation function
- **Output Layer:** 10 neurons representing digits 0–9, using Sigmoid activation to produce probability-like confidence values
- **Training Algorithm:** Stochastic Gradient Descent (SGD) with Backpropagation
- **Weight Initialization:** He initialization for ReLU layers, Xavier-like initialization for Sigmoid output layer
- **Learning Rate Scheduling:** Adaptive learning rate decay based on per-epoch test accuracy improvement
- **Weight Checkpointing:** Best-performing weights are backed up each epoch and restored if accuracy regresses
- **Parallelism:** OpenMP directives applied to forward propagation, backward propagation, and weight update loops

## Dataset Requirements

The following MNIST dataset files must be present in the project directory:

- `train-images.idx3-ubyte` — Training images (60,000 samples)
- `train-labels.idx1-ubyte` — Training labels  
- `t10k-images.idx3-ubyte` — Test images (10,000 samples)
- `t10k-labels.idx1-ubyte` — Test labels  

These files can be downloaded from the official MNIST dataset source at https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## File Structure

```
├── mnsit.c           — Main program, training and inference logic
├── requirements.h    — Header file: static inline activation functions + function prototypes
├── requirements.c    — Implementations: weight init, bias init, data loading, zero-fill
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

### `requirements.h`
Contains:
- `static inline` definitions of all activation functions and their derivatives (`sigmoid`, `sigmoidTurev`, `relu`, `reluTurev`) — inlined directly at call sites for zero function-call overhead
- Prototypes for all utility functions defined in `requirements.c`

### `requirements.c`
Contains implementations of:
- `agirlikDoldurma` — He initialization for weight matrices
- `biasDoldurma` — Zero initialization for bias arrays
- `diziSifirlama` — Zero-fills a 1D float array
- `matrisSifirlama` — Zero-fills a 2D float matrix
- `goruntuOkuma` — Reads and normalizes image data from MNIST binary format
- `etiketOkuma` — Reads label data from MNIST binary format

### `mnsit.c`
Contains:
- Global variable declarations (weights, biases, neuron arrays, hyperparameters)
- `egitimAlgoritmasi()` — Full training loop
- `sonTestAlgoritmasi()` — Final inference on last 10 test samples
- `main()` — Entry point

## Installation and Compilation

The implementation depends only on standard C libraries:

- `stdio.h`
- `stdlib.h`
- `math.h`
- `omp.h` (for parallel processing support via OpenMP)

**Compile (Linux / Unix):**

```bash
gcc mnsit.c requirements.c -o mnsit -lm -fopenmp -O3 -ffast-math -march=native
```

Both source files must be passed to the compiler. The `-lm` flag links the math library, `-fopenmp` enables parallelism, and `-O3 -ffast-math -march=native` enable full optimization.

## Execution

After compilation, run the program using:

```bash
./mnsit
```

## Hyperparameters

The model uses the following hyperparameters defined at global scope (configurable at compile time):

- **Epochs:** 25
- **Initial Learning Rate:** 0.07
- **Learning Rate Decay (Gamma):** 0.9
- **Hidden Layer Neurons:** 256

These can be modified by changing the `#define` directives and the global variable declaration in `mnsit.c`:

```c
float learningRate = 0.07;
#define epoch 25
#define gamma 0.9
#define hiddenLayer 256
```

### Adaptive Learning Rate Scheduling

The current implementation uses a dynamic, accuracy-driven learning rate decay mechanism:

- **Epochs 1–10:** If test accuracy improves by fewer than 5 correct predictions compared to the previous epoch, the learning rate is multiplied by `gamma` (0.9)
- **Epochs 11+:** The threshold tightens — if accuracy improves by fewer than 2 correct predictions, the learning rate is multiplied by `gamma`
- This adaptive approach ensures the learning rate decays only when the model plateaus, allowing larger steps during rapid improvement phases and finer adjustments as training converges

### Weight Checkpointing

To prevent accuracy regression from bad updates:

- After each epoch where accuracy improves, the current weights and biases are saved into backup arrays (`agirliklar1Yedek`, `agirliklar2Yedek`, `bias1Yedek`, `bias2Yedek`)
- If accuracy drops in any epoch, the weights are automatically restored from the last known best checkpoint before continuing

## Training and Evaluation

- The network trains for the specified number of epochs via `egitimAlgoritmasi()`
- The current learning rate is printed at the start of each epoch
- Progress is printed every 10,000 training samples during each epoch
- After each epoch completes, the model is evaluated on the entire test set (10,000 images) and accuracy is reported
- After all training completes, `sonTestAlgoritmasi()` displays the last 10 test predictions (images 9990–9999) with:
  - Image index
  - Predicted class
  - True label
  - Full output probability distribution across all 10 classes
  - Correctness indicator (EVET/HAYIR)

## Results and Performance

**Current Performance with Default Hyperparameters:**

- Training set: 60,000 images  
- Test set: 10,000 images
- Epochs: 25
- Initial Learning Rate: 0.07
- Learning Rate Decay (Gamma): 0.9 (adaptive, accuracy-driven)
- Hidden Layer Neurons: 256
- Activation Functions: ReLU (hidden), Sigmoid (output)

Achieved test accuracy with these settings: **%98.49**. Performance may vary slightly due to random weight initialization.

## Implementation Details

The project includes manual implementations of:

- **Binary file parsing** of MNIST `.idx3-ubyte` and `.idx1-ubyte` formats via `goruntuOkuma` and `etiketOkuma`
- **Input normalization** (pixel values scaled from 0–255 to 0.0–1.0)
- **He initialization** for weights via `agirlikDoldurma` (thread-safe, no OpenMP on `rand()`)
- **Activation functions:** `static inline` ReLU and Sigmoid with their derivatives, inlined at compile time for maximum performance
- **Forward propagation** with matrix-vector operations, parallelized via OpenMP
- **Backpropagation** with proper gradient computation through ReLU and Sigmoid, parallelized via OpenMP
- **Weight update loops** parallelized via OpenMP for additional throughput
- **Adaptive learning rate scheduling** driven by per-epoch accuracy delta
- **Weight checkpointing and rollback** to preserve the best model state
- **Per-epoch evaluation** for monitoring training progress
- **Detailed prediction output** for the last 10 test samples

## Code Structure

All weights, biases, neuron arrays, backup arrays, and hyperparameters are declared as **global variables** in `mnsit.c`, allowing them to be shared seamlessly between modular functions without passing large arrays as arguments.

The implementation is organized into the following functions:

### `egitimAlgoritmasi()` — in `mnsit.c`
Handles the full training loop across all epochs:
- Adaptive learning rate decay based on accuracy improvement thresholds (epochs 1–10 vs. 11+)
- Weight rollback to backup if accuracy regresses
- Forward propagation through hidden (ReLU) and output (Sigmoid) layers — all major loops parallelized with `#pragma omp parallel for schedule(static)`
- Error computation using one-hot encoded targets
- Backpropagation with gradient calculation — all independent loops parallelized
- Weight and bias updates using computed gradients
- Progress logging every 10,000 samples
- Per-epoch evaluation on the full test set with accuracy reporting
- Weight checkpointing when accuracy improves

### `sonTestAlgoritmasi()` — in `mnsit.c`
Runs inference on the last 10 test images (9990–9999) and prints:
- Predicted class and ground truth label
- Full output probability vector
- Correctness indicator (EVET/HAYIR)

### `main()` — in `mnsit.c`
Entry point responsible for:
1. Opening the four MNIST binary files
2. Skipping file headers via dummy reads
3. Loading and normalizing image and label data via `goruntuOkuma` and `etiketOkuma`
4. Initializing weights via `agirlikDoldurma` and biases via `biasDoldurma`
5. Calling `egitimAlgoritmasi()`
6. Calling `sonTestAlgoritmasi()`

### Utility functions — in `requirements.c`
- `agirlikDoldurma` — He-initialized random weight filling
- `biasDoldurma` — Zero-fills bias arrays
- `diziSifirlama` — Zero-fills 1D float arrays
- `matrisSifirlama` — Zero-fills 2D float matrices
- `goruntuOkuma` — MNIST image file reader + normalizer
- `etiketOkuma` — MNIST label file reader

## Evaluation Metrics

The model reports:
- **Test Accuracy:** Percentage of correctly classified test images (displayed after each epoch)
- **Per-Sample Output:** For the last 10 test images, displays all 10 output probabilities and the predicted class

Accuracy is calculated as:
```
Accuracy = (Number of Correct Predictions / Total Test Samples) × 100
```

## Purpose

This project is intended for **educational purposes**. It prioritizes clarity and low-level understanding over performance or scalability, making it ideal for:

- Learning neural network internals without abstraction
- Understanding backpropagation mathematics and gradient descent
- Seeing how modular C project structure works in practice
- Seeing how adaptive learning rate scheduling affects convergence
- Observing per-epoch training dynamics and model improvement
- Understanding activation functions and their derivatives
- Studying the impact of weight checkpointing and rollback strategies
- Benchmarking against higher-level frameworks

## Potential Improvements

While the current implementation is functional and educational, possible enhancements include:

- Implementing mini-batch gradient descent instead of per-sample updates
- Implementing adaptive learning rate methods (e.g., Adam, RMSprop)
- Adding validation set for early stopping
- Implementing cross-entropy loss calculation for monitoring
- Adding data augmentation for improved generalization
- Adding confusion matrix for detailed error analysis
- Saving and loading trained weights to/from disk

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
