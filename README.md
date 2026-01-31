# MNIST Digit Classifier from Scratch in C

This repository contains a standalone implementation of a feed-forward neural network designed to classify handwritten digits from the MNIST dataset. The project is written entirely in pure C and intentionally avoids high-level machine learning libraries in order to demonstrate the core mathematical principles behind neural networks.

## Overview

The project implements a simple yet complete neural network pipeline, including data loading, forward propagation, backpropagation, and evaluation. Every step is implemented manually to provide full transparency into how neural networks operate internally.

## Technical Specifications

The model uses a classic Multi-Layer Perceptron (MLP) architecture:

- Input Layer  
  784 neurons corresponding to 28×28 grayscale images.

- Hidden Layer  
  128 neurons using the Sigmoid activation function.

- Output Layer  
  10 neurons representing digits 0–9, using Sigmoid activation to produce confidence values.

- Training Algorithm  
  Stochastic Gradient Descent (SGD) with Backpropagation.

## Dataset Requirements

The following MNIST dataset files must be present in the project directory:

- train-images.idx3-ubyte — Training images  
- train-labels.idx1-ubyte — Training labels  
- t10k-images.idx3-ubyte — Test images  
- t10k-labels.idx1-ubyte — Test labels  

These files can be downloaded from the official MNIST dataset source.

## Installation and Compilation

The implementation depends only on standard C libraries:

- stdio.h  
- stdlib.h  
- math.h
- opm.h 

Since mathematical functions are used, the math library must be linked explicitly.

Compile (Linux / Unix):

    gcc mnsit.c -o mnsit -lm -fopenmp

## Execution

After compilation, run the program using:

    ./mnsit

## Training and Evaluation

- The network trains for 20 epochs by default.
- Progress information is printed every 1,000 training samples.
- After training, the model is automatically evaluated on the test set.

## Results and Performance

- Training set size: 60,000 images  
- Test set size: 10,000 images  
- Learning rate: 0.01  
- Epochs: 20  

With these hyperparameters, the model typically achieves an accuracy %97.88 on the test set. Results may vary slightly due to random weight initialization.

## Implementation Details

The project includes manual implementations of:

- Binary parsing of MNIST .ubyte files  
- Input normalization (scaling pixel values from 0–255 to 0.0–1.0)  
- Randomized weight and bias initialization  
- Forward propagation (matrix–vector multiplication and activation functions)  
- Backpropagation (error computation and gradient descent updates)

## Purpose

This project is intended for educational purposes. It prioritizes clarity and low-level understanding over performance or scalability, making it suitable for learning how neural networks work internally without abstraction layers.
