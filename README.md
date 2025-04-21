# Spiking Neural Network for MNIST Classification

This repository implements a Spiking Neural Network (SNN) for classifying handwritten digits from the MNIST dataset. The project demonstrates how biologically-inspired neural networks can effectively perform image classification tasks.

## Overview

Spiking Neural Networks are a class of artificial neural networks that more closely mimic the behavior of biological neurons. Unlike traditional ANNs, SNNs process information using discrete events (spikes) over time, making them more biologically plausible and potentially more energy-efficient.

This implementation:
- Converts static MNIST images to temporal spike trains using Poisson encoding
- Utilizes Leaky Integrate-and-Fire (LIF) neurons that accumulate input and fire when a threshold is reached
- Implements backpropagation through time with surrogate gradients for training

## Network Architecture

- **Input Layer**: 784 neurons (28�28 pixels)
- **Hidden Layer**: 100 LIF neurons
- **Output Layer**: 10 LIF neurons (one per digit)

## Key Features

- **Pure NumPy Implementation**: Core SNN functionality implemented using NumPy for transparency and educational purposes
- **Surrogate Gradient Learning**: Overcomes the non-differentiability of spike events
- **Visualizations**: Interactive plots showing spike trains, membrane potentials, and network dynamics
- **Customizable Parameters**: Adjustable time steps, neuron counts, membrane properties, etc.

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- PyTorch (for data loading)
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spiking-neural-network-on-mnist.git
cd spiking-neural-network-on-mnist

# Install dependencies
pip install -r requirements.txt
```

### Running the Model

```bash
uv run main.py
```

This will:
1. Load the MNIST dataset
2. Train the SNN for the specified number of epochs
3. Evaluate the model on the test set
4. Save training results and visualizations

## Visualization

The project includes React components for visualizing:
- Spike trains and firing patterns
- Membrane potential dynamics
- Weight distributions
- Encoding of static images to spike trains
- Training metrics (accuracy, loss)

## Results

The model achieves approximately 90% accuracy on the MNIST test set after training for 5 epochs.
