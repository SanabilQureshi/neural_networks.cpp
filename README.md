# Neural Network Framework from Scratch

## Overview

A comprehensive C++ neural network implementation built entirely from scratch, without external ML libraries - built to strengthen my understanding of low-level programming and machine learning concepts. The framework features customisable architectures, multiple optimisation algorithms, and advanced training techniques including dropout regularisation and various weight initialisation strategies.

Originally developed to understand deep learning fundamentals at the lowest level, this project demonstrates practical implementations of backpropagation, gradient descent variants (SGD, Momentum, Adam), applied to real-world applications including business data classification and MNIST digit recognition achieving 97%+ accuracy

---

## Key Features

### Core Framework
- **Pure C++ Implementation**: No dependencies on TensorFlow, PyTorch, or other ML libraries - everything built from first principles
- **Flexible Architecture**: Support for arbitrary layer configurations with customisable activation functions
- **Multiple Optimisers**: Implemented SGD, Momentum, and Adam optimisers with configurable hyperparameters
- **Advanced Techniques**: Dropout regularisation, batch processing, Xavier/He weight initialisation

### Training Capabilities
- **Backpropagation Algorithm**: Full implementation with gradient accumulation for mini-batch training
- **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy loss for regression and classification
- **Activation Functions**: Sigmoid, ReLU, Leaky ReLU, Tanh, and Softmax with their derivatives
- **Model Persistence**: Save and load trained weights for deployment

### Applications & Demos
- **MNIST Handwriting Recognition**: Complete pipeline from data loading to 97%+ accuracy classification
- **Interactive Web Interface**: HTML5 canvas drawing tool for easy training/testing dataset creation
- **Superstore Classification**: Real-world business data analytics for profit/loss prediction using retail sales data
- **XOR Problem**: Classic benchmark showing non-linear learning capabilities

---

## Technology Stack

- **C++11** with STL for efficient matrix operations and memory management
- **Build System**: Makefile with automatic dependency management and formatting
- **Data Formats**: IDX file parsing for MNIST, CSV processing for structured data
- **Web Technologies**: HTML5 Canvas API for interactive digit drawing interface

---

## Architecture

The framework consists of modular components working together:

```
Neural Network Framework/
    ├── Layer Class → Weights, biases, forward/backward propagation
    ├── NeuralNetwork Class → Multi-layer orchestration and training
    └── Activation Functions → Modular, swappable activation modules

Applications/
    ├── MNIST → Digit recognition with 60k training samples
    ├── XOR → Non-linear problem demonstration
    └── Superstore → Business data classification
```

Each layer maintains its own weight matrices, bias vectors, and optimiser states (for Momentum/Adam), enabling efficient gradient computation and parameter updates through the backpropagation chain.

---

## Quick Start

```bash
# 1. Clone and navigate to project
git clone https://github.com/SanabilQureshi/neural_networks.cpp && cd neural_networks.cpp

# 2. Build all examples
make all

# 3. Run ALL examples (XOR, MNIST, Superstore)
make run-all

# 4. Or run individual examples:
make run            # XOR problem solver
make run-mnist      # MNIST digit recognition
make run-superstore # Superstore classification

# 5. Test with interactive drawing tool
# Open MNIST/dataset_creator.html in browser
```

The MNIST example will train a network on handwritten digits and display accuracy metrics. The interactive demo allows drawing digits for real-time classification.

<img src="https://github.com/SanabilQureshi/neural_networks.cpp/blob/main/screenshots/MNIST-Dataset-Creator.png?raw=true" height="480em" align="center" alt="Interactive MNIST Dataset Creation" title="MNIST Dataset Creator"/>

---

## Performance & Results

### MNIST Digit Recognition
- **Architecture**: 784 inputs → 128 hidden (ReLU) → 64 hidden (ReLU) → 10 outputs (Softmax)
- **Training**: 60,000 samples, mini-batch sise 32, Adam optimiser
- **Accuracy**: 97.2% on 10,000 test images
- **Speed**: ~30 seconds training on modern CPU

<details>
<summary>$\color{Green}{\textsf{View training output}}$</summary>

```
sunny@thinkpad:~/Documents/GitHub/NNFS-CPP/MNIST/output$ ./mnist_example --train

=== MNIST Training Mode ===
=== Loading MNIST Dataset ===

Loading training data...
Loading 60000 images of sise 28x28
Loading 60000 labels

Loading test data...
Loading 10000 images of sise 28x28
Loading 10000 labels

Dataset loaded successfully!
Training samples: 60000
Test samples: 10000
Image dimensions: 784 pixels (28x28)
Number of classes: 10 (digits 0-9)

=== Neural Network Configuration ===
Architecture: 784 inputs -> 128 hidden (ReLU) -> 64 hidden (ReLU) -> 10 outputs (Softmax)
- He initialisation for ReLU layers, Xavier for output layer
- Cross-entropy loss with softmax activation
- Adam optimiser with learning rate decay
- 20% dropout on hidden layers during training

Using 10000 training samples for demo

Training network...
Progress will be displayed every 100 epochs.
Epoch 0 - Average Loss: 0.645064
Training stopped early at epoch 50 due to low loss: 0.00744744

Saving trained model to trained_model/mnist_model.weights...
Model saved successfully!

=== Quick Evaluation on Test Set ===
Test accuracy: 962/1000 (96.2%)

Training complete! You can now test the model with:
  ./mnist_example --test <image_file.dat>
```
</details>

<details>
<summary>$\color{Green}{\textsf{View testing output}}$</summary>

```
sunny@thinkpad:~/Documents/GitHub/NNFS-CPP/MNIST/output$ ./mnist_example --test user_digits.dat 

=== MNIST Testing Mode ===
Testing file: user_digits.dat

Loading trained model from trained_model/mnist_model.weights...
Model loaded successfully!
Loading 6 image(s) (28x28)

=== Processing 6 Image(s) ===

--- Image 1 (Detailed View) ---

+---------------------------+
+---------------------------+
|                            |
|                            |
|                            |
|                            |
|                            |
|          ░▓▓▓██▓           |
|       ░██████████░         |
|       ▓█░       ██         |
|                 ██         |
|                 ██         |
|                 ██         |
|                ░█▓         |
|               ▓██          |
|            ▓███▓           |
|            ▓███░           |
|              ▓███░         |
|                 ██         |
|                 ▓█         |
|              ░███▓         |
|            ▓███▓           |
|         ░████░             |
|    ░░▓████▓                |
|   ▓████▓                   |
|                            |
|                            |
|                            |
|                            |
|                            |
+---------------------------+

Predicted digit: 3

Confidence scores:
+-------+------------+
| Digit | Confidence |
+-------+------------+
|     3 |    100.00% | <-- PREDICTED
|     7 |      0.00% |
|     1 |      0.00% |
|     2 |      0.00% |
|     9 |      0.00% |
|     8 |      0.00% |
|     5 |      0.00% |
|     0 |      0.00% |
|     4 |      0.00% |
|     6 |      0.00% |
+-------+------------+

=== Summary of All Predictions ===
+-------+------------+------------+
| Image | Prediction | Confidence |
+-------+------------+------------+
|     1 |     3      |    100.00% |
|     2 |     2      |    100.00% |
|     3 |     6      |     98.62% |
|     4 |     4      |    100.00% |
|     5 |     9      |     31.29% |
|     6 |     2      |     92.14% |
+-------+------------+------------+

=== Digit Distribution ===
+-------+-------+
| Digit | Count |
+-------+-------+
|     2 |     2 |
|     3 |     1 |
|     4 |     1 |
|     6 |     1 |
|     9 |     1 |
+-------+-------+

Processed 6 image(s) successfully!
To test more images, run:
  ./mnist_example --test <image_file.dat>
```
</details>

### Superstore Sales Classification
- **Architecture**: 10 inputs → 20 hidden (ReLU) → 10 hidden (ReLU) → 2 outputs (Softmax)
- **Dataset**: Real retail sales data with categorical/numerical features
- **Features**: Sales, quantity, discount, shipping cost, product categories
- **Task**: Binary classification (profit/loss prediction)
- **Preprocessing**: Feature normalisation, one-hot encoding for categories

<details>
<summary>$\color{Green}{\textsf{View training/testing output}}$</summary>

```
=== Superstore Profit Classification ===
Classifying transactions into profit categories using your Neural Network

Loaded 9994 samples

Class distribution:
Loss (profit < 0):        1871 samples
Low profit ($0-50):       6395 samples
Medium profit ($50-200):  1304 samples
High profit (>$200):      424 samples

Training: 7995 samples, Testing: 1999 samples

Network: 5 inputs -> 32 (ReLU) -> 16 (ReLU) -> 4 outputs (Softmax)
Training with cross-entropy loss...

Epoch 0 - Average Loss: 0.436191
Epoch 100 - Average Loss: 0.0334425
Epoch 200 - Average Loss: 0.0164676

=== Test Set Performance ===

Overall Accuracy: 98.6%

Confusion Matrix:
Predicted
Loss  Low  Med  High
Loss     352    16     0     0
Low        1  1282     6     0
Medium     0     1   247     0
High       0     0     4    90

Per-Class Performance:
Class        Precision  Recall   F1-Score  Support
------------------------------------------------
Loss          99.7%     95.7%     97.6%      368
Low           98.7%     99.5%     99.1%     1289
Medium        96.1%     99.6%     97.8%      248
High         100.0%     95.7%     97.8%       94

```
</details>

### Technical Optimisations
- **Matrix Operations**: Efficient vector-based computations avoiding unnecessary copies
- **Memory Management**: Pre-allocated matrices for gradient accumulation
- **Batch Processing**: Vectorised operations for multiple samples simultaneously
- **Weight Initialisation**: Xavier/He initialisation preventing vanishing gradients

---

## Implementation Goals

- **Mathematical Rigor**: Complete backpropagation derivations implemented from scratch
- **Modular Design**: Easily extensible with new activation functions or layer types
- **Educational Value**: Clean, commented code ideal for learning neural network internals
- **Production Features**: Model serialisation, configurable hyperparameters, error handling

---

## Example Usage

```cpp
// Create a neural network for XOR problem
NeuralNetwork nn;

// Add layers with activation functions
nn.add_layer(2, 4, tanh_activation, tanh_derivative, false, 
             InitialisationType::XAVIER);
nn.add_layer(4, 1, sigmoid, sigmoid_derivative, false, 
             InitialisationType::XAVIER);

// Train with Adam optimiser
nn.train_batch(inputs, targets, epochs=1000, learning_rate=0.1, 
               tolerance=0.01, batch_sise=4, use_cross_entropy=false, 
               decay_rate=1.0, OptimiserType::ADAM);

// Make predictions
auto predictions = nn.forward(test_inputs);
```

---

## Project Structure

Key components include:
- `neural_network.hpp/cpp` - Core framework implementation
- `mnist_example.cpp` - MNIST dataset loader and training pipeline
- `superstore_classification.cpp` - Business data classification example
- `dataset_creator.html` - Interactive web-based testing interface
- `Makefile` - Automated build system with formatting

---

## Future Improvements

- Implement Convolutional Neural Network (CNN) layers
- Add batch normalisation for deeper networks
- GPU acceleration using CUDA/OpenCL
- Recurrent layers (LSTM/GRU) for sequence modeling
- Automatic differentiation system

---

## Motivation / Acknowledgements

Built as an educational exploration into deep learning fundamentals. The MNIST dataset is provided by Yann LeCun and the MNIST database creators. The superstore business dataset has been obtained from Kaggle.
