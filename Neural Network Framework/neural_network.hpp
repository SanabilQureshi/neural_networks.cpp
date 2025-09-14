#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Utility Functions
// ============================================================================

inline void print_vector(const std::vector<std::vector<double>> &x)
{
    for (size_t i = 0; i < x.size(); ++i)
    {
        for (size_t j = 0; j < x[i].size(); ++j)
        {
            std::cout << x[i][j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

inline void error_throw(std::string error_message)
{
    throw std::invalid_argument(error_message);
}

enum class InitializationType
{
    UNIFORM,
    XAVIER,
    HE
};

enum class OptimizerType
{
    SGD,      // Standard gradient descent
    MOMENTUM, // SGD with momentum
    ADAM      // Adam optimizer
};

inline std::vector<std::vector<double>> initialize_random_matrix(
    int rows, int cols, bool randomize = true, InitializationType init_type = InitializationType::UNIFORM,
    int fan_in = 0, int fan_out = 0)
{
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));

    if (randomize)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        if (init_type == InitializationType::XAVIER)
        {
            // Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
            double scale = std::sqrt(2.0 / (fan_in + fan_out));
            std::normal_distribution<> dis(0.0, scale);

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    matrix[i][j] = dis(gen);
                }
            }
        }
        else if (init_type == InitializationType::HE)
        {
            // He initialization: scale = sqrt(2 / fan_in)
            double scale = std::sqrt(2.0 / fan_in);
            std::normal_distribution<> dis(0.0, scale);

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    matrix[i][j] = dis(gen);
                }
            }
        }
        else
        {
            // Default uniform initialization
            std::uniform_real_distribution<> dis(-1.0, 1.0);

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    matrix[i][j] = dis(gen);
                }
            }
        }
    }

    return matrix;
}

// ============================================================================
// Activation Functions
// ============================================================================

inline double heaviside(double value)
{
    return (value >= 0) ? 1.0 : 0.0;
}

inline double sigmoid(double value)
{
    return 1 / (1 + exp(-value));
}

inline double sigmoid_derivative(double value)
{
    double s = sigmoid(value);
    return s * (1 - s);
}

inline double sigmoid_derivative_from_output(double sigmoid_output)
{
    return sigmoid_output * (1 - sigmoid_output);
}

inline double relu(double value)
{
    return std::max(0.0, value);
}

inline double relu_derivative(double value)
{
    return (value > 0.0) ? 1.0 : 0.0;
}

inline double tanh_activation(double value)
{
    return std::tanh(value);
}

inline double tanh_derivative(double value)
{
    double t = std::tanh(value);
    return 1.0 - t * t;
}

inline double leaky_relu(double value, double alpha = 0.01)
{
    return (value > 0.0) ? value : alpha * value;
}

inline double leaky_relu_derivative(double value, double alpha = 0.01)
{
    return (value > 0.0) ? 1.0 : alpha;
}

inline std::vector<double> softmax(const std::vector<double> &values)
{
    std::vector<double> result(values.size());
    double max_val = *std::max_element(values.begin(), values.end());

    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
    {
        result[i] = std::exp(values[i] - max_val);
        sum += result[i];
    }

    for (size_t i = 0; i < values.size(); ++i)
    {
        result[i] /= sum;
    }

    return result;
}

// ============================================================================
// Neural Network Classes
// ============================================================================

class Layer
{
  public:
    std::vector<std::vector<double>> weights;     // Weight matrix
    std::vector<std::vector<double>> biases;      // Bias vector
    std::vector<std::vector<double>> z_values;    // Weighted inputs (z = Wx + b)
    std::vector<std::vector<double>> activations; // Activated outputs (a = activation(z))
    std::vector<std::vector<double>> gradients;   // Gradients for backprop
    std::vector<std::vector<double>> inputs;      // Stored inputs for backprop

    std::function<double(double)> activation_fn;         // Activation function
    std::function<double(double)> activation_derivative; // Derivative of activation function
    bool use_softmax;                                    // Flag for softmax activation
    double dropout_rate;                                 // Dropout rate (0.0 = no dropout)
    bool training_mode;                                  // Flag to enable/disable dropout during training

    // Accumulated gradients for batch processing
    std::vector<std::vector<double>> weight_gradients;
    std::vector<std::vector<double>> bias_gradients;
    int batch_count;

    // Optimizer state variables (for momentum and Adam)
    std::vector<std::vector<double>> weight_velocity; // For momentum/Adam
    std::vector<std::vector<double>> bias_velocity;   // For momentum/Adam
    std::vector<std::vector<double>> weight_squared;  // For Adam only
    std::vector<std::vector<double>> bias_squared;    // For Adam only

    // Constructor: Initialize weights, biases, and activation functions
    Layer(int input_size, int output_size, std::function<double(double)> act_fn,
          std::function<double(double)> act_deriv, bool is_softmax = false,
          InitializationType init_type = InitializationType::UNIFORM, double dropout = 0.0);

    // Carry out a forward pass for this layer
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &inputs_);

    // Calculate gradients for output layer
    std::vector<std::vector<double>> backward_output_layer(const std::vector<std::vector<double>> &target_output,
                                                           bool use_cross_entropy = false);

    // Calculate gradients for hidden layer
    std::vector<std::vector<double>> backward_hidden_layer(
        const std::vector<std::vector<double>> &next_layer_weights,
        const std::vector<std::vector<double>> &next_layer_gradients);

    // Update weights and biases
    void update_parameters(double learning_rate, OptimizerType optimizer = OptimizerType::SGD,
                           double momentum_beta = 0.9, double adam_beta1 = 0.9, double adam_beta2 = 0.999);

  private:
    std::vector<std::vector<double>> calculate_output_layer_error(
        const std::vector<std::vector<double>> &network_output, const std::vector<std::vector<double>> &target_output);

    std::vector<std::vector<double>> calculate_output_layer_gradients(
        const std::vector<std::vector<double>> &network_output, const std::vector<std::vector<double>> &output_error);

    std::vector<std::vector<double>> dot_product(const std::vector<std::vector<double>> &x,
                                                 const std::vector<std::vector<double>> &y);

    std::vector<std::vector<double>> add_bias(const std::vector<std::vector<double>> &x,
                                              const std::vector<std::vector<double>> &bias);

    std::vector<std::vector<double>> activation(const std::vector<std::vector<double>> &x,
                                                std::function<double(double)> activation_fn);
};

class NeuralNetwork
{
  public:
    std::vector<Layer> layers;
    bool use_cross_entropy; // Flag to use cross-entropy loss instead of MSE

    // Constructor
    NeuralNetwork();

    // Add a layer to the network with specific activation and derivative functions
    void add_layer(int input_size, int output_size, std::function<double(double)> act_fn,
                   std::function<double(double)> act_deriv, bool is_softmax = false,
                   InitializationType init_type = InitializationType::UNIFORM, double dropout = 0.0);

    // Set training mode for all layers (enables/disables dropout)
    void set_training_mode(bool training);

    // Forward pass through entire network
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input);

    // Calculate MSE loss
    double MSE(const std::vector<std::vector<double>> &y_pred, const std::vector<std::vector<double>> &y_act);

    // Calculate cross-entropy loss
    double cross_entropy(const std::vector<std::vector<double>> &y_pred, const std::vector<std::vector<double>> &y_act);

    // Train the network on batch of examples
    void train_batch(const std::vector<std::vector<double>> &all_inputs,
                     const std::vector<std::vector<double>> &all_targets, int epochs, double learning_rate,
                     double tolerance = 0.001, int batch_size = 1, bool use_cross_entropy = false,
                     double decay_rate = 1.0, OptimizerType optimizer = OptimizerType::SGD);
    
    // Save network weights to file
    bool save_weights(const std::string &filename);
    
    // Load network weights from file
    bool load_weights(const std::string &filename);
};

#endif // NEURAL_NETWORK_HPP