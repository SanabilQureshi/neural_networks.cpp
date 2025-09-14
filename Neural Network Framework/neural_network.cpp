#include "neural_network.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

// Layer implementation
Layer::Layer(int input_size, int output_size, std::function<double(double)> act_fn,
             std::function<double(double)> act_deriv, bool is_softmax, InitializationType init_type, double dropout)
    : activation_fn(act_fn), activation_derivative(act_deriv), use_softmax(is_softmax), dropout_rate(dropout),
      training_mode(false), batch_count(0)
{
    weights = initialize_random_matrix(input_size, output_size, true, init_type, input_size, output_size);
    biases = initialize_random_matrix(1, output_size, false);
    weight_gradients = initialize_random_matrix(input_size, output_size, false);
    bias_gradients = initialize_random_matrix(1, output_size, false);

    // Initialize optimizer state variables (all start at zero)
    weight_velocity = initialize_random_matrix(input_size, output_size, false);
    bias_velocity = initialize_random_matrix(1, output_size, false);
    weight_squared = initialize_random_matrix(input_size, output_size, false);
    bias_squared = initialize_random_matrix(1, output_size, false);
}

std::vector<std::vector<double>> Layer::forward(const std::vector<std::vector<double>> &inputs_)
{
    inputs = inputs_;

    // Compute z = Wx + b
    z_values = dot_product(inputs, weights);
    z_values = add_bias(z_values, biases);

    // Compute activations a = activation(z)
    if (use_softmax)
    {
        // Apply softmax activation
        activations = z_values; // Copy structure
        for (size_t i = 0; i < z_values.size(); ++i)
        {
            activations[i] = softmax(z_values[i]);
        }
    }
    else
    {
        activations = activation(z_values, activation_fn);
    }

    // Apply dropout if in training mode and dropout rate > 0
    if (training_mode && dropout_rate > 0.0)
    {
        for (size_t i = 0; i < activations.size(); ++i)
        {
            for (size_t j = 0; j < activations[i].size(); ++j)
            {
                // Randomly drop neurons with probability dropout_rate
                double random_value = static_cast<double>(rand()) / RAND_MAX;
                if (random_value < dropout_rate)
                {
                    activations[i][j] = 0.0;
                }
                else
                {
                    // Scale the remaining activations
                    activations[i][j] /= (1.0 - dropout_rate);
                }
            }
        }
    }

    return activations;
}

std::vector<std::vector<double>> Layer::backward_output_layer(const std::vector<std::vector<double>> &target_output,
                                                              bool use_cross_entropy)
{
    if (use_cross_entropy && use_softmax)
    {
        // For softmax with cross-entropy, gradient is simply (y_pred - y_true)
        int rows = activations.size();
        int cols = activations[0].size();
        gradients = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                gradients[i][j] = activations[i][j] - target_output[i][j];
            }
        }
    }
    else
    {
        // Calculate output error
        std::vector<std::vector<double>> output_error = calculate_output_layer_error(activations, target_output);

        // Calculate gradients
        gradients = calculate_output_layer_gradients(activations, output_error);
    }

    return gradients;
}

std::vector<std::vector<double>> Layer::backward_hidden_layer(
    const std::vector<std::vector<double>> &next_layer_weights,
    const std::vector<std::vector<double>> &next_layer_gradients)
{
    int num_neurons = activations[0].size();
    gradients = std::vector<std::vector<double>>(1, std::vector<double>(num_neurons, 0.0));

    for (int i = 0; i < num_neurons; ++i)
    {
        for (size_t j = 0; j < next_layer_gradients[0].size(); ++j)
        {
            gradients[0][i] += next_layer_gradients[0][j] * next_layer_weights[i][j];
        }
        gradients[0][i] *= activation_derivative(activations[0][i]);
    }

    return gradients;
}

void Layer::update_parameters(double learning_rate, OptimizerType optimizer, double momentum_beta, double adam_beta1,
                              double adam_beta2)
{
    int rows = weights.size();
    int cols = weights[0].size();
    static int t = 1; // Time step for Adam (shared across all layers)

    if (optimizer == OptimizerType::SGD)
    {
        // Standard gradient descent
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                weights[i][j] -= learning_rate * weight_gradients[i][j];
            }
        }
        for (size_t j = 0; j < biases[0].size(); ++j)
        {
            biases[0][j] -= learning_rate * bias_gradients[0][j];
        }
    }
    else if (optimizer == OptimizerType::MOMENTUM)
    {
        // SGD with momentum
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                weight_velocity[i][j] = momentum_beta * weight_velocity[i][j] + weight_gradients[i][j];
                weights[i][j] -= learning_rate * weight_velocity[i][j];
            }
        }
        for (size_t j = 0; j < biases[0].size(); ++j)
        {
            bias_velocity[0][j] = momentum_beta * bias_velocity[0][j] + bias_gradients[0][j];
            biases[0][j] -= learning_rate * bias_velocity[0][j];
        }
    }
    else if (optimizer == OptimizerType::ADAM)
    {
        // Adam optimizer
        double epsilon = 1e-8;

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                // Update biased first moment estimate
                weight_velocity[i][j] = adam_beta1 * weight_velocity[i][j] + (1 - adam_beta1) * weight_gradients[i][j];
                // Update biased second raw moment estimate
                weight_squared[i][j] = adam_beta2 * weight_squared[i][j] +
                                       (1 - adam_beta2) * weight_gradients[i][j] * weight_gradients[i][j];

                // Compute bias-corrected moment estimates
                double m_hat = weight_velocity[i][j] / (1 - std::pow(adam_beta1, t));
                double v_hat = weight_squared[i][j] / (1 - std::pow(adam_beta2, t));

                // Update weights
                weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }

        for (size_t j = 0; j < biases[0].size(); ++j)
        {
            // Update biased moment estimates for biases
            bias_velocity[0][j] = adam_beta1 * bias_velocity[0][j] + (1 - adam_beta1) * bias_gradients[0][j];
            bias_squared[0][j] =
                adam_beta2 * bias_squared[0][j] + (1 - adam_beta2) * bias_gradients[0][j] * bias_gradients[0][j];

            // Compute bias-corrected moment estimates
            double m_hat = bias_velocity[0][j] / (1 - std::pow(adam_beta1, t));
            double v_hat = bias_squared[0][j] / (1 - std::pow(adam_beta2, t));

            // Update biases
            biases[0][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
        t++; // Increment time step
    }

    // Reset accumulated gradients
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weight_gradients[i][j] = 0.0;
        }
    }
    for (size_t j = 0; j < biases[0].size(); ++j)
    {
        bias_gradients[0][j] = 0.0;
    }
    batch_count = 0;
}

std::vector<std::vector<double>> Layer::calculate_output_layer_error(
    const std::vector<std::vector<double>> &network_output, const std::vector<std::vector<double>> &target_output)
{
    int rows = network_output.size();
    int cols = network_output[0].size();

    std::vector<std::vector<double>> error(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            error[i][j] = network_output[i][j] - target_output[i][j];
        }
    }
    return error;
}

std::vector<std::vector<double>> Layer::calculate_output_layer_gradients(
    const std::vector<std::vector<double>> &network_output, const std::vector<std::vector<double>> &output_error)
{
    int rows = network_output.size();
    int cols = network_output[0].size();
    std::vector<std::vector<double>> gradients(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            gradients[i][j] = output_error[i][j] * activation_derivative(network_output[i][j]);
        }
    }
    return gradients;
}

std::vector<std::vector<double>> Layer::dot_product(const std::vector<std::vector<double>> &x,
                                                    const std::vector<std::vector<double>> &y)
{
    int x_rows = x.size();
    int x_cols = x[0].size();
    int y_rows = y.size();
    int y_cols = y[0].size();

    if (x_cols != y_rows)
    {
        std::string error_concatenated =
            "Invalid dimensions for matrix multiplication, you have tried to multiply a (" + std::to_string(x_rows) +
            "x" + std::to_string(x_cols) + ") by a (" + std::to_string(y_rows) + "x" + std::to_string(y_cols) + ")!\n";
        error_throw(error_concatenated);
    }

    std::vector<std::vector<double>> C(x_rows, std::vector<double>(y_cols, 0));

    for (int i = 0; i < x_rows; ++i)
    {
        for (int j = 0; j < y_cols; ++j)
        {
            for (int k = 0; k < x_cols; ++k)
            {
                C[i][j] += x[i][k] * y[k][j]; // dot product
            }
        }
    }

    return C;
}

std::vector<std::vector<double>> Layer::add_bias(const std::vector<std::vector<double>> &x,
                                                 const std::vector<std::vector<double>> &bias)
{
    int x_rows = x.size();
    int x_cols = x[0].size();

    std::vector<std::vector<double>> temp = x;

    for (int i = 0; i < x_rows; ++i)
    {
        for (int j = 0; j < x_cols; ++j)
        {
            temp[i][j] += bias[0][j];
        }
    }

    return temp;
}

std::vector<std::vector<double>> Layer::activation(const std::vector<std::vector<double>> &x,
                                                   std::function<double(double)> activation_fn)
{
    int x_rows = x.size();
    int x_cols = x[0].size();

    std::vector<std::vector<double>> activated_x = x;

    for (int i = 0; i < x_rows; ++i)
    {
        for (int j = 0; j < x_cols; ++j)
        {
            activated_x[i][j] = activation_fn(activated_x[i][j]);
        }
    }

    return activated_x;
}

// NeuralNetwork implementation
NeuralNetwork::NeuralNetwork() : use_cross_entropy(false)
{
}

void NeuralNetwork::add_layer(int input_size, int output_size, std::function<double(double)> act_fn,
                              std::function<double(double)> act_deriv, bool is_softmax, InitializationType init_type,
                              double dropout)
{
    layers.push_back(Layer(input_size, output_size, act_fn, act_deriv, is_softmax, init_type, dropout));
}

void NeuralNetwork::set_training_mode(bool training)
{
    for (size_t i = 0; i < layers.size(); ++i)
    {
        layers[i].training_mode = training;
    }
}

std::vector<std::vector<double>> NeuralNetwork::forward(const std::vector<std::vector<double>> &input)
{
    std::vector<std::vector<double>> output = input;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        output = layers[i].forward(output);
    }
    return output;
}

double NeuralNetwork::MSE(const std::vector<std::vector<double>> &y_pred, const std::vector<std::vector<double>> &y_act)
{
    double total_error = 0.0;
    int rows = y_pred.size();
    int cols = y_pred[0].size();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double error = y_pred[i][j] - y_act[i][j];
            total_error += error * error;
        }
    }
    return total_error / (rows * cols);
}

double NeuralNetwork::cross_entropy(const std::vector<std::vector<double>> &y_pred,
                                    const std::vector<std::vector<double>> &y_act)
{
    double total_loss = 0.0;
    int rows = y_pred.size();
    int cols = y_pred[0].size();
    const double epsilon = 1e-7; // Small value to prevent log(0)

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Clip predictions to prevent numerical instability
            double pred = std::max(epsilon, std::min(1.0 - epsilon, y_pred[i][j]));
            total_loss -= y_act[i][j] * std::log(pred);
        }
    }
    return total_loss / rows;
}

void NeuralNetwork::train_batch(const std::vector<std::vector<double>> &all_inputs,
                                const std::vector<std::vector<double>> &all_targets, int epochs, double learning_rate,
                                double tolerance, int batch_size, bool use_cross_entropy, double decay_rate,
                                OptimizerType optimizer)
{
    int num_examples = all_inputs.size();
    double current_lr = learning_rate; // Track the current learning rate

    // Enable training mode for dropout
    set_training_mode(true);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;

        // Shuffle the data to ensure randomness
        std::vector<int> indices(num_examples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

        // Process each batch
        for (int batch_start = 0; batch_start < num_examples; batch_start += batch_size)
        {
            int batch_end = std::min(batch_start + batch_size, num_examples);
            int actual_batch_size = batch_end - batch_start;

            // Reset accumulated gradients for the batch
            for (Layer &layer : layers)
            {
                for (size_t i = 0; i < layer.weight_gradients.size(); ++i)
                {
                    for (size_t j = 0; j < layer.weight_gradients[i].size(); ++j)
                    {
                        layer.weight_gradients[i][j] = 0.0;
                    }
                }
                for (size_t j = 0; j < layer.bias_gradients[0].size(); ++j)
                {
                    layer.bias_gradients[0][j] = 0.0;
                }
                layer.batch_count = 0;
            }

            // Process each example in the batch
            for (int example = batch_start; example < batch_end; ++example)
            {
                std::vector<std::vector<double>> input = {all_inputs[indices[example]]};
                std::vector<std::vector<double>> target = {all_targets[indices[example]]};

                // Forward pass
                std::vector<std::vector<double>> output = forward(input);

                // Accumulate loss
                if (use_cross_entropy)
                {
                    total_loss += cross_entropy(output, target);
                }
                else
                {
                    total_loss += MSE(output, target);
                }

                // Backward pass - Output layer
                int last_layer_idx = layers.size() - 1;
                layers[last_layer_idx].backward_output_layer(target, use_cross_entropy);

                // Backward pass - Hidden layers
                for (int i = last_layer_idx - 1; i >= 0; --i)
                {
                    layers[i].backward_hidden_layer(layers[i + 1].weights, layers[i + 1].gradients);
                }

                // Accumulate gradients for the batch
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    // Accumulate weight gradients
                    for (size_t row = 0; row < layers[i].weights.size(); ++row)
                    {
                        for (size_t col = 0; col < layers[i].weights[row].size(); ++col)
                        {
                            layers[i].weight_gradients[row][col] +=
                                layers[i].inputs[0][row] * layers[i].gradients[0][col] / actual_batch_size;
                        }
                    }

                    // Accumulate bias gradients
                    for (size_t col = 0; col < layers[i].biases[0].size(); ++col)
                    {
                        layers[i].bias_gradients[0][col] += layers[i].gradients[0][col] / actual_batch_size;
                    }

                    layers[i].batch_count++;
                }
            }

            // Update parameters after the batch
            for (size_t i = 0; i < layers.size(); ++i)
            {
                layers[i].update_parameters(current_lr, optimizer);
            }
        }

        // Average loss
        double avg_loss = total_loss / num_examples;

        if (epoch % 100 == 0)
        {
            std::cout << "Epoch " << epoch << " - Average Loss: " << avg_loss << std::endl;
        }
        if (avg_loss < tolerance)
        {
            std::cout << "Training stopped early at epoch " << epoch << " due to low loss: " << avg_loss << std::endl;
            break;
        }

        // Apply learning rate decay
        current_lr *= decay_rate;
    }

    // Disable training mode after training (dropout will be disabled)
    set_training_mode(false);
}

// Save network weights to file
bool NeuralNetwork::save_weights(const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not create weights file: " << filename << std::endl;
        return false;
    }
    
    // Save number of layers
    size_t num_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Save each layer's weights and biases
    for (const auto &layer : layers)
    {
        // Save weight dimensions
        size_t rows = layer.weights.size();
        size_t cols = (rows > 0) ? layer.weights[0].size() : 0;
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        
        // Save weights
        for (const auto &row : layer.weights)
        {
            for (double val : row)
            {
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
        
        // Save bias dimensions
        size_t bias_rows = layer.biases.size();
        size_t bias_cols = (bias_rows > 0) ? layer.biases[0].size() : 0;
        file.write(reinterpret_cast<const char*>(&bias_rows), sizeof(bias_rows));
        file.write(reinterpret_cast<const char*>(&bias_cols), sizeof(bias_cols));
        
        // Save biases
        for (const auto &row : layer.biases)
        {
            for (double val : row)
            {
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
    }
    
    file.close();
    return true;
}

// Load network weights from file
bool NeuralNetwork::load_weights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open weights file: " << filename << std::endl;
        return false;
    }
    
    // Load number of layers
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    if (num_layers != layers.size())
    {
        std::cerr << "Error: Network architecture mismatch. Expected " << layers.size() 
                  << " layers but file has " << num_layers << std::endl;
        return false;
    }
    
    // Load each layer's weights and biases
    for (auto &layer : layers)
    {
        // Load weight dimensions
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        // Verify dimensions match
        if (rows != layer.weights.size() || 
            (rows > 0 && cols != layer.weights[0].size()))
        {
            std::cerr << "Error: Weight dimensions mismatch" << std::endl;
            return false;
        }
        
        // Load weights
        for (auto &row : layer.weights)
        {
            for (double &val : row)
            {
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
            }
        }
        
        // Load bias dimensions
        size_t bias_rows, bias_cols;
        file.read(reinterpret_cast<char*>(&bias_rows), sizeof(bias_rows));
        file.read(reinterpret_cast<char*>(&bias_cols), sizeof(bias_cols));
        
        // Verify dimensions match
        if (bias_rows != layer.biases.size() || 
            (bias_rows > 0 && bias_cols != layer.biases[0].size()))
        {
            std::cerr << "Error: Bias dimensions mismatch" << std::endl;
            return false;
        }
        
        // Load biases
        for (auto &row : layer.biases)
        {
            for (double &val : row)
            {
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
            }
        }
    }
    
    file.close();
    return true;
}
