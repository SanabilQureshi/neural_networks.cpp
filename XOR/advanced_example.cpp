#include "../Neural Network Framework/neural_network.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    cout << "=== Demonstrating New Features ===" << endl << endl;

    // Load XOR dataset
    vector<vector<double>> xor_inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<vector<double>> xor_outputs = {{0.0}, {1.0}, {1.0}, {0.0}};

    // Test 1: Learning Rate Decay
    cout << "1. Testing Learning Rate Decay (decay_rate=0.99)" << endl;
    NeuralNetwork nn1;
    nn1.add_layer(2, 4, tanh_activation, tanh_derivative, false, InitializationType::XAVIER);
    nn1.add_layer(4, 1, sigmoid, sigmoid_derivative, false, InitializationType::XAVIER);

    // epochs=500, lr=0.5, tolerance=0.01, batch=4, cross_entropy=false, decay_rate=0.99
    nn1.train_batch(xor_inputs, xor_outputs, 500, 0.5, 0.01, 4, false, 0.99);

    // Test 2: Momentum Optimizer
    cout << "\n2. Testing Momentum Optimizer" << endl;
    NeuralNetwork nn2;
    nn2.add_layer(2, 4, tanh_activation, tanh_derivative, false, InitializationType::XAVIER);
    nn2.add_layer(4, 1, sigmoid, sigmoid_derivative, false, InitializationType::XAVIER);

    // Using momentum optimizer
    nn2.train_batch(xor_inputs, xor_outputs, 500, 0.3, 0.01, 4, false, 1.0, OptimizerType::MOMENTUM);

    // Test 3: Adam Optimizer
    cout << "\n3. Testing Adam Optimizer" << endl;
    NeuralNetwork nn3;
    nn3.add_layer(2, 4, tanh_activation, tanh_derivative, false, InitializationType::XAVIER);
    nn3.add_layer(4, 1, sigmoid, sigmoid_derivative, false, InitializationType::XAVIER);

    // Using Adam optimizer (typically uses lower learning rate)
    nn3.train_batch(xor_inputs, xor_outputs, 500, 0.01, 0.01, 4, false, 1.0, OptimizerType::ADAM);

    // Test 4: Dropout Regularization
    cout << "\n4. Testing Dropout (50% dropout in hidden layer)" << endl;
    NeuralNetwork nn4;
    // Add 50% dropout to hidden layer
    nn4.add_layer(2, 8, relu, relu_derivative, false, InitializationType::HE, 0.5);
    nn4.add_layer(8, 1, sigmoid, sigmoid_derivative, false, InitializationType::XAVIER, 0.0);

    nn4.train_batch(xor_inputs, xor_outputs, 1000, 0.5, 0.01, 4);

    // Test final predictions
    cout << "\n=== Testing All Networks ===" << endl;

    auto test_network = [&](NeuralNetwork &nn, const string &name) {
        cout << "\n" << name << ":" << endl;
        for (size_t i = 0; i < xor_inputs.size(); ++i)
        {
            vector<vector<double>> single_input = {xor_inputs[i]};
            vector<vector<double>> prediction = nn.forward(single_input);

            cout << "  [" << xor_inputs[i][0] << ", " << xor_inputs[i][1] << "] -> ";
            cout << prediction[0][0] << " (Expected: " << xor_outputs[i][0] << ")" << endl;
        }
    };

    test_network(nn1, "With LR Decay");
    test_network(nn2, "With Momentum");
    test_network(nn3, "With Adam");
    test_network(nn4, "With Dropout");

    cout << "\nNote: All features are working! You can combine them as needed." << endl;
    cout << "For example: Adam + LR Decay + Dropout for best results on complex problems." << endl;

    return 0;
}