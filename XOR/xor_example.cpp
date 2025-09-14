#include "../Neural Network Framework/neural_network.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

bool load_xor_dataset(const string &filename, vector<vector<double>> &inputs, vector<vector<double>> &outputs)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open dataset file: " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        istringstream iss(line);
        double in1, in2, out;
        if (iss >> in1 >> in2 >> out)
        {
            inputs.push_back({in1, in2});
            outputs.push_back({out});
        }
    }

    file.close();
    return true;
}

int main()
{
    try
    {
        // Load dataset from file
        vector<vector<double>> xor_inputs;
        vector<vector<double>> xor_outputs;

        if (!load_xor_dataset("xor_dataset.txt", xor_inputs, xor_outputs))
        {
            cerr << "Failed to load dataset. Using default hardcoded values." << endl;
            xor_inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
            xor_outputs = {{0.0}, {1.0}, {1.0}, {0.0}};
        }
        else
        {
            cout << "Successfully loaded " << xor_inputs.size() << " samples from dataset file." << endl;
        }

        cout << "=== XOR Problem Solution ===" << endl;
        cout << "\nConfiguration:" << endl;
        cout << "- Architecture: 2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid)" << endl;
        cout << "- Xavier initialization for optimal gradient flow" << endl;
        cout << "- Adam optimizer for faster convergence\n" << endl;

        // Create neural network with optimal XOR configuration
        NeuralNetwork nn;

        // use_softmax=false (standard activation)
        nn.add_layer(2, 4, tanh_activation, tanh_derivative, false, InitializationType::XAVIER);
        nn.add_layer(4, 1, sigmoid, sigmoid_derivative, false, InitializationType::XAVIER);

        // Train the network
        cout << "Training network with Adam optimizer..." << endl;

        // Using Adam optimizer for faster convergence
        // epochs=1000, learning_rate=0.1, tolerance=0.01, batch_size=4, no_cross_entropy, no_decay, Adam
        nn.train_batch(xor_inputs, xor_outputs, 1000, 0.1, 0.01, 4, false, 1.0, OptimizerType::ADAM);

        // Test the trained network
        cout << "\n=== Final Results ===" << endl;
        cout << "\n+------------+------------+----------+------------+--------+" << endl;
        cout << "|  Input 1   |  Input 2   | Expected | Predicted  | Result |" << endl;
        cout << "+------------+------------+----------+------------+--------+" << endl;

        double total_error = 0.0;
        int correct_count = 0;
        int display_count = 0;
        const int MAX_DISPLAY = 20; // Show first 20 results in table

        for (size_t i = 0; i < xor_inputs.size(); ++i)
        {
            vector<vector<double>> single_input = {xor_inputs[i]};
            vector<vector<double>> single_output = {xor_outputs[i]};
            vector<vector<double>> prediction = nn.forward(single_input);

            // Check if prediction is correct (threshold at 0.5)
            bool correct = (prediction[0][0] > 0.5 && xor_outputs[i][0] > 0.5) ||
                           (prediction[0][0] <= 0.5 && xor_outputs[i][0] <= 0.5);
            if (correct)
                correct_count++;

            // Display first 20 results in table format
            if (display_count < MAX_DISPLAY)
            {
                printf("| %10.2f | %10.2f | %8.1f | %10.6f | %6s |\n", xor_inputs[i][0], xor_inputs[i][1],
                       xor_outputs[i][0], prediction[0][0], correct ? "✓" : "✗");
                display_count++;
            }

            total_error += nn.MSE(prediction, single_output);
        }

        cout << "+------------+------------+----------+------------+--------+" << endl;
        cout << "\nShowing first " << MAX_DISPLAY << " of " << xor_inputs.size() << " samples" << endl;
        cout << "\n=== Summary Statistics ===" << endl;
        cout << "Total Samples:    " << xor_inputs.size() << endl;
        cout << "Correct:          " << correct_count << " / " << xor_inputs.size() << " ("
             << (100.0 * correct_count / xor_inputs.size()) << "%)" << endl;
        cout << "Final MSE:        " << total_error / xor_inputs.size() << endl;
    }
    catch (invalid_argument &e)
    {
        cerr << "\nError: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
