#include "../Neural Network Framework/neural_network.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>

using namespace std;

// MNISTLoader class for handling MNIST dataset
class MNISTLoader
{
  public:
    vector<vector<double>> train_images;
    vector<vector<double>> train_labels;
    vector<vector<double>> test_images;
    vector<vector<double>> test_labels;

  private:
    // Convert big-endian to little-endian for 32-bit integers
    uint32_t reverse_int(uint32_t i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
    }

    // Load images from IDX3 format
    bool load_images(const string &filename, vector<vector<double>> &images)
    {
        ifstream file(filename, ios::binary);
        if (!file.is_open())
        {
            cerr << "Error: Cannot open file " << filename << endl;
            return false;
        }

        uint32_t magic_number = 0;
        uint32_t num_images = 0;
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if (magic_number != 2051)
        {
            cerr << "Error: Invalid magic number in " << filename << endl;
            return false;
        }

        file.read((char *)&num_images, sizeof(num_images));
        file.read((char *)&num_rows, sizeof(num_rows));
        file.read((char *)&num_cols, sizeof(num_cols));

        num_images = reverse_int(num_images);
        num_rows = reverse_int(num_rows);
        num_cols = reverse_int(num_cols);

        cout << "Loading " << num_images << " images of size " << num_rows << "x" << num_cols << endl;

        images.resize(num_images);
        for (uint32_t i = 0; i < num_images; ++i)
        {
            images[i].resize(num_rows * num_cols);
            for (uint32_t j = 0; j < num_rows * num_cols; ++j)
            {
                unsigned char pixel = 0;
                file.read((char *)&pixel, sizeof(pixel));
                // Normalize pixel values to [0, 1]
                images[i][j] = static_cast<double>(pixel) / 255.0;
            }
        }

        file.close();
        return true;
    }

    // Load labels from IDX1 format
    bool load_labels(const string &filename, vector<vector<double>> &labels)
    {
        ifstream file(filename, ios::binary);
        if (!file.is_open())
        {
            cerr << "Error: Cannot open file " << filename << endl;
            return false;
        }

        uint32_t magic_number = 0;
        uint32_t num_labels = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if (magic_number != 2049)
        {
            cerr << "Error: Invalid magic number in " << filename << endl;
            return false;
        }

        file.read((char *)&num_labels, sizeof(num_labels));
        num_labels = reverse_int(num_labels);

        cout << "Loading " << num_labels << " labels" << endl;

        labels.resize(num_labels);
        for (uint32_t i = 0; i < num_labels; ++i)
        {
            unsigned char label = 0;
            file.read((char *)&label, sizeof(label));

            // Convert to one-hot encoding
            labels[i].resize(10, 0.0);
            labels[i][label] = 1.0;
        }

        file.close();
        return true;
    }

  public:
    // Load complete MNIST dataset
    bool load_mnist_dataset()
    {
        cout << "=== Loading MNIST Dataset ===" << endl;

        // Load training data
        cout << "\nLoading training data..." << endl;
        if (!load_images("../archive/train-images.idx3-ubyte", train_images))
        {
            return false;
        }
        if (!load_labels("../archive/train-labels.idx1-ubyte", train_labels))
        {
            return false;
        }

        // Load test data
        cout << "\nLoading test data..." << endl;
        if (!load_images("../archive/t10k-images.idx3-ubyte", test_images))
        {
            return false;
        }
        if (!load_labels("../archive/t10k-labels.idx1-ubyte", test_labels))
        {
            return false;
        }

        cout << "\nDataset loaded successfully!" << endl;
        cout << "Training samples: " << train_images.size() << endl;
        cout << "Test samples: " << test_images.size() << endl;
        cout << "Image dimensions: " << train_images[0].size() << " pixels (28x28)" << endl;
        cout << "Number of classes: " << train_labels[0].size() << " (digits 0-9)" << endl;

        return true;
    }
    
    // Load multiple images from IDX3 format (for testing user drawings)
    bool load_user_images(const string &filename, vector<vector<double>> &images)
    {
        ifstream file(filename, ios::binary);
        if (!file.is_open())
        {
            cerr << "Error: Cannot open file " << filename << endl;
            return false;
        }

        uint32_t magic_number = 0;
        uint32_t num_images = 0;
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if (magic_number != 2051)
        {
            cerr << "Error: Invalid magic number. Expected 2051, got " << magic_number << endl;
            return false;
        }

        file.read((char *)&num_images, sizeof(num_images));
        file.read((char *)&num_rows, sizeof(num_rows));
        file.read((char *)&num_cols, sizeof(num_cols));

        num_images = reverse_int(num_images);
        num_rows = reverse_int(num_rows);
        num_cols = reverse_int(num_cols);

        if (num_images < 1)
        {
            cerr << "Error: File contains no images" << endl;
            return false;
        }

        cout << "Loading " << num_images << " image(s) (" << num_rows << "x" << num_cols << ")" << endl;

        images.resize(num_images);
        for (uint32_t i = 0; i < num_images; ++i)
        {
            images[i].resize(num_rows * num_cols);
            for (uint32_t j = 0; j < num_rows * num_cols; ++j)
            {
                unsigned char pixel = 0;
                file.read((char *)&pixel, sizeof(pixel));
                // Normalize pixel values to [0, 1]
                images[i][j] = static_cast<double>(pixel) / 255.0;
            }
        }

        file.close();
        return true;
    }
};

// Helper function to get the predicted class from network output
int get_predicted_class(const vector<double> &output)
{
    return max_element(output.begin(), output.end()) - output.begin();
}

// Helper function to get the actual class from one-hot encoded label
int get_actual_class(const vector<double> &label)
{
    return max_element(label.begin(), label.end()) - label.begin();
}

// Display a digit as ASCII art (simplified version)
void display_digit(const vector<double> &image, int actual_class = -1, int predicted_class = -1)
{
    cout << "\n+---------------------------+" << endl;
    if (actual_class >= 0 && predicted_class >= 0)
    {
        cout << "| Actual: " << actual_class << " | Predicted: " << predicted_class;
        if (actual_class == predicted_class)
            cout << " ✓ |" << endl;
        else
            cout << " ✗ |" << endl;
    }
    cout << "+---------------------------+" << endl;
    
    for (int i = 0; i < 28; ++i)
    {
        cout << "|";
        for (int j = 0; j < 28; ++j)
        {
            double pixel = image[i * 28 + j];
            if (pixel > 0.7)
                cout << "█";
            else if (pixel > 0.4)
                cout << "▓";
            else if (pixel > 0.2)
                cout << "░";
            else
                cout << " ";
        }
        cout << "|" << endl;
    }
    cout << "+---------------------------+" << endl;
}

// Create and configure the neural network architecture
NeuralNetwork create_network()
{
    NeuralNetwork nn;
    
    // Architecture: 784 (28x28 pixels) -> 128 -> 64 -> 10 (digit classes)
    // Using ReLU for hidden layers and softmax for output
    nn.add_layer(784, 128, relu, relu_derivative, false, InitializationType::HE, 0.2);     // Input to hidden 1 with dropout
    nn.add_layer(128, 64, relu, relu_derivative, false, InitializationType::HE, 0.2);      // Hidden 1 to hidden 2 with dropout
    nn.add_layer(64, 10, sigmoid, sigmoid_derivative, true, InitializationType::XAVIER);   // Hidden 2 to output (softmax)
    
    return nn;
}

// Training mode
void train_mode()
{
    cout << "\n=== MNIST Training Mode ===" << endl;
    
    // Load MNIST dataset
    MNISTLoader mnist;
    if (!mnist.load_mnist_dataset())
    {
        cerr << "Failed to load MNIST dataset. Please ensure the dataset files are in the archive/ directory." << endl;
        return;
    }
    
    cout << "\n=== Neural Network Configuration ===" << endl;
    cout << "Architecture: 784 inputs -> 128 hidden (ReLU) -> 64 hidden (ReLU) -> 10 outputs (Softmax)" << endl;
    cout << "- He initialization for ReLU layers, Xavier for output layer" << endl;
    cout << "- Cross-entropy loss with softmax activation" << endl;
    cout << "- Adam optimizer with learning rate decay" << endl;
    cout << "- 20% dropout on hidden layers during training" << endl;
    
    // Create neural network
    NeuralNetwork nn = create_network();
    
    // Use subset of training data for faster training in this demo
    int train_samples = min(10000, (int)mnist.train_images.size());
    
    vector<vector<double>> train_subset(mnist.train_images.begin(), mnist.train_images.begin() + train_samples);
    vector<vector<double>> train_labels_subset(mnist.train_labels.begin(), mnist.train_labels.begin() + train_samples);
    
    cout << "\nUsing " << train_samples << " training samples for demo" << endl;
    
    // Train the network
    cout << "\nTraining network..." << endl;
    cout << "Progress will be displayed every 100 epochs." << endl;
    
    // Training parameters: epochs=1000, learning_rate=0.001, tolerance=0.01, batch_size=32, 
    // cross_entropy=true, decay_rate=0.99, Adam optimizer
    nn.train_batch(train_subset, train_labels_subset, 1000, 0.001, 0.01, 32, true, 0.99, OptimizerType::ADAM);
    
    // Create trained_model directory if it doesn't exist
    struct stat st = {0};
    if (stat("trained_model", &st) == -1) {
        #ifdef _WIN32
            mkdir("trained_model");
        #else
            mkdir("trained_model", 0700);
        #endif
    }
    
    // Save the trained model
    string model_path = "trained_model/mnist_model.weights";
    cout << "\nSaving trained model to " << model_path << "..." << endl;
    if (nn.save_weights(model_path))
    {
        cout << "Model saved successfully!" << endl;
    }
    else
    {
        cerr << "Failed to save model!" << endl;
    }
    
    // Quick test on test set
    cout << "\n=== Quick Evaluation on Test Set ===" << endl;
    nn.set_training_mode(false);  // Disable dropout for testing
    
    int test_samples = min(1000, (int)mnist.test_images.size());
    int correct_predictions = 0;
    
    for (int i = 0; i < test_samples; ++i)
    {
        vector<vector<double>> single_image = {mnist.test_images[i]};
        vector<vector<double>> prediction = nn.forward(single_image);
        
        int predicted_class = get_predicted_class(prediction[0]);
        int actual_class = get_actual_class(mnist.test_labels[i]);
        
        if (predicted_class == actual_class)
        {
            correct_predictions++;
        }
    }
    
    double accuracy = (double)correct_predictions / test_samples * 100.0;
    cout << "Test accuracy: " << correct_predictions << "/" << test_samples 
         << " (" << accuracy << "%)" << endl;
         
    cout << "\nTraining complete! You can now test the model with:" << endl;
    cout << "  ./mnist_example --test <image_file.dat>" << endl;
}

// Testing mode
void test_mode(const string &test_file)
{
    cout << "\n=== MNIST Testing Mode ===" << endl;
    cout << "Testing file: " << test_file << endl;
    
    // Create neural network with same architecture
    NeuralNetwork nn = create_network();
    
    // Load trained weights
    string model_path = "trained_model/mnist_model.weights";
    cout << "\nLoading trained model from " << model_path << "..." << endl;
    
    if (!nn.load_weights(model_path))
    {
        cerr << "Failed to load model weights!" << endl;
        cerr << "Please train the model first with: ./mnist_example --train" << endl;
        return;
    }
    
    cout << "Model loaded successfully!" << endl;
    nn.set_training_mode(false);  // Disable dropout for testing
    
    // Load the test images
    MNISTLoader loader;
    vector<vector<double>> images;
    
    if (!loader.load_user_images(test_file, images))
    {
        cerr << "Failed to load test images!" << endl;
        return;
    }
    
    cout << "\n=== Processing " << images.size() << " Image(s) ===" << endl;
    
    // Process each image
    vector<int> predictions;
    vector<double> confidences;
    
    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx)
    {
        // Make prediction
        vector<vector<double>> input = {images[img_idx]};
        vector<vector<double>> output = nn.forward(input);
        
        // Get prediction and confidence
        int predicted_class = get_predicted_class(output[0]);
        double confidence = output[0][predicted_class] * 100.0;
        
        predictions.push_back(predicted_class);
        confidences.push_back(confidence);
        
        // For the first image only, show ASCII art and detailed scores
        if (img_idx == 0)
        {
            cout << "\n--- Image 1 (Detailed View) ---" << endl;
            display_digit(images[0]);
            
            cout << "\nPredicted digit: " << predicted_class << endl;
            cout << "\nConfidence scores:" << endl;
            cout << "+-------+------------+" << endl;
            cout << "| Digit | Confidence |" << endl;
            cout << "+-------+------------+" << endl;
            
            // Create sorted list of predictions
            vector<pair<double, int>> scores;
            for (int i = 0; i < 10; ++i)
            {
                scores.push_back({output[0][i], i});
            }
            sort(scores.begin(), scores.end(), greater<pair<double, int>>());
            
            // Display sorted predictions
            for (const auto& score : scores)
            {
                printf("| %5d | %9.2f%% |", score.second, score.first * 100);
                if (score.second == predicted_class)
                    cout << " <-- PREDICTED";
                cout << endl;
            }
            cout << "+-------+------------+" << endl;
        }
    }
    
    // If more than one image, show summary table
    if (images.size() > 1)
    {
        cout << "\n=== Summary of All Predictions ===" << endl;
        cout << "+-------+------------+------------+" << endl;
        cout << "| Image | Prediction | Confidence |" << endl;
        cout << "+-------+------------+------------+" << endl;
        
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            printf("| %5zu |     %d      | %9.2f%% |\n", 
                   i + 1, predictions[i], confidences[i]);
        }
        cout << "+-------+------------+------------+" << endl;
        
        // Show digit distribution if multiple images
        cout << "\n=== Digit Distribution ===" << endl;
        vector<int> digit_counts(10, 0);
        for (int pred : predictions)
        {
            digit_counts[pred]++;
        }
        
        cout << "+-------+-------+" << endl;
        cout << "| Digit | Count |" << endl;
        cout << "+-------+-------+" << endl;
        for (int i = 0; i < 10; ++i)
        {
            if (digit_counts[i] > 0)
            {
                printf("| %5d | %5d |\n", i, digit_counts[i]);
            }
        }
        cout << "+-------+-------+" << endl;
    }
    
    cout << "\nProcessed " << images.size() << " image(s) successfully!" << endl;
    cout << "To test more images, run:" << endl;
    cout << "  ./mnist_example --test <image_file.dat>" << endl;
}

// Print usage information
void print_usage(const char* program_name)
{
    cout << "Usage: " << program_name << " [--train | --test <file>]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  --train       Train the neural network on MNIST dataset" << endl;
    cout << "                Saves weights to trained_model/mnist_model.weights" << endl;
    cout << "  --test <file> Test the trained model on a single image file" << endl;
    cout << "                File should be in MNIST IDX format (from interactive demo)" << endl;
    cout << "\nExamples:" << endl;
    cout << "  " << program_name << " --train" << endl;
    cout << "  " << program_name << " --test user_drawing.dat" << endl;
    cout << "\nInteractive Demo:" << endl;
    cout << "  1. Open interactive_demo/digit_draw.html in a browser" << endl;
    cout << "  2. Draw a digit and save as .dat file" << endl;
    cout << "  3. Test with: " << program_name << " --test <saved_file.dat>" << endl;
}

int main(int argc, char* argv[])
{
    try
    {
        // Parse command-line arguments
        if (argc < 2)
        {
            print_usage(argv[0]);
            return 1;
        }
        
        string mode = argv[1];
        
        if (mode == "--train")
        {
            train_mode();
        }
        else if (mode == "--test")
        {
            if (argc < 3)
            {
                cerr << "Error: --test requires a file argument" << endl;
                print_usage(argv[0]);
                return 1;
            }
            test_mode(argv[2]);
        }
        else
        {
            cerr << "Error: Unknown option '" << mode << "'" << endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    catch (const std::invalid_argument &e)
    {
        cerr << "\nError: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
