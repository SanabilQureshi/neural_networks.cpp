#include "../Neural Network Framework/neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;

class SuperstoreClassifier
{
  public:
    vector<vector<double>> features;
    vector<vector<double>> targets;
    
    vector<string> parse_csv_line(const string& line) 
    {
        vector<string> result;
        string current;
        bool in_quotes = false;
        
        for (char c : line) {
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == ',' && !in_quotes) {
                result.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        result.push_back(current);
        return result;
    }
    
    bool load_for_classification(const string& filename)
    {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Cannot open " << filename << endl;
            return false;
        }
        
        string line;
        getline(file, line); // Skip header
        
        vector<vector<double>> raw_features;
        vector<int> raw_targets;
        
        while (getline(file, line)) {
            vector<string> fields = parse_csv_line(line);
            if (fields.size() < 21) continue;
            
            try {
                double sales = stod(fields[17]);
                double quantity = stod(fields[18]);
                double discount = stod(fields[19]);
                double profit = stod(fields[20]);
                
                // Skip outliers
                if (abs(profit) > 10000 || abs(sales) > 50000) continue;
                
                // Features for classification
                vector<double> row;
                row.push_back(sales);
                row.push_back(quantity);
                row.push_back(discount);
                row.push_back(sales * quantity); // total volume
                row.push_back(sales - profit); // approximate cost
                
                // Classification target: Is this transaction profitable?
                // 0: Loss (profit < 0)
                // 1: Low profit (0 <= profit < 50)
                // 2: Medium profit (50 <= profit < 200)
                // 3: High profit (profit >= 200)
                int category;
                if (profit < 0) category = 0;
                else if (profit < 50) category = 1;
                else if (profit < 200) category = 2;
                else category = 3;
                
                raw_features.push_back(row);
                raw_targets.push_back(category);
                
            } catch (...) {
                continue;
            }
        }
        file.close();
        
        if (raw_features.empty()) {
            cerr << "No valid data loaded" << endl;
            return false;
        }
        
        cout << "Loaded " << raw_features.size() << " samples" << endl;
        
        // Count class distribution
        vector<int> class_counts(4, 0);
        for (int cat : raw_targets) {
            class_counts[cat]++;
        }
        
        cout << "\nClass distribution:" << endl;
        cout << "  Loss (profit < 0):        " << class_counts[0] << " samples" << endl;
        cout << "  Low profit ($0-50):       " << class_counts[1] << " samples" << endl;
        cout << "  Medium profit ($50-200):  " << class_counts[2] << " samples" << endl;
        cout << "  High profit (>$200):      " << class_counts[3] << " samples" << endl;
        
        // Normalize features
        int num_features = raw_features[0].size();
        vector<double> means(num_features, 0);
        vector<double> stds(num_features, 0);
        
        // Calculate means
        for (const auto& row : raw_features) {
            for (int i = 0; i < num_features; i++) {
                means[i] += row[i];
            }
        }
        for (int i = 0; i < num_features; i++) {
            means[i] /= raw_features.size();
        }
        
        // Calculate stds
        for (const auto& row : raw_features) {
            for (int i = 0; i < num_features; i++) {
                double diff = row[i] - means[i];
                stds[i] += diff * diff;
            }
        }
        for (int i = 0; i < num_features; i++) {
            stds[i] = sqrt(stds[i] / raw_features.size());
            if (stds[i] < 1e-6) stds[i] = 1.0;
        }
        
        // Store normalized data with one-hot encoded targets
        features.resize(raw_features.size(), vector<double>(num_features));
        targets.resize(raw_features.size(), vector<double>(4, 0.0));
        
        for (size_t i = 0; i < raw_features.size(); i++) {
            // Normalize features
            for (int j = 0; j < num_features; j++) {
                features[i][j] = (raw_features[i][j] - means[j]) / stds[j];
            }
            // One-hot encode target
            targets[i][raw_targets[i]] = 1.0;
        }
        
        return true;
    }
};

int main()
{
    try {
        cout << "=== Superstore Profit Classification ===" << endl;
        cout << "Classifying transactions into profit categories using your Neural Network\n" << endl;
        
        SuperstoreClassifier classifier;
        if (!classifier.load_for_classification("Sample - Superstore.csv")) {
            return 1;
        }
        
        // Split data 80/20
        int total = classifier.features.size();
        int train_size = total * 0.8;
        
        // Shuffle indices
        vector<int> indices(total);
        for (int i = 0; i < total; i++) indices[i] = i;
        random_device rd;
        mt19937 gen(rd());
        shuffle(indices.begin(), indices.end(), gen);
        
        vector<vector<double>> X_train(train_size), y_train(train_size);
        vector<vector<double>> X_test(total - train_size), y_test(total - train_size);
        
        for (int i = 0; i < train_size; i++) {
            X_train[i] = classifier.features[indices[i]];
            y_train[i] = classifier.targets[indices[i]];
        }
        for (int i = train_size; i < total; i++) {
            X_test[i - train_size] = classifier.features[indices[i]];
            y_test[i - train_size] = classifier.targets[indices[i]];
        }
        
        cout << "\nTraining: " << train_size << " samples, Testing: " << (total - train_size) << " samples" << endl;
        
        // Create network for classification
        NeuralNetwork nn;
        nn.add_layer(5, 32, relu, relu_derivative, false, InitializationType::HE);
        nn.add_layer(32, 16, relu, relu_derivative, false, InitializationType::HE);
        nn.add_layer(16, 4, sigmoid, sigmoid_derivative, true, InitializationType::XAVIER); // softmax output
        
        cout << "\nNetwork: 5 inputs -> 32 (ReLU) -> 16 (ReLU) -> 4 outputs (Softmax)" << endl;
        cout << "Training with cross-entropy loss...\n" << endl;
        
        nn.train_batch(X_train, y_train, 300, 0.01, 0.001, 32, true, 0.995, OptimizerType::ADAM);
        
        // Evaluate
        cout << "\n=== Test Set Performance ===" << endl;
        
        int correct = 0;
        vector<vector<int>> confusion(4, vector<int>(4, 0));
        
        for (size_t i = 0; i < X_test.size(); i++) {
            auto pred = nn.forward({X_test[i]});
            
            // Get predicted and actual classes
            int pred_class = 0, actual_class = 0;
            double max_prob = pred[0][0];
            for (int j = 1; j < 4; j++) {
                if (pred[0][j] > max_prob) {
                    max_prob = pred[0][j];
                    pred_class = j;
                }
            }
            for (int j = 0; j < 4; j++) {
                if (y_test[i][j] > 0.5) {
                    actual_class = j;
                    break;
                }
            }
            
            confusion[actual_class][pred_class]++;
            if (pred_class == actual_class) correct++;
        }
        
        double accuracy = 100.0 * correct / X_test.size();
        cout << "\nOverall Accuracy: " << fixed << setprecision(1) << accuracy << "%" << endl;
        
        // Display confusion matrix
        cout << "\nConfusion Matrix:" << endl;
        cout << "              Predicted" << endl;
        cout << "         Loss  Low  Med  High" << endl;
        vector<string> labels = {"Loss  ", "Low   ", "Medium", "High  "};
        for (int i = 0; i < 4; i++) {
            cout << labels[i] << " ";
            for (int j = 0; j < 4; j++) {
                cout << setw(5) << confusion[i][j] << " ";
            }
            cout << endl;
        }
        
        // Per-class metrics
        cout << "\nPer-Class Performance:" << endl;
        cout << "Class        Precision  Recall   F1-Score  Support" << endl;
        cout << "------------------------------------------------" << endl;
        
        for (int i = 0; i < 4; i++) {
            int true_pos = confusion[i][i];
            int false_pos = 0, false_neg = 0;
            int support = 0;
            
            for (int j = 0; j < 4; j++) {
                if (i != j) {
                    false_pos += confusion[j][i];
                    false_neg += confusion[i][j];
                }
                support += confusion[i][j];
            }
            
            double precision = (true_pos + false_pos > 0) ? 
                100.0 * true_pos / (true_pos + false_pos) : 0;
            double recall = (support > 0) ? 
                100.0 * true_pos / support : 0;
            double f1 = (precision + recall > 0) ? 
                2 * precision * recall / (precision + recall) : 0;
            
            cout << labels[i] << "      " 
                 << setw(6) << fixed << setprecision(1) << precision << "%   "
                 << setw(6) << recall << "%   "
                 << setw(6) << f1 << "%     "
                 << setw(4) << support << endl;
        }
        
        cout << "\n\n=== Summary ===" << endl;
        if (accuracy > 60) {
            cout << "The neural network successfully learned to classify profit categories\n" << endl;
        } else if (accuracy > 40) {
            cout << "The model shows learning capability, beating random chance (25%)\n" << endl;
        } else {
            cout << "Model performance suggests the features may not be sufficiently predictive\n" << endl;
        }
        
    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
