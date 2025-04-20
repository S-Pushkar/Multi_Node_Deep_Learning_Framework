#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include "neural_network.hpp"

using namespace std;

struct Config {
    int num_threads;
    int num_hidden_layers;
    vector<int> neurons_per_hidden;
    double learning_rate;
    vector<string> activation_functions;
    int num_epochs;
    vector<string> dataset_files;
    vector<string> metrics;
    // Removed: output_files
};

// Config parse_config(const string &config_file) {
//     Config config;
//     try {
//         YAML::Node node = YAML::LoadFile(config_file);

//         config.num_threads = node["num_threads"].as<int>();
//         config.num_hidden_layers = node["num_hidden_layers"].as<int>();
//         config.neurons_per_hidden = node["neurons_per_hidden"].as<vector<int>>();
//         config.learning_rate = node["learning_rate"].as<double>();
//         config.activation_functions = node["activation_functions"].as<vector<string>>();
//         config.num_epochs = node["num_epochs"].as<int>();
//         config.dataset_files = node["dataset_files"].as<vector<string>>();

//         if(config.dataset_files.size() != config.num_threads) {
//             throw runtime_error("Number of dataset files must match number of threads");
//         }

//         return config;
//     } catch(const YAML::Exception &e) {
//         cerr << "Error parsing YAML config: " << e.what() << endl;
//         exit(1);
//     }
// }
Config parse_config(const string &config_file) {
    Config config;
    try {
        YAML::Node node = YAML::LoadFile(config_file);

        config.num_threads = node["num_threads"].as<int>();
        config.num_hidden_layers = node["num_hidden_layers"].as<int>();
        config.neurons_per_hidden = node["neurons_per_hidden"].as<vector<int>>();
        config.learning_rate = node["learning_rate"].as<double>();
        config.activation_functions = node["activation_functions"].as<vector<string>>();
        config.num_epochs = node["num_epochs"].as<int>();
        config.dataset_files = node["dataset_files"].as<vector<string>>();
        if (node["metrics"])
            config.metrics = node["metrics"].as<vector<string>>();
        else
            config.metrics = {};

        if(config.dataset_files.size() != config.num_threads) {
            throw runtime_error("Number of dataset files must match number of threads");
        }

        return config;
    } catch(const YAML::Exception &e) {
        cerr << "Error parsing YAML config: " << e.what() << endl;
        exit(1);
    }
}
vector<Activation> parse_activations(const vector<string> &activations_str) {
    vector<Activation> activations;
    for(const auto &act : activations_str) {
        if(act == "sigmoid")
            activations.push_back(Activation::SIGMOID);
        else if(act == "tanh")
            activations.push_back(Activation::TANH);
        else if(act == "relu")
            activations.push_back(Activation::RELU);
        else if(act == "linear")
            activations.push_back(Activation::LINEAR);
        else {
            cerr << "Unknown activation function: " << act << endl;
            exit(1);
        }
    }
    return activations;
}

double calculate_mae(const vector<vector<double>>& preds, const vector<vector<double>>& targets) {
    double total = 0.0;
    for (size_t i = 0; i < preds.size(); ++i)
        for (size_t j = 0; j < preds[i].size(); ++j)
            total += abs(preds[i][j] - targets[i][j]);
    return total / (preds.size() * preds[0].size());
}

double calculate_rmse(const vector<vector<double>>& preds, const vector<vector<double>>& targets) {
    double total = 0.0;
    for (size_t i = 0; i < preds.size(); ++i)
        for (size_t j = 0; j < preds[i].size(); ++j)
            total += pow(preds[i][j] - targets[i][j], 2);
    return sqrt(total / (preds.size() * preds[0].size()));
}

double calculate_r2(const vector<vector<double>>& preds, const vector<vector<double>>& targets) {
    double mean = 0.0;
    int count = 0;
    for (const auto& t : targets)
        for (double val : t) {
            mean += val;
            count++;
        }
    mean /= count;

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        for (size_t j = 0; j < targets[i].size(); ++j) {
            ss_res += pow(targets[i][j] - preds[i][j], 2);
            ss_tot += pow(targets[i][j] - mean, 2);
        }
    }
    return 1.0 - (ss_res / ss_tot);
}

vector<vector<double>> load_dataset(const string &filename) {
    vector<vector<double>> dataset;
    ifstream file(filename);
    if(!file.is_open()) {
        cerr << "Failed to open dataset file: " << filename << endl;
        exit(1);
    }

    string line;
    while(getline(file, line)) {
        vector<double> row;
        size_t pos = 0;
        while((pos = line.find(',')) != string::npos) {
            string token = line.substr(0, pos);
            row.push_back(stod(token));
            line.erase(0, pos + 1);
        }
        row.push_back(stod(line));
        dataset.push_back(row);
    }

    return dataset;
}

vector<vector<vector<double>>> initialize_weights(const Config &config, int input_size) {
    vector<vector<vector<double>>> weights;
    int prev_size = input_size;

    for(int i = 0; i < config.num_hidden_layers; ++i) {
        int current_size = config.neurons_per_hidden[i];
        vector<vector<double>> layer_weights(current_size, vector<double>(prev_size));

        // Simple random initialization between -0.5 and 0.5
        for(auto &neuron_weights : layer_weights) {
            for(auto &w : neuron_weights) {
                w = (rand() / (double)RAND_MAX) - 0.5;
            }
        }

        weights.push_back(layer_weights);
        prev_size = current_size;
    }

    // Output layer weights
    int output_size = config.neurons_per_hidden.back();
    vector<vector<double>> output_weights(output_size, vector<double>(prev_size));
    for(auto &neuron_weights : output_weights) {
        for(auto &w : neuron_weights) {
            w = (rand() / (double)RAND_MAX) - 0.5;
        }
    }
    weights.push_back(output_weights);

    return weights;
}

vector<vector<double>> initialize_biases(const Config &config) {
    vector<vector<double>> biases;

    for(int i = 0; i < config.num_hidden_layers; ++i) {
        int current_size = config.neurons_per_hidden[i];
        biases.emplace_back(current_size, 0.0); // Initialize with zeros
    }

    // Output layer biases
    int output_size = config.neurons_per_hidden.back();
    biases.emplace_back(output_size, 0.0);

    return biases;
}

double calculate_accuracy(NeuralNetwork &nn,
                          const vector<vector<double>> &test_data,
                          const vector<vector<vector<double>>> &weights,
                          const vector<vector<double>> &biases,
                          int output_size) {
    int correct = 0;
    for(const auto &sample : test_data) {
        vector<double> input(sample.begin(), sample.end() - output_size);
        vector<double> target(sample.end() - output_size, sample.end());

        vector<double> prediction = nn.forward_pass(input, weights, biases);

        // For classification: compare predicted class vs true class
        int predicted_class = distance(prediction.begin(),
                                       max_element(prediction.begin(), prediction.end()));
        int true_class = distance(target.begin(),
                                  max_element(target.begin(), target.end()));

        if(predicted_class == true_class) {
            correct++;
        }
    }
    return static_cast<double>(correct) / test_data.size();
}

double calculate_mse(NeuralNetwork &nn,
                     const vector<vector<double>> &test_data,
                     const vector<vector<vector<double>>> &weights,
                     const vector<vector<double>> &biases,
                     int output_size) {
    double total_error = 0.0;
    for(const auto &sample : test_data) {
        vector<double> input(sample.begin(), sample.end() - output_size);
        vector<double> target(sample.end() - output_size, sample.end());

        vector<double> prediction = nn.forward_pass(input, weights, biases);

        for(size_t i = 0; i < prediction.size(); ++i) {
            double error = prediction[i] - target[i];
            total_error += error * error;
        }
    }
    return total_error / (test_data.size() * output_size);
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file.yml>" << endl;
        return 1;
    }

    // Parse configuration
    Config config = parse_config(argv[1]);
    vector<Activation> activations = parse_activations(config.activation_functions);

    // Load datasets - each thread gets its own dataset file
    vector<vector<vector<double>>> thread_data(config.num_threads);

    #pragma omp parallel num_threads(config.num_threads)
    {
        int thread_id = omp_get_thread_num();
        thread_data[thread_id] = load_dataset(config.dataset_files[thread_id]);

        if(thread_data[thread_id].empty()) {
            cerr << "Thread " << thread_id << " got empty dataset from "
                 << config.dataset_files[thread_id] << endl;
            exit(1);
        }
    }

    int input_size = thread_data[0][0].size() - config.neurons_per_hidden.back();
    int output_size = config.neurons_per_hidden.back();

    // Initialize global weights and biases
    auto global_weights = initialize_weights(config, input_size);
    auto global_biases = initialize_biases(config);

    // Create neural network instances for each thread
    vector<NeuralNetwork> networks;
    for(int i = 0; i < config.num_threads; ++i) {
        networks.emplace_back(
            config.num_hidden_layers,
            config.neurons_per_hidden,
            output_size,
            config.learning_rate,
            activations);
    }

    // Training loop
    for(int epoch = 0; epoch < config.num_epochs; ++epoch) {
        vector<vector<vector<vector<double>>>> all_updated_weights(config.num_threads);
        vector<vector<vector<double>>> all_updated_biases(config.num_threads);

        #pragma omp parallel num_threads(config.num_threads)
        {
            int thread_id = omp_get_thread_num();
            auto &nn = networks[thread_id];
            auto &data = thread_data[thread_id];

            // Local copies of weights and biases
            auto local_weights = global_weights;
            auto local_biases = global_biases;

            // Process each sample in this thread's dataset
            for(const auto &sample : data) {
                vector<double> input(sample.begin(), sample.end() - output_size);
                vector<double> target(sample.end() - output_size, sample.end());

                auto [updated_weights, updated_biases] = nn.train(input, target, local_weights, local_biases);
                local_weights = updated_weights;
                local_biases = updated_biases;
            }

            // Store the updated weights and biases
            all_updated_weights[thread_id] = local_weights;
            all_updated_biases[thread_id] = local_biases;
        }

// Average the weights and biases from all threads
        #pragma omp parallel for
        for(size_t l = 0; l < global_weights.size(); ++l) {
            for(size_t i = 0; i < global_weights[l].size(); ++i) {
                for(size_t j = 0; j < global_weights[l][i].size(); ++j) {
                    double sum = 0.0;
                    for(int t = 0; t < config.num_threads; ++t) {
                        sum += all_updated_weights[t][l][i][j];
                    }
                    global_weights[l][i][j] = sum / config.num_threads;
                }
            }
        }

        #pragma omp parallel for
        for(size_t l = 0; l < global_biases.size(); ++l) {
            for(size_t i = 0; i < global_biases[l].size(); ++i) {
                double sum = 0.0;
                for(int t = 0; t < config.num_threads; ++t) {
                    sum += all_updated_biases[t][l][i];
                }
                global_biases[l][i] = sum / config.num_threads;
            }
        }

        cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << " completed" << endl;
    }

    // Save the final model (single output file)
    const string output_file = "final_model.yml";
    ofstream out_file(output_file);
    if(!out_file.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        return 1;
    }

    // Save weights
    out_file << "weights:" << endl;
    for(const auto &layer : global_weights) {
        out_file << "  - layer:" << endl;
        for(const auto &neuron : layer) {
            out_file << "      - [";
            for(size_t i = 0; i < neuron.size(); ++i) {
                out_file << neuron[i];
                if(i != neuron.size() - 1)
                    out_file << ", ";
            }
            out_file << "]" << endl;
        }
    }

    // Save biases
    out_file << "biases:" << endl;
    for(const auto &layer : global_biases) {
        out_file << "  - [";
        for(size_t i = 0; i < layer.size(); ++i) {
            out_file << layer[i];
            if(i != layer.size() - 1)
                out_file << ", ";
        }
        out_file << "]" << endl;
    }
    // Evaluation Section
    cout << "\nEvaluating model performance..." << endl;

    // Load test data (using first dataset for demonstration)
    // auto test_data = load_dataset(config.dataset_files[2]);
    vector<vector<double>> test_data;
    for (const auto& file : config.dataset_files) {
        auto data = load_dataset(file);
        test_data.insert(test_data.end(), data.begin(), data.end());
    }


    vector<vector<double>> predictions, targets;
    for (const auto &sample : test_data) {
        vector<double> input(sample.begin(), sample.end() - output_size);
        vector<double> target(sample.end() - output_size, sample.end());
        vector<double> pred = networks[0].forward_pass(input, global_weights, global_biases);
        predictions.push_back(pred);
        targets.push_back(target);
    }

    // Calculate metrics
    double accuracy = calculate_accuracy(networks[0], test_data,
                                         global_weights, global_biases, output_size);
    double mse = calculate_mse(networks[0], test_data,
                               global_weights, global_biases, output_size);

    cout << fixed << setprecision(4);
    cout << "Evaluation Results:" << endl;
    cout << "------------------" << endl;
    // cout << "Accuracy: " << accuracy * 100 << "%" << endl;
    // cout << "Mean Squared Error: " << mse << endl;
    for (const auto& metric : config.metrics) {
    if (metric == "Accuracy" && output_size > 1) {
        double acc = calculate_accuracy(networks[0], test_data, global_weights, global_biases, output_size);
        cout << "Accuracy: " << acc * 100 << "%" << endl;
    } else if (metric == "Loss") {
        double mse = calculate_mse(networks[0], test_data, global_weights, global_biases, output_size);
        cout << "Mean Squared Error: " << mse << endl;
    } else if (metric == "MAE") {
        cout << "Mean Absolute Error: " << calculate_mae(predictions, targets) << endl;
    } else if (metric == "RMSE") {
        cout << "Root Mean Square Error: " << calculate_rmse(predictions, targets) << endl;
    } else if (metric == "R2") {
        cout << "R^2 Score: " << calculate_r2(predictions, targets) << endl;
    } else if (metric == "Confusion Matrix" && output_size > 1) {
    vector<vector<int>> confusion(output_size, vector<int>(output_size, 0));
    for (size_t i = 0; i < predictions.size(); ++i) {
        int true_label = distance(targets[i].begin(), max_element(targets[i].begin(), targets[i].end()));
        int pred_label = distance(predictions[i].begin(), max_element(predictions[i].begin(), predictions[i].end()));
        confusion[true_label][pred_label]++;
    }
    cout << "Confusion Matrix:\n";
    for (const auto& row : confusion) {
        for (int val : row)
            cout << setw(5) << val << " ";
        cout << endl;
    }
}
else if (metric == "AUC" && output_size > 1) {
    auto auc_1v_all = [&](int class_idx) -> double {
        vector<pair<double, int>> scores;
        for (size_t i = 0; i < predictions.size(); ++i) {
            scores.emplace_back(predictions[i][class_idx], targets[i][class_idx] > 0.5 ? 1 : 0);
        }
        sort(scores.begin(), scores.end(), greater<>());
        int pos = 0, neg = 0;
        for (const auto& s : scores) s.second ? ++pos : ++neg;
        if (pos == 0 || neg == 0) return 0.0;
        double tp = 0, fp = 0, auc = 0.0;
        for (const auto& [score, label] : scores) {
            if (label == 1) tp++;
            else {
                auc += tp;
                fp++;
            }
        }
        return auc / (pos * neg);
    };

    double macro_auc = 0.0;
    for (int i = 0; i < output_size; ++i)
        macro_auc += auc_1v_all(i);
    macro_auc /= output_size;

    cout << "AUC Score (macro-averaged): " << macro_auc << endl;
        }

    }
    if(output_size > 1) {
        // Only for classification
        // cout << "\nConfusion Matrix:" << endl;
        // Implement confusion matrix calculation here
    }

    cout << "Training complete. Model saved to " << output_file << endl;
    return 0;
}
