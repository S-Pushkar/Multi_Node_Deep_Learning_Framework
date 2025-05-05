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
    string test_file;
    vector<string> metrics;
    // Removed: output_files
};

struct NormParams {
    vector<double> feature_means;
    vector<double> feature_stds;
    vector<double> target_means;
    vector<double> target_stds;
};

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

        if (node["test_file"]) {
            config.test_file = node["test_file"].as<string>();
        } else {
            config.test_file = "";
        }

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
        else if(act == "leaky_relu")
            activations.push_back(Activation::LEAKY_RELU);
        else if(act == "linear")
            activations.push_back(Activation::LINEAR);
        else if(act == "softmax")
            activations.push_back(Activation::SOFTMAX);
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

NormParams normalize_dataset(vector<vector<double>>& data, int num_features, int output_size) {
    NormParams params;
    params.feature_means.resize(num_features, 0.0);
    params.feature_stds.resize(num_features, 0.0);
    params.target_means.resize(output_size, 0.0);
    params.target_stds.resize(output_size, 0.0);

    // 1. Calculate means
    for (const auto &sample : data) {
        for (int i = 0; i < num_features; ++i) {
            params.feature_means[i] += sample[i];
        }
        for (int i = 0; i < output_size; ++i) {
            params.target_means[i] += sample[num_features + i];
        }
    }

    for (int i = 0; i < num_features; ++i) {
        params.feature_means[i] /= data.size();
    }
    for (int i = 0; i < output_size; ++i) {
        params.target_means[i] /= data.size();
    }

    // 2. Calculate standard deviations
    for (const auto &sample : data) {
        for (int i = 0; i < num_features; ++i) {
            double diff = sample[i] - params.feature_means[i];
            params.feature_stds[i] += diff * diff;
        }
        for (int i = 0; i < output_size; ++i) {
            double diff = sample[num_features + i] - params.target_means[i];
            params.target_stds[i] += diff * diff;
        }
    }

    // Handle division by zero and minimum std
    const double min_std = 1e-7;  // Prevent division by zero
    for (int i = 0; i < num_features; ++i) {
        params.feature_stds[i] = sqrt(params.feature_stds[i] / data.size());
        if (params.feature_stds[i] < min_std) {
            cerr << "Warning: Feature " << i << " has near-zero std (" 
                 << params.feature_stds[i] << "), using min_std" << endl;
            params.feature_stds[i] = min_std;
        }
    }
    for (int i = 0; i < output_size; ++i) {
        params.target_stds[i] = sqrt(params.target_stds[i] / data.size());
        if (params.target_stds[i] < min_std) {
            cerr << "Warning: Target " << i << " has near-zero std (" 
                 << params.target_stds[i] << "), using min_std" << endl;
            params.target_stds[i] = min_std;
        }
    }

    // 3. Apply normalization
    for (auto &sample : data) {
        for (int i = 0; i < num_features; ++i) {
            sample[i] = (sample[i] - params.feature_means[i]) / params.feature_stds[i];
        }
        for (int i = 0; i < output_size; ++i) {
            sample[num_features + i] = (sample[num_features + i] - params.target_means[i]) / params.target_stds[i];
        }
    }

    return params;
}

vector<double> denormalize_output(const vector<double>& output, const NormParams& params) {
    vector<double> denormalized(output.size());
    for(size_t i = 0; i < output.size(); ++i) {
        denormalized[i] = output[i] * params.target_stds[i] + params.target_means[i];
    }
    return denormalized;
}

vector<vector<vector<double>>> initialize_weights(const Config &config, int input_size) {
    vector<vector<vector<double>>> weights;
    int prev_size = input_size;
    int output_size = config.neurons_per_hidden.back();

    for(int i = 0; i < config.num_hidden_layers; ++i) {
        int current_size = config.neurons_per_hidden[i];
        vector<vector<double>> layer_weights(current_size, vector<double>(prev_size));

        // He initialization for Leaky ReLU
        double limit = sqrt(2.0 / prev_size);
        for(auto &neuron_weights : layer_weights) {
            for(auto &w : neuron_weights) {
                w = ((rand() / (double)RAND_MAX) * 2 * limit) - limit;
            }
        }
        weights.push_back(layer_weights);
        prev_size = current_size;
    }

    // Output layer (linear activation)
    double output_limit = sqrt(2.0 / prev_size);
    vector<vector<double>> output_weights(output_size, vector<double>(prev_size));
    for(auto &neuron_weights : output_weights) {
        for(auto &w : neuron_weights) {
            w = ((rand() / (double)RAND_MAX) * 2 * output_limit) - output_limit;
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

double calculate_loss(NeuralNetwork &nn,
                     const vector<vector<double>> &test_data,
                     const vector<vector<vector<double>>> &weights,
                     const vector<vector<double>> &biases,
                     int output_size) {
    double total_loss = 0.0;

    for(const auto &sample : test_data) {
        vector<double> input(sample.begin(), sample.end() - output_size);
        vector<double> target(sample.end() - output_size, sample.end());

        vector<double> prediction = nn.forward_pass(input, weights, biases);
        total_loss += nn.cross_entropy_loss(prediction, target);
    }
    
    return total_loss / test_data.size();
}

double calculate_mse(NeuralNetwork &nn,
                     const vector<vector<double>> &test_data,
                     const vector<vector<vector<double>>> &weights,
                     const vector<vector<double>> &biases,
                     int output_size,
                     const NormParams &params) {
    double total_error = 0.0;
    for(const auto &sample : test_data) {
        vector<double> input(sample.begin(), sample.end() - output_size);
        vector<double> target(sample.end() - output_size, sample.end());

        vector<double> prediction = nn.forward_pass(input, weights, biases);

        prediction = denormalize_output(prediction, params);
        target = denormalize_output(target, params);

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
    int output_size = config.neurons_per_hidden.back();

    // Load and normalize training data
    vector<vector<double>> all_data_raw;
    for (const auto &file : config.dataset_files) {
        auto data = load_dataset(file);
        all_data_raw.insert(all_data_raw.end(), data.begin(), data.end());
    }

    int input_size = all_data_raw[0].size() - output_size;
    vector<vector<double>> all_data_normalized = all_data_raw;
    NormParams norm_params = normalize_dataset(all_data_normalized, input_size, output_size);

    // Split normalized data into threads
    int samples_per_thread = all_data_normalized.size() / config.num_threads;
    vector<vector<vector<double>>> thread_data(config.num_threads);
    for (int i = 0; i < config.num_threads; ++i) {
        int start = i * samples_per_thread;
        int end = (i == config.num_threads - 1) ? all_data_normalized.size() : start + samples_per_thread;
        thread_data[i] = vector<vector<double>>(all_data_normalized.begin() + start, all_data_normalized.begin() + end);
    }

    // Initialize global weights and biases
    auto global_weights = initialize_weights(config, input_size);
    auto global_biases = initialize_biases(config);

    // Create neural network instances
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

            auto local_weights = global_weights;
            auto local_biases = global_biases;

            for(const auto &sample : data) {
                vector<double> input(sample.begin(), sample.end() - output_size);
                vector<double> target(sample.end() - output_size, sample.end());

                auto [updated_weights, updated_biases] = nn.train(input, target, local_weights, local_biases);
                local_weights = updated_weights;
                local_biases = updated_biases;
            }

            all_updated_weights[thread_id] = local_weights;
            all_updated_biases[thread_id] = local_biases;
        }

        // Average weights and biases
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

    // Save model
    const string output_file = "final_model.yml";
    ofstream out_file(output_file);
    if(!out_file.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        return 1;
    }

    // Save weights and biases (same as before)
    out_file << "weights:" << endl;
    for(const auto &layer : global_weights) {
        out_file << "  - layer:" << endl;
        for(const auto &neuron : layer) {
            out_file << "      - [";
            for(size_t i = 0; i < neuron.size(); ++i) {
                out_file << neuron[i];
                if(i != neuron.size() - 1) out_file << ", ";
            }
            out_file << "]" << endl;
        }
    }
    out_file << "biases:" << endl;
    for(const auto &layer : global_biases) {
        out_file << "  - [";
        for(size_t i = 0; i < layer.size(); ++i) {
            out_file << layer[i];
            if(i != layer.size() - 1) out_file << ", ";
        }
        out_file << "]" << endl;
    }

    // Evaluation with proper denormalization
    cout << "\nEvaluating model performance..." << endl;
    
    // Load fresh test data (or use holdout set)
    vector<vector<double>> test_data_raw = load_dataset(config.dataset_files[0]); // Using first file as test
    vector<vector<double>> test_data_normalized = test_data_raw;
    normalize_dataset(test_data_normalized, input_size, output_size);

    vector<vector<double>> predictions, true_targets;
    for (size_t i = 0; i < test_data_normalized.size(); ++i) {
        vector<double> input(test_data_normalized[i].begin(), test_data_normalized[i].end() - output_size);
        vector<double> pred = networks[0].forward_pass(input, global_weights, global_biases);
        
        // Denormalize predictions
        vector<double> denorm_pred = denormalize_output(pred, norm_params);
        vector<double> original_target(test_data_raw[i].end() - output_size, test_data_raw[i].end());
        
        predictions.push_back(denorm_pred);
        true_targets.push_back(original_target);
    }

    // Calculate and print metrics
    cout << fixed << setprecision(6);
    cout << "Evaluation Results:" << endl;
    cout << "------------------" << endl;

    for (const auto& metric : config.metrics) {
        if (metric == "Accuracy" && output_size > 1) {
            // For classification, use normalized data directly
            double acc = calculate_accuracy(networks[0], test_data_normalized, 
                                          global_weights, global_biases, output_size);
            cout << "Accuracy: " << acc * 100 << "%" << endl;
        } 
        else if (metric == "MSE") {
            cout << "Mean Squared Error: " << calculate_mae(predictions, true_targets) << endl;
        }
        else if (metric == "MAE") {
            cout << "Mean Absolute Error: " << calculate_mae(predictions, true_targets) << endl;
        }
        else if (metric == "RMSE") {
            cout << "Root Mean Square Error: " << calculate_rmse(predictions, true_targets) << endl;
        }
        else if (metric == "R2") {
            cout << "R-squared: " << calculate_r2(predictions, true_targets) << endl;
        } else if (metric == "Cross-Entropy") {
            double loss = calculate_loss(networks[0], test_data_normalized, 
                                       global_weights, global_biases, output_size);
            cout << "Cross-Entropy Loss: " << loss << endl;
        }
        else if (metric == "Confusion Matrix" && output_size > 1) {
            vector<vector<int>> confusion(output_size, vector<int>(output_size, 0));
            for (size_t i = 0; i < predictions.size(); ++i) {
                int true_label = distance(true_targets[i].begin(), 
                                        max_element(true_targets[i].begin(), true_targets[i].end()));
                int pred_label = distance(predictions[i].begin(), 
                                        max_element(predictions[i].begin(), predictions[i].end()));
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
                    scores.emplace_back(predictions[i][class_idx], 
                                      true_targets[i][class_idx] > 0.5 ? 1 : 0);
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

    cout << "\nTraining complete. Model saved to " << output_file << endl;

    if (!config.test_file.empty()) {
        cout << "\nEvaluating on separate test file: " << config.test_file << endl;
        
        // Load and normalize test file data
        vector<vector<double>> test_file_raw = load_dataset(config.test_file);
        if (test_file_raw.empty()) {
            cerr << "Warning: Test file is empty or could not be loaded" << endl;
        } else {
            // Verify test file has same dimensions as training data
            if (test_file_raw[0].size() != input_size + output_size) {
                cerr << "Error: Test file has different dimensions than training data ("
                     << test_file_raw[0].size() << " vs " << input_size + output_size << ")" << endl;
            } else {
                vector<vector<double>> test_file_normalized = test_file_raw;
                NormParams test_norm_params = normalize_dataset(test_file_normalized, input_size, output_size);

                vector<vector<double>> test_predictions, test_true_targets;
                for (size_t i = 0; i < test_file_normalized.size(); ++i) {
                    vector<double> input(test_file_normalized[i].begin(), test_file_normalized[i].end() - output_size);
                    vector<double> pred = networks[0].forward_pass(input, global_weights, global_biases);
                    
                    // Denormalize predictions
                    vector<double> denorm_pred = denormalize_output(pred, norm_params);
                    vector<double> original_target(test_file_raw[i].end() - output_size, test_file_raw[i].end());
                    
                    test_predictions.push_back(denorm_pred);
                    test_true_targets.push_back(original_target);
                }

                // Calculate and print metrics for test file
                cout << fixed << setprecision(6);
                cout << "Test File Evaluation Results:" << endl;
                cout << "----------------------------" << endl;

                for (const auto& metric : config.metrics) {
                    if (metric == "Accuracy" && output_size > 1) {
                        double acc = calculate_accuracy(networks[0], test_file_normalized, 
                                                      global_weights, global_biases, output_size);
                        cout << "Accuracy: " << acc * 100 << "%" << endl;
                    } 
                    else if (metric == "MSE") {
                        cout << "Mean Squared Error: " << calculate_mae(test_predictions, test_true_targets) << endl;
                    }
                    else if (metric == "MAE") {
                        cout << "Mean Absolute Error: " << calculate_mae(test_predictions, test_true_targets) << endl;
                    }
                    else if (metric == "RMSE") {
                        cout << "Root Mean Square Error: " << calculate_rmse(test_predictions, test_true_targets) << endl;
                    }
                    else if (metric == "R2") {
                        cout << "R-squared: " << calculate_r2(test_predictions, test_true_targets) << endl;
                    } 
                    else if (metric == "Cross-Entropy") {
                        double loss = calculate_loss(networks[0], test_file_normalized, 
                                                   global_weights, global_biases, output_size);
                        cout << "Cross-Entropy Loss: " << loss << endl;
                    }
                    else if (metric == "Confusion Matrix" && output_size > 1) {
                        vector<vector<int>> confusion(output_size, vector<int>(output_size, 0));
                        for (size_t i = 0; i < test_predictions.size(); ++i) {
                            int true_label = distance(test_true_targets[i].begin(), 
                                                    max_element(test_true_targets[i].begin(), test_true_targets[i].end()));
                            int pred_label = distance(test_predictions[i].begin(), 
                                                    max_element(test_predictions[i].begin(), test_predictions[i].end()));
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
                            for (size_t i = 0; i < test_predictions.size(); ++i) {
                                scores.emplace_back(test_predictions[i][class_idx], 
                                                  test_true_targets[i][class_idx] > 0.5 ? 1 : 0);
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

                // Print sample predictions for verification
                cout << "\nSample predictions from test file:\n";
                for (int i = 0; i < min(5, (int)test_predictions.size()); i++) {
                    cout << "Sample " << i << ": Pred = [";
                    for (auto p : test_predictions[i]) cout << p << " ";
                    cout << "], True = [";
                    for (auto t : test_true_targets[i]) cout << t << " ";
                    cout << "]\n";
                }
            }
        }
    }

    return 0;
}