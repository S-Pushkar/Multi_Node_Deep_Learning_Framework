#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>

using namespace std;

enum class Activation {
    Sigmoid,
    ReLU,
    Softmax,
    Tanh
};

class NeuralNetwork {
private:
    int num_hidden_layers;
    vector<int> neurons_per_hidden;
    float learning_rate;
    vector<Activation> activations;
    vector<vector<float>> inputs;
    vector<vector<float>> targets;
    vector<vector<vector<float>>> weights;
    vector<vector<float>> biases;

    vector<vector<float>> initialize_weights(int rows, int cols) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(-0.5f, 0.5f);
        
        vector<vector<float>> matrix(rows, vector<float>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = dist(gen);
            }
        }
        return matrix;
    }

    vector<float> initialize_biases(int size) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(-0.5f, 0.5f);
        
        vector<float> bias(size);
        for (int i = 0; i < size; ++i) {
            bias[i] = dist(gen);
        }
        return bias;
    }

    vector<float> matrix_vector_multiply(const vector<vector<float>>& mat, const vector<float>& vec) {
        if (mat.empty() || mat[0].size() != vec.size())
            throw invalid_argument("Matrix columns must match vector size");
        
        vector<float> result(mat.size(), 0.0f);
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    vector<vector<float>> transpose(const vector<vector<float>>& mat) {
        if (mat.empty()) return {};
        size_t rows = mat.size();
        size_t cols = mat[0].size();
        vector<vector<float>> transposed(cols, vector<float>(rows));
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                transposed[i][j] = mat[j][i];
            }
        }
        return transposed;
    }

    void apply_activation(vector<float>& vec, Activation func) {
        switch(func) {
            case Activation::Sigmoid:
                for (auto& x : vec) x = 1.0f / (1.0f + exp(-x));
                break;
            case Activation::ReLU:
                for (auto& x : vec) x = max(0.0f, x);
                break;
            case Activation::Softmax: {
                float max_val = *max_element(vec.begin(), vec.end());
                float sum = 0.0f;
                for (auto& x : vec) {
                    x = exp(x - max_val);
                    sum += x;
                }
                for (auto& x : vec) x /= sum;
                break;
            }
            case Activation::Tanh:
                for (auto& x : vec) x = tanh(x);
                break;
        }
    }

    vector<float> compute_activation_derivative(const vector<float>& vec, Activation func) {
        vector<float> derivative(vec.size());
        switch(func) {
            case Activation::Sigmoid:
                for (size_t i = 0; i < vec.size(); ++i)
                    derivative[i] = vec[i] * (1 - vec[i]);
                break;
            case Activation::ReLU:
                for (size_t i = 0; i < vec.size(); ++i)
                    derivative[i] = (vec[i] > 0) ? 1.0f : 0.0f;
                break;
            case Activation::Tanh:
                for (size_t i = 0; i < vec.size(); ++i) {
                    float t = tanh(vec[i]);
                    derivative[i] = 1 - t * t;
                }
                break;
            default:
                for (auto& d : derivative) d = 1.0f;
        }
        return derivative;
    }

    vector<float> elementwise_multiply(const vector<float>& a, const vector<float>& b) {
        if (a.size() != b.size())
            throw invalid_argument("Vectors must be same size");
            
        vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            result[i] = a[i] * b[i];
        return result;
    }

    vector<vector<float>> outer_product(const vector<float>& a, const vector<float>& b) {
        vector<vector<float>> result(a.size(), vector<float>(b.size()));
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < b.size(); ++j)
                result[i][j] = a[i] * b[j];
        return result;
    }

public:
    NeuralNetwork(int num_hidden, 
                 vector<int> neurons_per_hidden,
                 float lr,
                 vector<Activation> activations,
                 const vector<vector<float>>& inputs,
                 const vector<vector<float>>& targets)
        : num_hidden_layers(num_hidden),
          neurons_per_hidden(neurons_per_hidden),
          learning_rate(lr),
          activations(activations),
          inputs(inputs),
          targets(targets) {
        
        if (activations.size() != num_hidden_layers + 1)
            throw invalid_argument("Invalid number of activation functions");
            
        if (inputs.empty() || targets.empty())
            throw invalid_argument("Empty training data");
            
        int input_size = inputs[0].size();
        int output_size = targets[0].size();
        
        // Initialize weights and biases
        // Input to first hidden layer
        weights.push_back(initialize_weights(neurons_per_hidden[0], input_size));
        biases.push_back(initialize_biases(neurons_per_hidden[0]));
        
        // Hidden layers
        for (int i = 1; i < num_hidden_layers; ++i) {
            weights.push_back(initialize_weights(neurons_per_hidden[i], neurons_per_hidden[i-1]));
            biases.push_back(initialize_biases(neurons_per_hidden[i]));
        }
        
        // Last hidden to output layer
        weights.push_back(initialize_weights(output_size, neurons_per_hidden.back()));
        biases.push_back(initialize_biases(output_size));
    }

    pair<vector<vector<vector<float>>>, vector<vector<float>>> train() {
        for (size_t s = 0; s < inputs.size(); ++s) {
            const auto& input = inputs[s];
            const auto& target = targets[s];
            
            // Forward pass
            vector<vector<float>> activs = {input};
            for (size_t i = 0; i < weights.size(); ++i) {
                auto z = matrix_vector_multiply(weights[i], activs.back());
                // Add bias
                for (size_t j = 0; j < z.size(); ++j) {
                    z[j] += biases[i][j];
                }
                apply_activation(z, activations[i]);
                activs.push_back(z);
            }
            
            // Backward pass
            vector<vector<float>> deltas(weights.size());
            
            // Output layer
            vector<float> error(activs.back().size());
            for (size_t i = 0; i < error.size(); ++i)
                error[i] = activs.back()[i] - target[i];
            
            if (activations.back() == Activation::Softmax) {
                deltas.back() = error;
            } else {
                auto deriv = compute_activation_derivative(activs.back(), activations.back());
                deltas.back() = elementwise_multiply(error, deriv);
            }
            
            // Hidden layers
            for (int i = weights.size()-2; i >= 0; --i) {
                auto transposed = transpose(weights[i+1]);
                auto delta = matrix_vector_multiply(transposed, deltas[i+1]);
                auto deriv = compute_activation_derivative(activs[i+1], activations[i]);
                deltas[i] = elementwise_multiply(delta, deriv);
            }
            
            // Update weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                auto grad = outer_product(deltas[i], activs[i]);
                for (size_t r = 0; r < weights[i].size(); ++r) {
                    for (size_t c = 0; c < weights[i][r].size(); ++c) {
                        weights[i][r][c] -= learning_rate * grad[r][c];
                    }
                }
                // Update biases
                for (size_t j = 0; j < biases[i].size(); ++j) {
                    biases[i][j] -= learning_rate * deltas[i][j];
                }
            }
        }
        return {weights, biases};
    }
};

int main() {
    vector<vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<float>> targets = {{0}, {1}, {1}, {0}};

    NeuralNetwork nn(1,                                       // 1 hidden layer
                     {2},                                     // 2 neurons in hidden layer
                     0.1,                                     // learning rate
                     {Activation::ReLU, Activation::Sigmoid}, // activations
                     inputs,
                     targets);

    auto trained = nn.train();
    auto trained_weights = trained.first;
    auto trained_biases = trained.second;

    for (size_t i = 0; i < trained_weights.size(); ++i) {
        cout << "Layer " << i << " weights:\n";
        for (const auto &row : trained_weights[i]) {
            for (const auto &weight : row) {
                cout << weight << " ";
            }
            cout << "\n";
        }
        cout << "Layer " << i << " biases:\n";
        for (const auto &bias : trained_biases[i]) {
            cout << bias << " ";
        }
        cout << "\n-----\n";
    }
    return 0;
}