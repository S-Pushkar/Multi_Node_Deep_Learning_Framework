#include "neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <omp.h>

using namespace std;

vector<double> NeuralNetwork::apply_activation(const vector<double>& z, Activation& activation) {
    vector<double> result(z.size());
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
        double val = z[i];
        switch (activation) {
            case Activation::SIGMOID:
                result[i] = 1.0 / (1 + exp(-val));
                break;
            case Activation::TANH:
                result[i] = tanh(val);
                break;
            case Activation::RELU:
                result[i] = max(0.0, val);
                break;
            case Activation::LEAKY_RELU:
                result[i] = val > 0 ? val : 0.01 * val;
                break;
            case Activation::LINEAR:
                result[i] = val;
                break;
            default:
                throw invalid_argument("Unknown activation function");
        }
    }
    return result;
}

vector<double> NeuralNetwork::apply_activation_derivative(const vector<double>& z, Activation& activation) {
    vector<double> result(z.size());
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
        double val = z[i];
        switch (activation) {
            case Activation::SIGMOID: {
                double s = 1.0 / (1 + exp(-val));
                result[i] = s * (1 - s);
                break;
            }
            case Activation::TANH: {
                double t = tanh(val);
                result[i] = 1 - t * t;
                break;
            }
            case Activation::RELU:
                result[i] = val > 0 ? 1.0 : 0.0;
                break;
            case Activation::LEAKY_RELU:
                result[i] = val > 0 ? 1.0 : 0.01;
                break;
            case Activation::LINEAR:
                result[i] = 1.0;
                break;
            default:
                throw invalid_argument("Unknown activation function");
        }
    }
    return result;
}

vector<double> NeuralNetwork::subtract_vectors(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vector sizes do not match for subtraction");
    }
    vector<double> result(a.size());
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

vector<double> NeuralNetwork::hadamard_product(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vector sizes do not match for Hadamard product");
    }
    vector<double> result(a.size());
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

NeuralNetwork::NeuralNetwork(int num_hidden, const vector<int>& neurons_hidden, int output_size,
                            double lr, const vector<Activation>& activations)
    : num_hidden_layers(num_hidden), neurons_per_hidden(neurons_hidden),
      output_neurons(output_size), learning_rate(lr),
      activation_functions(activations) {
    if (activations.size() != (num_hidden + 1)) {
        throw invalid_argument("Number of activation functions must match hidden layers + output layer");
    }
}

vector<double> NeuralNetwork::forward_pass(const vector<double>& input,
                                             const vector<vector<vector<double>>>& weights,
                                             const vector<vector<double>>& biases) {
    vector<double> activation = input;
    
    for (int l = 0; l < num_hidden_layers + 1; ++l) {
        const vector<vector<double>>& W = weights[l];
        const vector<double>& b = biases[l];
        vector<double> z(W.size());

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W.size()); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < activation.size(); ++j) {
                sum += activation[j] * W[i][j];
            }
            sum += b[i];
            z[i] = sum;
        }

        activation = apply_activation(z, activation_functions[l]);
    }

    return activation;
}

pair<vector<vector<vector<double>>>, vector<vector<double>>> NeuralNetwork::train(
    const vector<double>& input,
    const vector<double>& target,
    const vector<vector<vector<double>>>& weights,
    const vector<vector<double>>& biases) {

    int num_layers = num_hidden_layers + 1;

    // Forward pass
    vector<vector<double>> activations = {input};
    vector<vector<double>> zs;

    for (int l = 0; l < num_layers; ++l) {
        const vector<double>& a_prev = activations.back();
        const vector<vector<double>>& W = weights[l];
        const vector<double>& b = biases[l];

        vector<double> z(W.size());
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W.size()); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < a_prev.size(); ++j) {
                sum += a_prev[j] * W[i][j];
            }
            sum += b[i];
            z[i] = sum;
        }
        zs.push_back(z);

        vector<double> a = apply_activation(z, activation_functions[l]);
        activations.push_back(a);
    }

    // Backward pass
    vector<vector<double>> deltas;
    vector<double> a_output = activations.back();
    vector<double> z_output = zs.back();
    vector<double> delta_output = hadamard_product(
        subtract_vectors(a_output, target),
        apply_activation_derivative(z_output, activation_functions.back())
    );
    deltas.push_back(delta_output);

    for (int l = num_layers - 2; l >= 0; --l) {
        vector<double> delta_next = deltas.back();
        vector<vector<double>> W_next = weights[l + 1];
        vector<double> z_prev = zs[l];

        vector<double> wT_delta(W_next[0].size(), 0.0);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W_next[0].size()); ++i) {
            for (size_t j = 0; j < W_next.size(); ++j) {
                wT_delta[i] += W_next[j][i] * delta_next[j];
            }
        }

        vector<double> delta_l = hadamard_product(
            wT_delta,
            apply_activation_derivative(z_prev, activation_functions[l])
        );
        deltas.push_back(delta_l);
    }

    reverse(deltas.begin(), deltas.end());

    // Update weights and biases
    vector<vector<vector<double>>> updated_weights = weights;
    vector<vector<double>> updated_biases = biases;

    for (int l = 0; l < num_layers; ++l) {
        vector<double> delta = deltas[l];
        vector<double> a_prev = activations[l];
        vector<vector<double>>& W = updated_weights[l];
        vector<double>& b = updated_biases[l];

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W.size()); ++i) {
            for (size_t j = 0; j < a_prev.size(); ++j) {
                W[i][j] -= learning_rate * a_prev[j] * delta[i];
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(b.size()); ++i) {
            b[i] -= learning_rate * delta[i];
        }
    }

    return {updated_weights, updated_biases};
}