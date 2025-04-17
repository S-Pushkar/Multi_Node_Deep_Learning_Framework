#include "neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

std::vector<double> NeuralNetwork::apply_activation(const std::vector<double>& z, Activation& activation) {
    std::vector<double> result;
    for (double val : z) {
        switch (activation) {
            case Activation::SIGMOID:
                result.push_back(1.0 / (1 + std::exp(-val)));
                break;
            case Activation::TANH:
                result.push_back(std::tanh(val));
                break;
            case Activation::RELU:
                result.push_back(std::max(0.0, val));
                break;
            case Activation::LINEAR:
                result.push_back(val);
                break;
            default:
                throw std::invalid_argument("Unknown activation function");
        }
    }
    return result;
}

std::vector<double> NeuralNetwork::apply_activation_derivative(const std::vector<double>& z, Activation& activation) {
    std::vector<double> result;
    for (double val : z) {
        switch (activation) {
            case Activation::SIGMOID: {
                double s = 1.0 / (1 + std::exp(-val));
                result.push_back(s * (1 - s));
                break;
            }
            case Activation::TANH: {
                double t = std::tanh(val);
                result.push_back(1 - t * t);
                break;
            }
            case Activation::RELU:
                result.push_back(val > 0 ? 1.0 : 0.0);
                break;
            case Activation::LINEAR:
                result.push_back(1.0);
                break;
            default:
                throw std::invalid_argument("Unknown activation function");
        }
    }
    return result;
}

std::vector<double> NeuralNetwork::subtract_vectors(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes do not match for subtraction");
    }
    std::vector<double> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] - b[i]);
    }
    return result;
}

std::vector<double> NeuralNetwork::hadamard_product(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes do not match for Hadamard product");
    }
    std::vector<double> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] * b[i]);
    }
    return result;
}

NeuralNetwork::NeuralNetwork(int num_hidden, const std::vector<int>& neurons_hidden, int output_size,
                            double lr, const std::vector<Activation>& activations)
    : num_hidden_layers(num_hidden), neurons_per_hidden(neurons_hidden),
      output_neurons(output_size), learning_rate(lr),
      activation_functions(activations) {
    if (activations.size() != (num_hidden + 1)) {
        throw std::invalid_argument("Number of activation functions must match hidden layers + output layer");
    }
}

std::vector<double> NeuralNetwork::forward_pass(const std::vector<double>& input,
                                             const std::vector<std::vector<std::vector<double>>>& weights,
                                             const std::vector<std::vector<double>>& biases) {
    std::vector<double> activation = input;
    
    for (int l = 0; l < num_hidden_layers + 1; ++l) {
        const std::vector<std::vector<double>>& W = weights[l];
        const std::vector<double>& b = biases[l];
        std::vector<double> z;

        for (size_t i = 0; i < W.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < activation.size(); ++j) {
                sum += activation[j] * W[i][j];
            }
            sum += b[i];
            z.push_back(sum);
        }

        activation = apply_activation(z, activation_functions[l]);
    }

    return activation;
}

std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> NeuralNetwork::train(
    const std::vector<double>& input,
    const std::vector<double>& target,
    const std::vector<std::vector<std::vector<double>>>& weights,
    const std::vector<std::vector<double>>& biases) {

    int num_layers = num_hidden_layers + 1;

    // Forward pass
    std::vector<std::vector<double>> activations = {input};
    std::vector<std::vector<double>> zs;

    for (int l = 0; l < num_layers; ++l) {
        const std::vector<double>& a_prev = activations.back();
        const std::vector<std::vector<double>>& W = weights[l];
        const std::vector<double>& b = biases[l];

        std::vector<double> z;
        for (size_t i = 0; i < W.size(); ++i) {
            double sum = 0.0;
            if (W[i].size() != a_prev.size()) {
                throw std::invalid_argument("Weight matrix dimension mismatch in layer " + std::to_string(l));
            }
            for (size_t j = 0; j < a_prev.size(); ++j) {
                sum += a_prev[j] * W[i][j];
            }
            sum += b[i];
            z.push_back(sum);
        }
        zs.push_back(z);

        std::vector<double> a = apply_activation(z, activation_functions[l]);
        activations.push_back(a);
    }

    // Backward pass
    std::vector<std::vector<double>> deltas;
    std::vector<double> a_output = activations.back();
    std::vector<double> z_output = zs.back();
    std::vector<double> delta_output = hadamard_product(
        subtract_vectors(a_output, target),
        apply_activation_derivative(z_output, activation_functions.back())
    );
    deltas.push_back(delta_output);

    for (int l = num_layers - 2; l >= 0; --l) {
        std::vector<double> delta_next = deltas.back();
        std::vector<std::vector<double>> W_next = weights[l + 1];
        std::vector<double> z_prev = zs[l];

        std::vector<double> wT_delta(W_next[0].size(), 0.0);
        for (size_t i = 0; i < W_next[0].size(); ++i) {
            for (size_t j = 0; j < W_next.size(); ++j) {
                wT_delta[i] += W_next[j][i] * delta_next[j];
            }
        }

        std::vector<double> delta_l = hadamard_product(
            wT_delta,
            apply_activation_derivative(z_prev, activation_functions[l])
        );
        deltas.push_back(delta_l);
    }

    std::reverse(deltas.begin(), deltas.end());

    // Update weights and biases
    std::vector<std::vector<std::vector<double>>> updated_weights = weights;
    std::vector<std::vector<double>> updated_biases = biases;

    for (int l = 0; l < num_layers; ++l) {
        std::vector<double> delta = deltas[l];
        std::vector<double> a_prev = activations[l];
        std::vector<std::vector<double>>& W = updated_weights[l];
        std::vector<double>& b = updated_biases[l];

        for (size_t i = 0; i < W.size(); ++i) {
            for (size_t j = 0; j < a_prev.size(); ++j) {
                W[i][j] -= learning_rate * a_prev[j] * delta[i];
            }
        }

        for (size_t i = 0; i < b.size(); ++i) {
            b[i] -= learning_rate * delta[i];
        }
    }

    return {updated_weights, updated_biases};
}