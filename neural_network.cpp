#include "neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <omp.h>

using namespace std;

vector<double> NeuralNetwork::softmax(const vector<double>& z) {
    vector<double> res(z.size());
    double max_z = *max_element(z.begin(), z.end());
    double sum = 0.0;
    
    // Combined exp and sum calculation
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
        res[i] = exp(z[i] - max_z);
        sum += res[i];
    }

    // Normalization with single loop
    double inv_sum = 1.0 / sum;  // Multiplication is faster than division
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(res.size()); ++i) {
        res[i] *= inv_sum;
    }
    
    return res;
}

vector<double> NeuralNetwork::apply_activation(const vector<double>& z, const Activation& activation) {
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
            case Activation::SOFTMAX:
                break;
            default:
                throw invalid_argument("Unknown activation function");
        }
    }

    if (activation == Activation::SOFTMAX) {
        return softmax(z);
    }

    return result;
}

vector<double> NeuralNetwork::apply_activation_derivative(const vector<double>& z, const Activation& activation) {
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
            case Activation::SOFTMAX:
                throw invalid_argument("Softmax derivative should not be called directly");
                break;
            default:
                throw invalid_argument("Unknown activation function");
        }
    }
    return result;
}

double NeuralNetwork::cross_entropy_loss(const vector<double>& pred, const vector<double>& target) {
    double loss = 0.0;
    #pragma omp parallel for reduction(-:loss)
    for (int i = 0; i < static_cast<int>(pred.size()); ++i) {
        loss += target[i] * log(max(pred[i], 1e-15));
    }
    return -loss;
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

    const int num_layers = num_hidden_layers + 1;

    // ================== FORWARD PASS ==================
    vector<vector<double>> activations = {input};
    vector<vector<double>> zs;

    for (int l = 0; l < num_layers; ++l) {
        const vector<double>& a_prev = activations.back();
        const vector<vector<double>>& W = weights[l];
        const vector<double>& b = biases[l];
        const Activation activation = activation_functions[l];

        vector<double> z(W.size());
        
        // Parallel matrix-vector multiplication + bias
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W.size()); ++i) {
            double sum = b[i];  // Start with bias
            #pragma omp simd
            for (size_t j = 0; j < a_prev.size(); ++j) {
                sum += a_prev[j] * W[i][j];
            }
            z[i] = sum;
        }
        zs.push_back(z);

        // Apply activation (with special softmax handling)
        if (l == num_layers - 1 && activation == Activation::SOFTMAX) {
            activations.push_back(softmax(z));
        } else {
            activations.push_back(apply_activation(z, activation));
        }
    }

    // ================== BACKWARD PASS ==================
    vector<vector<double>> deltas;
    vector<double> a_output = activations.back();
    vector<double> z_output = zs.back();

    // Output layer delta - special handling for softmax + cross-entropy
    vector<double> delta_output(a_output.size());
    if (activation_functions.back() == Activation::SOFTMAX) {
        // Simplified gradient: ∂L/∂z = (a - y) for softmax + cross-entropy
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(a_output.size()); ++i) {
            delta_output[i] = a_output[i] - target[i];
        }
    } else {
        // Standard backprop for other activations
        vector<double> deriv = apply_activation_derivative(z_output, activation_functions.back());
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(a_output.size()); ++i) {
            delta_output[i] = (a_output[i] - target[i]) * deriv[i];
        }
    }
    deltas.push_back(delta_output);

    // Hidden layer deltas (backwards pass)
    for (int l = num_layers - 2; l >= 0; --l) {
        const vector<double>& delta_next = deltas.back();
        const vector<vector<double>>& W_next = weights[l + 1];
        const vector<double>& z_prev = zs[l];
        const Activation activation = activation_functions[l];

        // Calculate W^T * delta_next
        vector<double> wT_delta(W_next[0].size(), 0.0);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W_next[0].size()); ++i) {
            double sum = 0.0;
            #pragma omp simd
            for (size_t j = 0; j < W_next.size(); ++j) {
                sum += W_next[j][i] * delta_next[j];
            }
            wT_delta[i] = sum;
        }

        // Hadamard product with activation derivative
        vector<double> delta_l(wT_delta.size());
        vector<double> deriv = apply_activation_derivative(z_prev, activation);
        
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(delta_l.size()); ++i) {
            delta_l[i] = wT_delta[i] * deriv[i];
        }
        
        deltas.push_back(delta_l);
    }
    reverse(deltas.begin(), deltas.end());

    // ================== UPDATE WEIGHTS ==================
    vector<vector<vector<double>>> updated_weights = weights;
    vector<vector<double>> updated_biases = biases;

    for (int l = 0; l < num_layers; ++l) {
        const vector<double>& delta = deltas[l];
        const vector<double>& a_prev = activations[l];
        vector<vector<double>>& W = updated_weights[l];
        vector<double>& b = updated_biases[l];

        // Update weights
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(W.size()); ++i) {
            const double delta_i = delta[i] * learning_rate;
            #pragma omp simd
            for (size_t j = 0; j < a_prev.size(); ++j) {
                W[i][j] -= delta_i * a_prev[j];
            }
        }

        // Update biases
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(b.size()); ++i) {
            b[i] -= learning_rate * delta[i];
        }
    }

    return {updated_weights, updated_biases};
}