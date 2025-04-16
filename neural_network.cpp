#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

enum class Activation {
    SIGMOID,
    TANH,
    RELU,
    LINEAR
};

class NeuralNetwork {
private:
    int num_hidden_layers;
    vector<int> neurons_per_hidden;
    int output_neurons;
    double learning_rate;
    vector<Activation> activation_functions;

    vector<double> apply_activation(const vector<double>& z, Activation& activation) {
        vector<double> result;
        for (double val : z) {
            switch (activation) {
                case Activation::SIGMOID:
                    result.push_back(1.0 / (1 + exp(-val)));
                    break;
                case Activation::TANH:
                    result.push_back(tanh(val));
                    break;
                case Activation::RELU:
                    result.push_back(max(0.0, val));
                    break;
                case Activation::LINEAR:
                    result.push_back(val);
                    break;
                default:
                    throw invalid_argument("Unknown activation function");
            }
        }
        return result;
    }

    vector<double> apply_activation_derivative(const vector<double>& z, Activation& activation) {
        vector<double> result;
        for (double val : z) {
            switch (activation) {
                case Activation::SIGMOID: {
                    double s = 1.0 / (1 + exp(-val));
                    result.push_back(s * (1 - s));
                    break;
                }
                case Activation::TANH: {
                    double t = tanh(val);
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
                    throw invalid_argument("Unknown activation function");
            }
        }
        return result;
    }

    vector<double> subtract_vectors(const vector<double>& a, const vector<double>& b) {
        if (a.size() != b.size()) {
            throw invalid_argument("Vector sizes do not match for subtraction");
        }
        vector<double> result;
        for (size_t i = 0; i < a.size(); ++i) {
            result.push_back(a[i] - b[i]);
        }
        return result;
    }

    vector<double> hadamard_product(const vector<double>& a, const vector<double>& b) {
        if (a.size() != b.size()) {
            throw invalid_argument("Vector sizes do not match for Hadamard product");
        }
        vector<double> result;
        for (size_t i = 0; i < a.size(); ++i) {
            result.push_back(a[i] * b[i]);
        }
        return result;
    }

public:
    NeuralNetwork(int num_hidden, const vector<int>& neurons_hidden, int output_size,
                  double lr, const vector<Activation>& activations, int epochs)
        : num_hidden_layers(num_hidden), neurons_per_hidden(neurons_hidden),
          output_neurons(output_size), learning_rate(lr),
          activation_functions(activations) {
        if (activations.size() != (num_hidden + 1)) {
            throw invalid_argument("Number of activation functions must match hidden layers + output layer");
        }
    }

    vector<double> forward_pass(const vector<double>& input,
                               const vector<vector<vector<double>>>& weights,
                               const vector<vector<double>>& biases) {
        vector<double> activation = input;
        
        for (int l = 0; l < num_hidden_layers + 1; ++l) {
            const vector<vector<double>>& W = weights[l];
            const vector<double>& b = biases[l];
            vector<double> z;

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

    pair<vector<vector<vector<double>>>, vector<vector<double>>> train(
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

            vector<double> z;
            for (size_t i = 0; i < W.size(); ++i) {
                double sum = 0.0;
                if (W[i].size() != a_prev.size()) {
                    throw invalid_argument("Weight matrix dimension mismatch in layer " + to_string(l));
                }
                for (size_t j = 0; j < a_prev.size(); ++j) {
                    sum += a_prev[j] * W[i][j];
                }
                sum += b[i];
                z.push_back(sum);
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
            for (size_t i = 0; i < W_next[0].size(); ++i) {
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
};

void print_weights_biases(const vector<vector<vector<double>>>& weights, 
                         const vector<vector<double>>& biases) {
    cout << fixed << setprecision(4);
    for (size_t l = 0; l < weights.size(); ++l) {
        cout << "Layer " << l + 1 << " weights:" << endl;
        for (size_t i = 0; i < weights[l].size(); ++i) {
            cout << "  Neuron " << i + 1 << ": [";
            for (size_t j = 0; j < weights[l][i].size(); ++j) {
                cout << weights[l][i][j];
                if (j < weights[l][i].size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        cout << "Layer " << l + 1 << " biases: [";
        for (size_t i = 0; i < biases[l].size(); ++i) {
            cout << biases[l][i];
            if (i < biases[l].size() - 1) cout << ", ";
        }
        cout << "]" << endl << endl;
    }
    cout << "----------------------------------------" << endl;
}

int main() {
    // Network configuration
    int num_hidden_layers = 2;
    vector<int> neurons_per_hidden = {4, 3}; // 4 neurons in first hidden layer, 3 in second
    int output_neurons = 2;
    double learning_rate = 0.1;
    vector<Activation> activation_functions = {Activation::RELU, Activation::TANH, Activation::SIGMOID}; // for hidden layers and output
    int epochs = 200;

    // Create neural network
    NeuralNetwork nn(num_hidden_layers, neurons_per_hidden, output_neurons,
                    learning_rate, activation_functions, epochs);

    // Sample input and target
    vector<double> input = {0.5, -0.2, 0.8};
    vector<double> target = {0.7, 0.3};

    // Initialize weights and biases (random values for demonstration)
    vector<vector<vector<double>>> weights = {
        // Layer 1 weights (4 neurons, each with 3 inputs)
        {{0.1, -0.2, 0.3}, {0.4, 0.1, -0.1}, {-0.2, 0.3, 0.1}, {0.2, -0.1, 0.4}},
        // Layer 2 weights (3 neurons, each with 4 inputs)
        {{0.2, -0.1, 0.3, 0.1}, {0.1, 0.4, -0.2, 0.3}, {-0.3, 0.2, 0.1, -0.2}},
        // Output layer weights (2 neurons, each with 3 inputs)
        {{0.3, -0.2, 0.1}, {0.1, 0.4, -0.3}}
    };

    vector<vector<double>> biases = {
        // Layer 1 biases (4 neurons)
        {0.1, -0.1, 0.2, -0.2},
        // Layer 2 biases (3 neurons)
        {0.3, -0.2, 0.1},
        // Output layer biases (2 neurons)
        {0.2, -0.1}
    };

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << epoch + 1 << ":" << endl;
        
        auto [updated_weights, updated_biases] = nn.train(input, target, weights, biases);
        
        // Print weights and biases
        print_weights_biases(updated_weights, updated_biases);
        
        // Update for next epoch
        weights = updated_weights;
        biases = updated_biases;
    }

    // After training, perform forward pass with final weights and biases
    cout << "\nFinal Forward Pass Results:" << endl;
    vector<double> outputs = nn.forward_pass(input, weights, biases);
    cout << "Outputs: [";
    for (size_t i = 0; i < outputs.size(); ++i) {
        cout << outputs[i];
        if (i < outputs.size() - 1) cout << ", ";
    }
    cout << "]" << endl;

    return 0;
}