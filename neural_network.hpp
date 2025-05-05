#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <utility>

enum class Activation {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR,
    SOFTMAX
};

class NeuralNetwork {
private:
    int num_hidden_layers;
    std::vector<int> neurons_per_hidden;
    int output_neurons;
    double learning_rate;
    std::vector<Activation> activation_functions;

    std::vector<double> apply_activation(const std::vector<double>& z, const Activation& activation);
    std::vector<double> apply_activation_derivative(const std::vector<double>& z, const Activation& activation);
    std::vector<double> subtract_vectors(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> hadamard_product(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> softmax(const std::vector<double>& z);

public:
    NeuralNetwork(int num_hidden, const std::vector<int>& neurons_hidden, int output_size,
                  double lr, const std::vector<Activation>& activations);

    std::vector<double> forward_pass(const std::vector<double>& input,
                                     const std::vector<std::vector<std::vector<double>>>& weights,
                                     const std::vector<std::vector<double>>& biases);

    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> train(
        const std::vector<double>& input,
        const std::vector<double>& target,
        const std::vector<std::vector<std::vector<double>>>& weights,
        const std::vector<std::vector<double>>& biases);
    
    double cross_entropy_loss(const std::vector<double>& pred, const std::vector<double>& target);
};

#endif // NEURAL_NETWORK_HPP