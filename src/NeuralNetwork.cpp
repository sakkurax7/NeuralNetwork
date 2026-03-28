#include "global_neuralnetwork.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace {
std::mt19937 &globalRng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

std::vector<double> matVecMul(const std::vector<std::vector<double>> &m,
                              const std::vector<double> &v,
                              const std::vector<double> &b) {
    std::vector<double> out(m.size(), 0.0);
    for (std::size_t row = 0; row < m.size(); ++row) {
        double sum = b[row];
        for (std::size_t col = 0; col < v.size(); ++col) {
            sum += m[row][col] * v[col];
        }
        out[row] = sum;
    }
    return out;
}
} // namespace

NeuralNetwork::NeuralNetwork(const std::vector<uint> &layerSizes,
                             const std::vector<ActivationType> &activations,
                             double learningRate)
    : lr(learningRate) {
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers");
    }
    if (activations.size() != layerSizes.size() - 1) {
        throw std::invalid_argument("Activation count must equal layer count minus input layer");
    }

    layers.reserve(layerSizes.size() - 1);
    for (std::size_t i = 1; i < layerSizes.size(); ++i) {
        const std::size_t inSize = layerSizes[i - 1];
        const std::size_t outSize = layerSizes[i];

        DenseLayer layer;
        layer.activation = activations[i - 1];
        layer.weights.assign(outSize, std::vector<double>(inSize, 0.0));
        layer.biases.assign(outSize, 0.0);

        const double stdDev =
            (layer.activation == ActivationType::ReLU)
                ? std::sqrt(2.0 / static_cast<double>(inSize))
                : std::sqrt(1.0 / static_cast<double>(inSize));
        std::normal_distribution<double> dist(0.0, stdDev);

        for (auto &row : layer.weights) {
            for (double &w : row) {
                w = dist(globalRng());
            }
        }

        layers.push_back(std::move(layer));
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double> &input) {
    return forwardInternal(input);
}

double NeuralNetwork::trainSample(const std::vector<double> &input,
                                  const std::vector<double> &target) {
    forwardInternal(input);
    return backwardAndUpdate(target);
}

std::uint8_t NeuralNetwork::predictClass(const std::vector<double> &input) {
    const std::vector<double> output = forward(input);
    return static_cast<std::uint8_t>(
        std::distance(output.begin(), std::max_element(output.begin(), output.end())));
}

std::vector<double> NeuralNetwork::forwardInternal(const std::vector<double> &input) {
    std::vector<double> current = input;

    for (DenseLayer &layer : layers) {
        layer.inputCache = current;
        std::vector<double> z = matVecMul(layer.weights, current, layer.biases);

        if (layer.activation == ActivationType::Softmax) {
            layer.outputCache = softmax(z);
        } else {
            layer.outputCache.resize(z.size());
            for (std::size_t i = 0; i < z.size(); ++i) {
                layer.outputCache[i] = activate(z[i], layer.activation);
            }
        }

        current = layer.outputCache;
    }

    return current;
}

double NeuralNetwork::backwardAndUpdate(const std::vector<double> &target) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no trainable layers");
    }

    DenseLayer &outputLayer = layers.back();
    if (target.size() != outputLayer.outputCache.size()) {
        throw std::invalid_argument("Target size does not match output size");
    }

    std::vector<std::vector<double>> deltas(layers.size());

    deltas.back().assign(outputLayer.outputCache.size(), 0.0);
    if (outputLayer.activation == ActivationType::Softmax) {
        for (std::size_t i = 0; i < outputLayer.outputCache.size(); ++i) {
            deltas.back()[i] = outputLayer.outputCache[i] - target[i];
        }
    } else {
        for (std::size_t i = 0; i < outputLayer.outputCache.size(); ++i) {
            const double error = outputLayer.outputCache[i] - target[i];
            deltas.back()[i] =
                error * activationDerivativeFromOutput(outputLayer.outputCache[i], outputLayer.activation);
        }
    }

    for (std::size_t layerIdx = layers.size(); layerIdx-- > 1;) {
        const DenseLayer &current = layers[layerIdx];
        const DenseLayer &previous = layers[layerIdx - 1];

        deltas[layerIdx - 1].assign(previous.outputCache.size(), 0.0);
        for (std::size_t i = 0; i < previous.outputCache.size(); ++i) {
            double weighted = 0.0;
            for (std::size_t j = 0; j < current.weights.size(); ++j) {
                weighted += current.weights[j][i] * deltas[layerIdx][j];
            }
            deltas[layerIdx - 1][i] =
                weighted * activationDerivativeFromOutput(previous.outputCache[i], previous.activation);
        }
    }

    for (std::size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
        DenseLayer &layer = layers[layerIdx];

        for (std::size_t out = 0; out < layer.weights.size(); ++out) {
            for (std::size_t in = 0; in < layer.weights[out].size(); ++in) {
                layer.weights[out][in] -= lr * deltas[layerIdx][out] * layer.inputCache[in];
            }
            layer.biases[out] -= lr * deltas[layerIdx][out];
        }
    }

    double loss = 0.0;
    if (outputLayer.activation == ActivationType::Softmax) {
        const double eps = 1e-12;
        for (std::size_t i = 0; i < target.size(); ++i) {
            loss -= target[i] * std::log(outputLayer.outputCache[i] + eps);
        }
    } else {
        for (std::size_t i = 0; i < target.size(); ++i) {
            const double diff = outputLayer.outputCache[i] - target[i];
            loss += 0.5 * diff * diff;
        }
    }

    return loss;
}

double NeuralNetwork::activate(double x, ActivationType type) {
    switch (type) {
    case ActivationType::Sigmoid:
        return 1.0 / (1.0 + std::exp(-x));
    case ActivationType::Tanh:
        return std::tanh(x);
    case ActivationType::ReLU:
        return (x > 0.0) ? x : 0.0;
    case ActivationType::Softmax:
        throw std::invalid_argument("Softmax is vector-valued and handled at layer level");
    }
    throw std::invalid_argument("Unsupported activation type");
}

double NeuralNetwork::activationDerivativeFromOutput(double activatedValue,
                                                     ActivationType type) {
    switch (type) {
    case ActivationType::Sigmoid:
        return activatedValue * (1.0 - activatedValue);
    case ActivationType::Tanh:
        return 1.0 - activatedValue * activatedValue;
    case ActivationType::ReLU:
        return (activatedValue > 0.0) ? 1.0 : 0.0;
    case ActivationType::Softmax:
        throw std::invalid_argument("Softmax derivative is handled with cross-entropy shortcut");
    }
    throw std::invalid_argument("Unsupported activation type");
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double> &z) {
    if (z.empty()) {
        return {};
    }

    const double maxLogit = *std::max_element(z.begin(), z.end());
    std::vector<double> exps(z.size(), 0.0);
    double sum = 0.0;

    for (std::size_t i = 0; i < z.size(); ++i) {
        exps[i] = std::exp(z[i] - maxLogit);
        sum += exps[i];
    }

    for (double &v : exps) {
        v /= sum;
    }

    return exps;
}
