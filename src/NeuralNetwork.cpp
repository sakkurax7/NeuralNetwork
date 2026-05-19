#include "global_neuralnetwork.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
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
    : lr(learningRate), layerShape(layerSizes), layerActivations(activations) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be greater than zero");
    }
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers");
    }
    if (activations.size() != layerSizes.size() - 1) {
        throw std::invalid_argument("Activation count must equal layer count minus input layer");
    }
    for (std::size_t i = 0; i < layerSizes.size(); ++i) {
        if (layerSizes[i] == 0) {
            throw std::invalid_argument("Layer size must be greater than zero");
        }
    }
    for (std::size_t i = 0; i + 1 < activations.size(); ++i) {
        if (activations[i] == ActivationType::Softmax) {
            throw std::invalid_argument("Softmax is supported only on the output layer");
        }
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

std::vector<double> NeuralNetwork::run(const std::vector<double> &input) {
    return forwardInternal(input);
}

std::vector<double> NeuralNetwork::forward(const std::vector<double> &input) {
    return forwardInternal(input);
}

double NeuralNetwork::trainSample(const std::vector<double> &input,
                                  const std::vector<double> &target) {
    forwardInternal(input);
    return backwardAndUpdate(target);
}

double NeuralNetwork::train(const std::vector<std::vector<double>> &inputs,
                            const std::vector<std::vector<double>> &targets,
                            std::size_t epochs,
                            bool shuffleEachEpoch,
                            std::uint32_t seed) {
    if (inputs.empty()) {
        throw std::invalid_argument("Training inputs cannot be empty");
    }
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Input and target batch sizes must match");
    }
    if (epochs == 0) {
        throw std::invalid_argument("Epoch count must be greater than zero");
    }

    std::vector<std::size_t> order(inputs.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);

    double averageLoss = 0.0;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffleEachEpoch) {
            std::shuffle(order.begin(), order.end(), rng);
        }

        double epochLoss = 0.0;
        for (std::size_t idx : order) {
            epochLoss += trainSample(inputs[idx], targets[idx]);
        }
        averageLoss = epochLoss / static_cast<double>(inputs.size());
    }

    return averageLoss;
}

std::uint8_t NeuralNetwork::predictClass(const std::vector<double> &input) {
    const std::vector<double> output = forward(input);
    return static_cast<std::uint8_t>(
        std::distance(output.begin(), std::max_element(output.begin(), output.end())));
}

std::vector<std::uint8_t>
NeuralNetwork::predictClasses(const std::vector<std::vector<double>> &inputs) {
    std::vector<std::uint8_t> predictions;
    predictions.reserve(inputs.size());
    for (const auto &input : inputs) {
        predictions.push_back(predictClass(input));
    }
    return predictions;
}

void NeuralNetwork::setLearningRate(double learningRate) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be greater than zero");
    }
    lr = learningRate;
}

double NeuralNetwork::getLearningRate() const {
    return lr;
}

const std::vector<uint> &NeuralNetwork::getTopology() const {
    return layerShape;
}

const std::vector<ActivationType> &NeuralNetwork::getActivations() const {
    return layerActivations;
}

void NeuralNetwork::saveModel(const std::string &modelPath) const {
    std::ofstream out(modelPath);
    if (!out) {
        throw std::runtime_error("Unable to open model file for writing: " + modelPath);
    }

    out << "NN_MODEL_V1\n";
    out << "shape " << layerShape.size() << "\n";
    for (uint size : layerShape) {
        out << size << " ";
    }
    out << "\n";

    out << "activations " << layerActivations.size() << "\n";
    for (ActivationType type : layerActivations) {
        out << activationToString(type) << " ";
    }
    out << "\n";

    out << std::setprecision(17);
    out << "learning_rate " << lr << "\n";
    out << "layer_count " << layers.size() << "\n";

    for (std::size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
        const DenseLayer &layer = layers[layerIdx];
        const std::size_t outSize = layer.weights.size();
        const std::size_t inSize = outSize == 0 ? 0 : layer.weights[0].size();

        out << "layer " << layerIdx << " " << outSize << " " << inSize << "\n";
        for (const auto &row : layer.weights) {
            for (double weight : row) {
                out << weight << " ";
            }
            out << "\n";
        }
        out << "biases ";
        for (double bias : layer.biases) {
            out << bias << " ";
        }
        out << "\n";
    }
}

NeuralNetwork NeuralNetwork::loadModel(const std::string &modelPath, double learningRate) {
    std::ifstream in(modelPath);
    if (!in) {
        throw std::runtime_error("Unable to open model file: " + modelPath);
    }

    std::string magic;
    in >> magic;
    if (magic != "NN_MODEL_V1") {
        throw std::runtime_error("Unsupported model file format");
    }

    std::string token;
    std::size_t shapeCount = 0;
    in >> token >> shapeCount;
    if (token != "shape" || shapeCount < 2) {
        throw std::runtime_error("Invalid model shape header");
    }

    std::vector<uint> shape(shapeCount);
    for (std::size_t i = 0; i < shapeCount; ++i) {
        in >> shape[i];
    }

    std::size_t activationCount = 0;
    in >> token >> activationCount;
    if (token != "activations" || activationCount != shapeCount - 1) {
        throw std::runtime_error("Invalid model activations header");
    }

    std::vector<ActivationType> activations;
    activations.reserve(activationCount);
    for (std::size_t i = 0; i < activationCount; ++i) {
        std::string activationName;
        in >> activationName;
        activations.push_back(activationFromString(activationName));
    }

    double storedLearningRate = 0.0;
    in >> token >> storedLearningRate;
    if (token != "learning_rate") {
        throw std::runtime_error("Invalid model learning rate header");
    }
    if (storedLearningRate <= 0.0) {
        throw std::runtime_error("Invalid stored learning rate in model");
    }

    std::size_t layerCount = 0;
    in >> token >> layerCount;
    if (token != "layer_count" || layerCount != activationCount) {
        throw std::runtime_error("Invalid model layer count");
    }

    const double effectiveLearningRate = (learningRate > 0.0) ? learningRate : storedLearningRate;
    NeuralNetwork net(shape, activations, effectiveLearningRate);

    for (std::size_t layerIdx = 0; layerIdx < layerCount; ++layerIdx) {
        std::size_t parsedLayerIdx = 0;
        std::size_t outSize = 0;
        std::size_t inSize = 0;

        in >> token >> parsedLayerIdx >> outSize >> inSize;
        if (token != "layer" || parsedLayerIdx != layerIdx) {
            throw std::runtime_error("Invalid layer block header");
        }

        DenseLayer &layer = net.layers[layerIdx];
        if (layer.weights.size() != outSize ||
            (!layer.weights.empty() && layer.weights[0].size() != inSize)) {
            throw std::runtime_error("Layer dimensions do not match model shape");
        }

        for (std::size_t r = 0; r < outSize; ++r) {
            for (std::size_t c = 0; c < inSize; ++c) {
                in >> layer.weights[r][c];
            }
        }

        in >> token;
        if (token != "biases") {
            throw std::runtime_error("Expected biases block");
        }
        for (std::size_t b = 0; b < outSize; ++b) {
            in >> layer.biases[b];
        }
    }

    if (!in) {
        throw std::runtime_error("Model file ended unexpectedly");
    }

    return net;
}

std::vector<double> NeuralNetwork::forwardInternal(const std::vector<double> &input) {
    if (input.size() != static_cast<std::size_t>(layerShape.front())) {
        throw std::invalid_argument("Input size does not match network input layer");
    }

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

std::string NeuralNetwork::activationToString(ActivationType type) {
    switch (type) {
    case ActivationType::Sigmoid:
        return "sigmoid";
    case ActivationType::Tanh:
        return "tanh";
    case ActivationType::ReLU:
        return "relu";
    case ActivationType::Softmax:
        return "softmax";
    }
    throw std::invalid_argument("Unsupported activation type");
}

ActivationType NeuralNetwork::activationFromString(const std::string &name) {
    std::string normalized = name;
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (normalized == "sigmoid") {
        return ActivationType::Sigmoid;
    }
    if (normalized == "tanh") {
        return ActivationType::Tanh;
    }
    if (normalized == "relu") {
        return ActivationType::ReLU;
    }
    if (normalized == "softmax") {
        return ActivationType::Softmax;
    }
    throw std::invalid_argument("Unknown activation name: " + name);
}

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(std::size_t inputRows,
                                                       std::size_t inputCols,
                                                       std::size_t filterCount,
                                                       std::size_t kernelSize,
                                                       std::size_t hiddenUnits,
                                                       std::size_t outputClasses,
                                                       double learningRate)
    : inputRows(inputRows),
      inputCols(inputCols),
      filters(filterCount),
      kernel(kernelSize),
      hidden(hiddenUnits),
      classes(outputClasses),
      convRows(0),
      convCols(0),
      poolRows(0),
      poolCols(0),
      lr(learningRate) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be greater than zero");
    }
    if (inputRows == 0 || inputCols == 0) {
        throw std::invalid_argument("Input dimensions must be greater than zero");
    }
    if (filters == 0) {
        throw std::invalid_argument("Filter count must be greater than zero");
    }
    if (kernel == 0) {
        throw std::invalid_argument("Kernel size must be greater than zero");
    }
    if (hidden == 0) {
        throw std::invalid_argument("Hidden layer size must be greater than zero");
    }
    if (classes == 0) {
        throw std::invalid_argument("Output classes must be greater than zero");
    }
    if (kernel > inputRows || kernel > inputCols) {
        throw std::invalid_argument("Kernel size must fit inside input dimensions");
    }

    convRows = inputRows - kernel + 1;
    convCols = inputCols - kernel + 1;
    poolRows = convRows / 2;
    poolCols = convCols / 2;

    if (poolRows == 0 || poolCols == 0) {
        throw std::invalid_argument(
            "Convolution output is too small for 2x2 max-pooling; adjust kernel/input dimensions");
    }

    const std::size_t kernelArea = kernel * kernel;
    const std::size_t convArea = convRows * convCols;
    const std::size_t poolArea = poolRows * poolCols;
    const std::size_t flattenSize = filters * poolArea;

    convKernels.assign(filters, std::vector<double>(kernelArea, 0.0));
    convBiases.assign(filters, 0.0);
    dense1Weights.assign(hidden, std::vector<double>(flattenSize, 0.0));
    dense1Biases.assign(hidden, 0.0);
    dense2Weights.assign(classes, std::vector<double>(hidden, 0.0));
    dense2Biases.assign(classes, 0.0);

    std::normal_distribution<double> convDist(0.0,
                                              std::sqrt(2.0 / static_cast<double>(kernelArea)));
    for (auto &kernelWeights : convKernels) {
        for (double &weight : kernelWeights) {
            weight = convDist(globalRng());
        }
    }

    std::normal_distribution<double> dense1Dist(
        0.0, std::sqrt(2.0 / static_cast<double>(flattenSize)));
    for (auto &row : dense1Weights) {
        for (double &weight : row) {
            weight = dense1Dist(globalRng());
        }
    }

    std::normal_distribution<double> dense2Dist(0.0,
                                                std::sqrt(1.0 / static_cast<double>(hidden)));
    for (auto &row : dense2Weights) {
        for (double &weight : row) {
            weight = dense2Dist(globalRng());
        }
    }

    convLinearCache.assign(filters, std::vector<double>(convArea, 0.0));
    convActivatedCache.assign(filters, std::vector<double>(convArea, 0.0));
    pooledCache.assign(filters, std::vector<double>(poolArea, 0.0));
    poolIndexCache.assign(filters, std::vector<std::size_t>(poolArea, 0));
    flattenedCache.assign(flattenSize, 0.0);
    hiddenLinearCache.assign(hidden, 0.0);
    hiddenActivatedCache.assign(hidden, 0.0);
    outputCache.assign(classes, 0.0);
}

std::vector<double> ConvolutionalNeuralNetwork::run(const std::vector<double> &input) {
    return forwardInternal(input);
}

std::vector<double> ConvolutionalNeuralNetwork::forward(const std::vector<double> &input) {
    return forwardInternal(input);
}

double ConvolutionalNeuralNetwork::trainSample(const std::vector<double> &input,
                                               const std::vector<double> &target) {
    forwardInternal(input);
    return backwardAndUpdate(target);
}

std::uint8_t
ConvolutionalNeuralNetwork::predictClass(const std::vector<double> &input) {
    const std::vector<double> output = forward(input);
    return static_cast<std::uint8_t>(
        std::distance(output.begin(), std::max_element(output.begin(), output.end())));
}

void ConvolutionalNeuralNetwork::setLearningRate(double learningRate) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be greater than zero");
    }
    lr = learningRate;
}

double ConvolutionalNeuralNetwork::getLearningRate() const {
    return lr;
}

std::size_t ConvolutionalNeuralNetwork::getOutputClasses() const {
    return classes;
}

std::size_t ConvolutionalNeuralNetwork::getInputSize() const {
    return inputRows * inputCols;
}

std::string ConvolutionalNeuralNetwork::architectureSummary() const {
    std::ostringstream out;
    out << "Conv(filters=" << filters << ", kernel=" << kernel << "x" << kernel << ") -> "
        << "MaxPool(2x2) -> Dense(" << hidden << ") -> Dense(" << classes << ")";
    return out.str();
}

void ConvolutionalNeuralNetwork::saveModel(const std::string &modelPath) const {
    std::ofstream out(modelPath);
    if (!out) {
        throw std::runtime_error("Unable to open model file for writing: " + modelPath);
    }

    out << "CNN_MODEL_V1\n";
    out << "input_rows " << inputRows << "\n";
    out << "input_cols " << inputCols << "\n";
    out << "filters " << filters << "\n";
    out << "kernel_size " << kernel << "\n";
    out << "hidden_units " << hidden << "\n";
    out << "output_classes " << classes << "\n";
    out << std::setprecision(17);
    out << "learning_rate " << lr << "\n";

    out << "conv_kernels " << convKernels.size() << " "
        << (convKernels.empty() ? 0 : convKernels[0].size()) << "\n";
    for (const auto &kernelWeights : convKernels) {
        for (double weight : kernelWeights) {
            out << weight << " ";
        }
        out << "\n";
    }

    out << "conv_biases " << convBiases.size() << "\n";
    for (double bias : convBiases) {
        out << bias << " ";
    }
    out << "\n";

    out << "dense1_weights " << dense1Weights.size() << " "
        << (dense1Weights.empty() ? 0 : dense1Weights[0].size()) << "\n";
    for (const auto &row : dense1Weights) {
        for (double weight : row) {
            out << weight << " ";
        }
        out << "\n";
    }

    out << "dense1_biases " << dense1Biases.size() << "\n";
    for (double bias : dense1Biases) {
        out << bias << " ";
    }
    out << "\n";

    out << "dense2_weights " << dense2Weights.size() << " "
        << (dense2Weights.empty() ? 0 : dense2Weights[0].size()) << "\n";
    for (const auto &row : dense2Weights) {
        for (double weight : row) {
            out << weight << " ";
        }
        out << "\n";
    }

    out << "dense2_biases " << dense2Biases.size() << "\n";
    for (double bias : dense2Biases) {
        out << bias << " ";
    }
    out << "\n";
}

ConvolutionalNeuralNetwork
ConvolutionalNeuralNetwork::loadModel(const std::string &modelPath,
                                      double learningRate) {
    std::ifstream in(modelPath);
    if (!in) {
        throw std::runtime_error("Unable to open model file: " + modelPath);
    }

    std::string magic;
    in >> magic;
    if (magic != "CNN_MODEL_V1") {
        throw std::runtime_error("Unsupported CNN model file format");
    }

    std::string token;
    std::size_t fileInputRows = 0;
    std::size_t fileInputCols = 0;
    std::size_t fileFilters = 0;
    std::size_t fileKernel = 0;
    std::size_t fileHidden = 0;
    std::size_t fileClasses = 0;
    double storedLearningRate = 0.0;

    in >> token >> fileInputRows;
    if (token != "input_rows") {
        throw std::runtime_error("Invalid CNN model header: expected input_rows");
    }
    in >> token >> fileInputCols;
    if (token != "input_cols") {
        throw std::runtime_error("Invalid CNN model header: expected input_cols");
    }
    in >> token >> fileFilters;
    if (token != "filters") {
        throw std::runtime_error("Invalid CNN model header: expected filters");
    }
    in >> token >> fileKernel;
    if (token != "kernel_size") {
        throw std::runtime_error("Invalid CNN model header: expected kernel_size");
    }
    in >> token >> fileHidden;
    if (token != "hidden_units") {
        throw std::runtime_error("Invalid CNN model header: expected hidden_units");
    }
    in >> token >> fileClasses;
    if (token != "output_classes") {
        throw std::runtime_error("Invalid CNN model header: expected output_classes");
    }
    in >> token >> storedLearningRate;
    if (token != "learning_rate") {
        throw std::runtime_error("Invalid CNN model header: expected learning_rate");
    }
    if (storedLearningRate <= 0.0) {
        throw std::runtime_error("Invalid stored learning rate in CNN model");
    }

    const double effectiveLearningRate = (learningRate > 0.0) ? learningRate : storedLearningRate;
    ConvolutionalNeuralNetwork net(fileInputRows,
                                   fileInputCols,
                                   fileFilters,
                                   fileKernel,
                                   fileHidden,
                                   fileClasses,
                                   effectiveLearningRate);

    std::size_t rowCount = 0;
    std::size_t colCount = 0;

    in >> token >> rowCount >> colCount;
    if (token != "conv_kernels") {
        throw std::runtime_error("Expected conv_kernels block");
    }
    if (rowCount != net.convKernels.size() ||
        (!net.convKernels.empty() && colCount != net.convKernels[0].size())) {
        throw std::runtime_error("CNN conv_kernels dimensions do not match architecture");
    }
    for (std::size_t r = 0; r < rowCount; ++r) {
        for (std::size_t c = 0; c < colCount; ++c) {
            in >> net.convKernels[r][c];
        }
    }

    std::size_t biasCount = 0;
    in >> token >> biasCount;
    if (token != "conv_biases" || biasCount != net.convBiases.size()) {
        throw std::runtime_error("CNN conv_biases block is invalid");
    }
    for (std::size_t i = 0; i < biasCount; ++i) {
        in >> net.convBiases[i];
    }

    in >> token >> rowCount >> colCount;
    if (token != "dense1_weights") {
        throw std::runtime_error("Expected dense1_weights block");
    }
    if (rowCount != net.dense1Weights.size() ||
        (!net.dense1Weights.empty() && colCount != net.dense1Weights[0].size())) {
        throw std::runtime_error("CNN dense1_weights dimensions do not match architecture");
    }
    for (std::size_t r = 0; r < rowCount; ++r) {
        for (std::size_t c = 0; c < colCount; ++c) {
            in >> net.dense1Weights[r][c];
        }
    }

    in >> token >> biasCount;
    if (token != "dense1_biases" || biasCount != net.dense1Biases.size()) {
        throw std::runtime_error("CNN dense1_biases block is invalid");
    }
    for (std::size_t i = 0; i < biasCount; ++i) {
        in >> net.dense1Biases[i];
    }

    in >> token >> rowCount >> colCount;
    if (token != "dense2_weights") {
        throw std::runtime_error("Expected dense2_weights block");
    }
    if (rowCount != net.dense2Weights.size() ||
        (!net.dense2Weights.empty() && colCount != net.dense2Weights[0].size())) {
        throw std::runtime_error("CNN dense2_weights dimensions do not match architecture");
    }
    for (std::size_t r = 0; r < rowCount; ++r) {
        for (std::size_t c = 0; c < colCount; ++c) {
            in >> net.dense2Weights[r][c];
        }
    }

    in >> token >> biasCount;
    if (token != "dense2_biases" || biasCount != net.dense2Biases.size()) {
        throw std::runtime_error("CNN dense2_biases block is invalid");
    }
    for (std::size_t i = 0; i < biasCount; ++i) {
        in >> net.dense2Biases[i];
    }

    if (!in) {
        throw std::runtime_error("CNN model file ended unexpectedly");
    }

    return net;
}

std::vector<double>
ConvolutionalNeuralNetwork::forwardInternal(const std::vector<double> &input) {
    if (input.size() != getInputSize()) {
        throw std::invalid_argument("Input size does not match CNN input dimensions");
    }

    inputCache = input;
    const std::size_t poolArea = poolRows * poolCols;

    for (std::size_t filterIdx = 0; filterIdx < filters; ++filterIdx) {
        const std::vector<double> &kernelWeights = convKernels[filterIdx];
        std::vector<double> &convLinear = convLinearCache[filterIdx];
        std::vector<double> &convActivated = convActivatedCache[filterIdx];
        std::vector<double> &pooled = pooledCache[filterIdx];
        std::vector<std::size_t> &poolIndex = poolIndexCache[filterIdx];

        for (std::size_t r = 0; r < convRows; ++r) {
            for (std::size_t c = 0; c < convCols; ++c) {
                double sum = convBiases[filterIdx];
                for (std::size_t kr = 0; kr < kernel; ++kr) {
                    for (std::size_t kc = 0; kc < kernel; ++kc) {
                        const std::size_t inputRow = r + kr;
                        const std::size_t inputCol = c + kc;
                        const std::size_t inputIdx = inputRow * inputCols + inputCol;
                        const std::size_t kernelIdx = kr * kernel + kc;
                        sum += kernelWeights[kernelIdx] * input[inputIdx];
                    }
                }

                const std::size_t convIdx = r * convCols + c;
                convLinear[convIdx] = sum;
                convActivated[convIdx] = (sum > 0.0) ? sum : 0.0;
            }
        }

        for (std::size_t pr = 0; pr < poolRows; ++pr) {
            for (std::size_t pc = 0; pc < poolCols; ++pc) {
                const std::size_t baseRow = pr * 2;
                const std::size_t baseCol = pc * 2;
                double maxValue = std::numeric_limits<double>::lowest();
                std::size_t maxIndex = 0;

                for (std::size_t dr = 0; dr < 2; ++dr) {
                    for (std::size_t dc = 0; dc < 2; ++dc) {
                        const std::size_t convRow = baseRow + dr;
                        const std::size_t convCol = baseCol + dc;
                        const std::size_t convIdx = convRow * convCols + convCol;
                        const double candidate = convActivated[convIdx];
                        if (candidate > maxValue) {
                            maxValue = candidate;
                            maxIndex = convIdx;
                        }
                    }
                }

                const std::size_t poolIdx = pr * poolCols + pc;
                pooled[poolIdx] = maxValue;
                poolIndex[poolIdx] = maxIndex;
            }
        }
    }

    for (std::size_t filterIdx = 0; filterIdx < filters; ++filterIdx) {
        const std::size_t offset = filterIdx * poolArea;
        for (std::size_t i = 0; i < poolArea; ++i) {
            flattenedCache[offset + i] = pooledCache[filterIdx][i];
        }
    }

    hiddenLinearCache = matVecMul(dense1Weights, flattenedCache, dense1Biases);
    for (std::size_t i = 0; i < hiddenLinearCache.size(); ++i) {
        hiddenActivatedCache[i] = (hiddenLinearCache[i] > 0.0) ? hiddenLinearCache[i] : 0.0;
    }

    const std::vector<double> logits = matVecMul(dense2Weights, hiddenActivatedCache, dense2Biases);
    outputCache = softmax(logits);

    return outputCache;
}

double ConvolutionalNeuralNetwork::backwardAndUpdate(const std::vector<double> &target) {
    if (target.size() != classes) {
        throw std::invalid_argument("Target size does not match CNN output classes");
    }

    const std::size_t flattenSize = flattenedCache.size();
    const std::size_t convArea = convRows * convCols;
    const std::size_t poolArea = poolRows * poolCols;

    std::vector<double> deltaOutput(classes, 0.0);
    for (std::size_t i = 0; i < classes; ++i) {
        deltaOutput[i] = outputCache[i] - target[i];
    }

    std::vector<double> deltaHidden(hidden, 0.0);
    for (std::size_t h = 0; h < hidden; ++h) {
        double weighted = 0.0;
        for (std::size_t o = 0; o < classes; ++o) {
            weighted += dense2Weights[o][h] * deltaOutput[o];
        }
        deltaHidden[h] = (hiddenLinearCache[h] > 0.0) ? weighted : 0.0;
    }

    std::vector<double> deltaFlatten(flattenSize, 0.0);
    for (std::size_t i = 0; i < flattenSize; ++i) {
        double weighted = 0.0;
        for (std::size_t h = 0; h < hidden; ++h) {
            weighted += dense1Weights[h][i] * deltaHidden[h];
        }
        deltaFlatten[i] = weighted;
    }

    std::vector<std::vector<double>> deltaConvLinear(
        filters, std::vector<double>(convArea, 0.0));
    for (std::size_t filterIdx = 0; filterIdx < filters; ++filterIdx) {
        const std::size_t offset = filterIdx * poolArea;
        for (std::size_t i = 0; i < poolArea; ++i) {
            const std::size_t convIdx = poolIndexCache[filterIdx][i];
            deltaConvLinear[filterIdx][convIdx] += deltaFlatten[offset + i];
        }
    }

    for (std::size_t filterIdx = 0; filterIdx < filters; ++filterIdx) {
        for (std::size_t i = 0; i < convArea; ++i) {
            if (convLinearCache[filterIdx][i] <= 0.0) {
                deltaConvLinear[filterIdx][i] = 0.0;
            }
        }
    }

    for (std::size_t out = 0; out < classes; ++out) {
        for (std::size_t in = 0; in < hidden; ++in) {
            dense2Weights[out][in] -= lr * deltaOutput[out] * hiddenActivatedCache[in];
        }
        dense2Biases[out] -= lr * deltaOutput[out];
    }

    for (std::size_t out = 0; out < hidden; ++out) {
        for (std::size_t in = 0; in < flattenSize; ++in) {
            dense1Weights[out][in] -= lr * deltaHidden[out] * flattenedCache[in];
        }
        dense1Biases[out] -= lr * deltaHidden[out];
    }

    for (std::size_t filterIdx = 0; filterIdx < filters; ++filterIdx) {
        for (std::size_t kr = 0; kr < kernel; ++kr) {
            for (std::size_t kc = 0; kc < kernel; ++kc) {
                double grad = 0.0;
                for (std::size_t r = 0; r < convRows; ++r) {
                    for (std::size_t c = 0; c < convCols; ++c) {
                        const std::size_t convIdx = r * convCols + c;
                        const std::size_t inputRow = r + kr;
                        const std::size_t inputCol = c + kc;
                        const std::size_t inputIdx = inputRow * inputCols + inputCol;
                        grad += deltaConvLinear[filterIdx][convIdx] * inputCache[inputIdx];
                    }
                }

                const std::size_t kernelIdx = kr * kernel + kc;
                convKernels[filterIdx][kernelIdx] -= lr * grad;
            }
        }

        double biasGrad = 0.0;
        for (double delta : deltaConvLinear[filterIdx]) {
            biasGrad += delta;
        }
        convBiases[filterIdx] -= lr * biasGrad;
    }

    double loss = 0.0;
    const double eps = 1e-12;
    for (std::size_t i = 0; i < classes; ++i) {
        loss -= target[i] * std::log(outputCache[i] + eps);
    }

    return loss;
}

std::vector<double>
ConvolutionalNeuralNetwork::softmax(const std::vector<double> &z) {
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

    for (double &value : exps) {
        value /= sum;
    }

    return exps;
}
