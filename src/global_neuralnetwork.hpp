#ifndef GLOBAL_NEURALNETWORK_HPP
#define GLOBAL_NEURALNETWORK_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

typedef unsigned int uint;

enum class ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax
};

struct MnistSample {
    std::vector<double> pixels;
    std::uint8_t label;
};

class MnistDataset {
  public:
    static MnistDataset load(const std::string &imagesPath,
                             const std::string &labelsPath,
                             std::size_t limit = 0);
    static MnistDataset loadImagesOnly(const std::string &imagesPath,
                                       std::size_t limit = 0);

    std::size_t size() const;
    const MnistSample &operator[](std::size_t index) const;
    std::vector<double> oneHotLabel(std::size_t index, std::size_t classes) const;

  private:
    std::vector<MnistSample> samples;
};

class NeuralNetwork {
  public:
    NeuralNetwork(const std::vector<uint> &layerSizes,
                  const std::vector<ActivationType> &activations,
                  double learningRate);

    std::vector<double> forward(const std::vector<double> &input);
    double trainSample(const std::vector<double> &input,
                       const std::vector<double> &target);
    std::uint8_t predictClass(const std::vector<double> &input);
    void saveModel(const std::string &modelPath) const;
    static NeuralNetwork loadModel(const std::string &modelPath, double learningRate);

  private:
    struct DenseLayer {
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        ActivationType activation;

        std::vector<double> inputCache;
        std::vector<double> outputCache;
    };

    std::vector<DenseLayer> layers;
    double lr;
    std::vector<uint> layerShape;
    std::vector<ActivationType> layerActivations;

    static double activate(double x, ActivationType type);
    static double activationDerivativeFromOutput(double activatedValue,
                                                 ActivationType type);
    static std::vector<double> softmax(const std::vector<double> &z);
    static std::string activationToString(ActivationType type);
    static ActivationType activationFromString(const std::string &name);

    std::vector<double> forwardInternal(const std::vector<double> &input);
    double backwardAndUpdate(const std::vector<double> &target);
};

#endif
