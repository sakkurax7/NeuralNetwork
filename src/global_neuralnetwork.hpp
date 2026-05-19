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

    std::vector<double> run(const std::vector<double> &input);
    std::vector<double> forward(const std::vector<double> &input);
    double trainSample(const std::vector<double> &input,
                       const std::vector<double> &target);
    double train(const std::vector<std::vector<double>> &inputs,
                 const std::vector<std::vector<double>> &targets,
                 std::size_t epochs = 1,
                 bool shuffleEachEpoch = true,
                 std::uint32_t seed = 42);
    std::uint8_t predictClass(const std::vector<double> &input);
    std::vector<std::uint8_t> predictClasses(const std::vector<std::vector<double>> &inputs);
    void setLearningRate(double learningRate);
    double getLearningRate() const;
    const std::vector<uint> &getTopology() const;
    const std::vector<ActivationType> &getActivations() const;
    void saveModel(const std::string &modelPath) const;
    static NeuralNetwork loadModel(const std::string &modelPath, double learningRate);
    static std::string activationToString(ActivationType type);
    static ActivationType activationFromString(const std::string &name);

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

    std::vector<double> forwardInternal(const std::vector<double> &input);
    double backwardAndUpdate(const std::vector<double> &target);
};

class ConvolutionalNeuralNetwork {
  public:
    ConvolutionalNeuralNetwork(std::size_t inputRows,
                               std::size_t inputCols,
                               std::size_t filterCount,
                               std::size_t kernelSize,
                               std::size_t hiddenUnits,
                               std::size_t outputClasses,
                               double learningRate);

    std::vector<double> run(const std::vector<double> &input);
    std::vector<double> forward(const std::vector<double> &input);
    double trainSample(const std::vector<double> &input,
                       const std::vector<double> &target);
    std::uint8_t predictClass(const std::vector<double> &input);
    void setLearningRate(double learningRate);
    double getLearningRate() const;
    std::size_t getOutputClasses() const;
    std::size_t getInputSize() const;
    std::string architectureSummary() const;
    void saveModel(const std::string &modelPath) const;
    static ConvolutionalNeuralNetwork loadModel(const std::string &modelPath,
                                                double learningRate);

  private:
    std::size_t inputRows;
    std::size_t inputCols;
    std::size_t filters;
    std::size_t kernel;
    std::size_t hidden;
    std::size_t classes;
    std::size_t convRows;
    std::size_t convCols;
    std::size_t poolRows;
    std::size_t poolCols;
    double lr;

    std::vector<std::vector<double>> convKernels;
    std::vector<double> convBiases;
    std::vector<std::vector<double>> dense1Weights;
    std::vector<double> dense1Biases;
    std::vector<std::vector<double>> dense2Weights;
    std::vector<double> dense2Biases;

    std::vector<double> inputCache;
    std::vector<std::vector<double>> convLinearCache;
    std::vector<std::vector<double>> convActivatedCache;
    std::vector<std::vector<double>> pooledCache;
    std::vector<std::vector<std::size_t>> poolIndexCache;
    std::vector<double> flattenedCache;
    std::vector<double> hiddenLinearCache;
    std::vector<double> hiddenActivatedCache;
    std::vector<double> outputCache;

    std::vector<double> forwardInternal(const std::vector<double> &input);
    double backwardAndUpdate(const std::vector<double> &target);
    static std::vector<double> softmax(const std::vector<double> &z);
};

#endif
