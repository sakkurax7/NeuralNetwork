#include "global_neuralnetwork.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct TrainConfig {
    std::string imagesPath = "images";
    std::string labelsPath = "labels";
    std::size_t sampleLimit = 0;
    std::size_t epochs = 3;
    double learningRate = 0.01;
};

TrainConfig parseArgs(int argc, char **argv) {
    TrainConfig cfg;
    if (argc > 1) {
        cfg.imagesPath = argv[1];
    }
    if (argc > 2) {
        cfg.labelsPath = argv[2];
    }
    if (argc > 3) {
        cfg.epochs = static_cast<std::size_t>(std::stoul(argv[3]));
    }
    if (argc > 4) {
        cfg.learningRate = std::stod(argv[4]);
    }
    if (argc > 5) {
        cfg.sampleLimit = static_cast<std::size_t>(std::stoul(argv[5]));
    }
    return cfg;
}

double evaluateAccuracy(NeuralNetwork &net, const MnistDataset &dataset) {
    if (dataset.size() == 0) {
        return 0.0;
    }

    std::size_t correct = 0;
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const MnistSample &sample = dataset[i];
        if (net.predictClass(sample.pixels) == sample.label) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}
} // namespace

int main(int argc, char **argv) {
    try {
        const TrainConfig cfg = parseArgs(argc, argv);

        std::cout << "Loading MNIST data..." << std::endl;
        const MnistDataset dataset =
            MnistDataset::load(cfg.imagesPath, cfg.labelsPath, cfg.sampleLimit);

        std::cout << "Loaded " << dataset.size() << " samples from:"
                  << " images='" << cfg.imagesPath << "'"
                  << " labels='" << cfg.labelsPath << "'" << std::endl;

        const std::vector<uint> topology = {784, 128, 64, 10};
        const std::vector<ActivationType> activations = {
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::Softmax,
        };

        NeuralNetwork net(topology, activations, cfg.learningRate);

        std::vector<std::size_t> order(dataset.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(std::random_device{}());

        for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), rng);

            double epochLoss = 0.0;
            for (std::size_t idx : order) {
                const MnistSample &sample = dataset[idx];
                const std::vector<double> target = dataset.oneHotLabel(idx, 10);
                epochLoss += net.trainSample(sample.pixels, target);
            }

            epochLoss /= static_cast<double>(dataset.size());
            const double acc = evaluateAccuracy(net, dataset);

            std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs
                      << " - loss: " << epochLoss
                      << " - accuracy: " << (acc * 100.0) << "%"
                      << std::endl;
        }

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
