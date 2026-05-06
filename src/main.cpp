#include "global_neuralnetwork.hpp"

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
volatile std::sig_atomic_t g_stopRequested = 0;

void handleSignal(int) {
    g_stopRequested = 1;
}

struct TrainConfig {
    std::string imagesPath = "images";
    std::string labelsPath = "labels";
    std::string modelOutputPath = "model.nn";
    std::size_t sampleLimit = 0;
    std::size_t epochs = 5;
    double learningRate = 0.01;
    double validationSplit = 0.1;
    double targetValAccuracy = 1.0;
    std::size_t earlyStopPatience = 0;
    std::size_t checkpointEvery = 1;
    std::uint32_t seed = 42;
    std::string resumeModelPath = "";
    std::vector<uint> topology = {784, 128, 64, 10};
    std::vector<ActivationType> activations = {
        ActivationType::ReLU,
        ActivationType::ReLU,
        ActivationType::Softmax,
    };
    bool hasTopologyOverride = false;
    bool hasActivationOverride = false;
};

struct PredictConfig {
    std::string modelPath = "model.nn";
    std::string imagesPath = "images";
    std::string labelsPath = "";
    std::size_t sampleIndex = 0;
};

void printUsage(const char *exeName) {
    std::cout << "Usage:\n"
              << "  " << exeName
              << " train [images] [labels] [model_out] [epochs] [learning_rate] [sample_limit] [validation_split] [options]\n"
              << "  " << exeName
              << " predict [model] [images] [sample_index] [labels_optional]\n\n"
              << "Train options:\n"
              << "  --target-val-acc <float>       Stop when val accuracy reaches this value (default 1.0)\n"
              << "  --early-stop-patience <int>    Stop if val acc does not improve for N epochs (default 0=off)\n"
              << "  --checkpoint-every <int>       Save weights every N epochs (default 1)\n"
              << "  --resume <model_path>          Load existing weights before training\n"
              << "  --seed <int>                   RNG seed for deterministic split/shuffle (default 42)\n"
              << "  --topology <csv>              Layer sizes (example: 784,256,128,10)\n"
              << "  --activations <csv>           Activations per non-input layer\n"
              << "                                (example: relu,relu,softmax)\n\n"
              << "Examples:\n"
              << "  " << exeName
              << " train images labels model.nn 20 0.01 60000 0.1 --target-val-acc 0.995 --early-stop-patience 3\n"
              << "  " << exeName
              << " train images labels model.nn 15 0.005 60000 0.1 --topology 784,256,10 --activations relu,softmax\n"
              << "  " << exeName << " train images labels model.nn 10 0.005 0 0.1 --resume model.nn\n"
              << "  " << exeName << " predict model.nn images 42 labels\n"
              << "  " << exeName << " predict model.nn images 42\n";
}

bool isOptionToken(const std::string &arg) {
    return arg.rfind("--", 0) == 0;
}

std::string trim(const std::string &value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::vector<std::string> splitCsv(const std::string &csv) {
    std::vector<std::string> tokens;
    std::stringstream stream(csv);
    std::string part;
    while (std::getline(stream, part, ',')) {
        const std::string cleaned = trim(part);
        if (cleaned.empty()) {
            throw std::invalid_argument("CSV values cannot contain empty tokens");
        }
        tokens.push_back(cleaned);
    }

    if (tokens.empty()) {
        throw std::invalid_argument("CSV value cannot be empty");
    }
    return tokens;
}

std::vector<uint> parseTopologyCsv(const std::string &csv) {
    const std::vector<std::string> tokens = splitCsv(csv);
    if (tokens.size() < 2) {
        throw std::invalid_argument("Topology must contain at least input and output layers");
    }

    std::vector<uint> topology;
    topology.reserve(tokens.size());

    for (const std::string &token : tokens) {
        const unsigned long long parsed = std::stoull(token);
        if (parsed == 0ULL || parsed > std::numeric_limits<uint>::max()) {
            throw std::invalid_argument("Topology layer sizes must be in range [1, UINT_MAX]");
        }
        topology.push_back(static_cast<uint>(parsed));
    }

    return topology;
}

std::vector<ActivationType> parseActivationsCsv(const std::string &csv) {
    const std::vector<std::string> tokens = splitCsv(csv);
    std::vector<ActivationType> activations;
    activations.reserve(tokens.size());
    for (const std::string &token : tokens) {
        activations.push_back(NeuralNetwork::activationFromString(token));
    }
    return activations;
}

std::string formatTopology(const std::vector<uint> &topology) {
    std::ostringstream out;
    for (std::size_t i = 0; i < topology.size(); ++i) {
        if (i != 0) {
            out << " -> ";
        }
        out << topology[i];
    }
    return out.str();
}

std::string formatActivations(const std::vector<ActivationType> &activations) {
    std::ostringstream out;
    for (std::size_t i = 0; i < activations.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << NeuralNetwork::activationToString(activations[i]);
    }
    return out.str();
}

std::string requireOptionValue(int argc, char **argv, int &i, const std::string &optionName) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + optionName);
    }
    return argv[++i];
}

TrainConfig parseTrainConfig(int argc, char **argv) {
    TrainConfig cfg;
    std::vector<std::string> positional;
    int i = 2;
    for (; i < argc; ++i) {
        const std::string arg = argv[i];
        if (isOptionToken(arg)) {
            break;
        }
        positional.push_back(arg);
    }

    if (positional.size() > 0) {
        cfg.imagesPath = positional[0];
    }
    if (positional.size() > 1) {
        cfg.labelsPath = positional[1];
    }
    if (positional.size() > 2) {
        cfg.modelOutputPath = positional[2];
    }
    if (positional.size() > 3) {
        cfg.epochs = static_cast<std::size_t>(std::stoul(positional[3]));
    }
    if (positional.size() > 4) {
        cfg.learningRate = std::stod(positional[4]);
    }
    if (positional.size() > 5) {
        cfg.sampleLimit = static_cast<std::size_t>(std::stoul(positional[5]));
    }
    if (positional.size() > 6) {
        cfg.validationSplit = std::stod(positional[6]);
    }
    if (positional.size() > 7) {
        throw std::invalid_argument("Too many positional arguments for train mode");
    }

    for (; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--target-val-acc") {
            cfg.targetValAccuracy = std::stod(requireOptionValue(argc, argv, i, "--target-val-acc"));
        } else if (arg == "--early-stop-patience") {
            cfg.earlyStopPatience =
                static_cast<std::size_t>(std::stoul(requireOptionValue(argc, argv, i, "--early-stop-patience")));
        } else if (arg == "--checkpoint-every") {
            cfg.checkpointEvery =
                static_cast<std::size_t>(std::stoul(requireOptionValue(argc, argv, i, "--checkpoint-every")));
            if (cfg.checkpointEvery == 0) {
                throw std::invalid_argument("--checkpoint-every must be >= 1");
            }
        } else if (arg == "--resume") {
            cfg.resumeModelPath = requireOptionValue(argc, argv, i, "--resume");
        } else if (arg == "--seed") {
            cfg.seed = static_cast<std::uint32_t>(std::stoul(requireOptionValue(argc, argv, i, "--seed")));
        } else if (arg == "--topology") {
            cfg.topology = parseTopologyCsv(requireOptionValue(argc, argv, i, "--topology"));
            cfg.hasTopologyOverride = true;
        } else if (arg == "--activations") {
            cfg.activations =
                parseActivationsCsv(requireOptionValue(argc, argv, i, "--activations"));
            cfg.hasActivationOverride = true;
        } else {
            throw std::invalid_argument("Unknown train option: " + arg);
        }
    }

    if (cfg.learningRate <= 0.0 && cfg.resumeModelPath.empty()) {
        throw std::invalid_argument(
            "learning_rate must be greater than zero when starting a new model");
    }
    if (cfg.topology.size() < 2) {
        throw std::invalid_argument("Topology must contain at least 2 layers");
    }
    if (cfg.activations.size() != cfg.topology.size() - 1) {
        throw std::invalid_argument("Activation count must match topology layer count minus one");
    }
    if (cfg.validationSplit < 0.0 || cfg.validationSplit >= 1.0) {
        throw std::invalid_argument("validation_split must be in [0.0, 1.0)");
    }
    if (cfg.targetValAccuracy < 0.0 || cfg.targetValAccuracy > 1.0) {
        throw std::invalid_argument("--target-val-acc must be in [0.0, 1.0]");
    }

    return cfg;
}

PredictConfig parsePredictConfig(int argc, char **argv) {
    PredictConfig cfg;
    if (argc > 2) {
        cfg.modelPath = argv[2];
    }
    if (argc > 3) {
        cfg.imagesPath = argv[3];
    }
    if (argc > 4) {
        cfg.sampleIndex = static_cast<std::size_t>(std::stoul(argv[4]));
    }
    if (argc > 5) {
        cfg.labelsPath = argv[5];
    }
    return cfg;
}

double evaluateAccuracy(NeuralNetwork &net,
                        const MnistDataset &dataset,
                        const std::vector<std::size_t> &indices) {
    if (indices.empty()) {
        return 0.0;
    }

    std::size_t correct = 0;
    for (std::size_t idx : indices) {
        const MnistSample &sample = dataset[idx];
        if (net.predictClass(sample.pixels) == sample.label) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(indices.size());
}

bool shouldSaveCheckpoint(std::size_t epochIndex, std::size_t checkpointEvery, std::size_t totalEpochs) {
    if ((epochIndex + 1) % checkpointEvery == 0) {
        return true;
    }
    return (epochIndex + 1) == totalEpochs;
}

void runTrainMode(const TrainConfig &cfg) {
    std::signal(SIGINT, handleSignal);

    std::cout << "Loading MNIST data..." << std::endl;
    const MnistDataset dataset = MnistDataset::load(cfg.imagesPath, cfg.labelsPath, cfg.sampleLimit);

    std::cout << "Loaded " << dataset.size() << " samples from images='" << cfg.imagesPath
              << "' labels='" << cfg.labelsPath << "'" << std::endl;

    if (dataset.size() < 2) {
        throw std::runtime_error("Need at least 2 samples to run train/validation split");
    }
    if (!cfg.resumeModelPath.empty() && (cfg.hasTopologyOverride || cfg.hasActivationOverride)) {
        throw std::invalid_argument("--topology/--activations cannot be used together with --resume");
    }

    NeuralNetwork net = cfg.resumeModelPath.empty() ? NeuralNetwork(cfg.topology, cfg.activations, cfg.learningRate)
                                                    : NeuralNetwork::loadModel(cfg.resumeModelPath, cfg.learningRate);

    if (!cfg.resumeModelPath.empty()) {
        std::cout << "Resumed from model: '" << cfg.resumeModelPath << "'" << std::endl;
    }
    std::cout << "Training topology: " << formatTopology(net.getTopology()) << std::endl;
    std::cout << "Activations: " << formatActivations(net.getActivations()) << std::endl;

    const std::size_t outputClasses = net.getTopology().back();
    if (outputClasses == 0) {
        throw std::runtime_error("Output layer must have at least one neuron");
    }
    if (outputClasses < 10) {
        throw std::invalid_argument(
            "MNIST labels require output layer size >= 10. Increase --topology output width.");
    }

    std::vector<std::size_t> allIndices(dataset.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::mt19937 rng(cfg.seed);
    std::shuffle(allIndices.begin(), allIndices.end(), rng);

    std::size_t validationCount = static_cast<std::size_t>(dataset.size() * cfg.validationSplit);
    if (cfg.validationSplit > 0.0) {
        validationCount = std::max<std::size_t>(1, validationCount);
        validationCount = std::min<std::size_t>(validationCount, dataset.size() - 1);
    }

    const std::size_t trainCount = dataset.size() - validationCount;
    std::vector<std::size_t> trainIndices(allIndices.begin(), allIndices.begin() + trainCount);
    std::vector<std::size_t> validationIndices(allIndices.begin() + trainCount, allIndices.end());

    std::cout << "Train samples: " << trainIndices.size()
              << " | Validation samples: " << validationIndices.size() << std::endl;

    double bestValAcc = -1.0;
    std::size_t epochsWithoutImprove = 0;

    for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        if (g_stopRequested) {
            std::cout << "\nStop requested before epoch start. Saving model..." << std::endl;
            net.saveModel(cfg.modelOutputPath);
            std::cout << "Saved model to '" << cfg.modelOutputPath << "'" << std::endl;
            return;
        }

        std::shuffle(trainIndices.begin(), trainIndices.end(), rng);

        double trainLoss = 0.0;
        for (std::size_t samplePos = 0; samplePos < trainIndices.size(); ++samplePos) {
            const std::size_t idx = trainIndices[samplePos];
            const MnistSample &sample = dataset[idx];
            const std::vector<double> target = dataset.oneHotLabel(idx, outputClasses);
            trainLoss += net.trainSample(sample.pixels, target);

            if (g_stopRequested) {
                std::cout << "\nStop requested during epoch. Saving model..." << std::endl;
                net.saveModel(cfg.modelOutputPath);
                std::cout << "Saved model to '" << cfg.modelOutputPath << "'" << std::endl;
                return;
            }
        }
        trainLoss /= static_cast<double>(trainIndices.size());

        const double trainAcc = evaluateAccuracy(net, dataset, trainIndices);
        const double valAcc = evaluateAccuracy(net, dataset, validationIndices);

        std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs
                  << " - train_loss: " << trainLoss
                  << " - train_acc: " << std::fixed << std::setprecision(2) << (trainAcc * 100.0) << "%"
                  << " - val_acc: " << (valAcc * 100.0) << "%"
                  << std::defaultfloat << std::endl;

        if (shouldSaveCheckpoint(epoch, cfg.checkpointEvery, cfg.epochs)) {
            net.saveModel(cfg.modelOutputPath);
            std::cout << "Checkpoint saved to '" << cfg.modelOutputPath << "'" << std::endl;
        }

        if (valAcc > bestValAcc) {
            bestValAcc = valAcc;
            epochsWithoutImprove = 0;
        } else {
            ++epochsWithoutImprove;
        }

        if (valAcc >= cfg.targetValAccuracy) {
            std::cout << "Early stop: validation accuracy reached target ("
                      << (cfg.targetValAccuracy * 100.0) << "%)." << std::endl;
            return;
        }

        if (cfg.earlyStopPatience > 0 && epochsWithoutImprove >= cfg.earlyStopPatience) {
            std::cout << "Early stop: no validation improvement for "
                      << cfg.earlyStopPatience << " epoch(s)." << std::endl;
            return;
        }
    }

    net.saveModel(cfg.modelOutputPath);
    std::cout << "Saved final model to '" << cfg.modelOutputPath << "'" << std::endl;
}

void runPredictMode(const PredictConfig &cfg) {
    NeuralNetwork net = NeuralNetwork::loadModel(cfg.modelPath, 0.0);
    const bool hasLabels = !cfg.labelsPath.empty();
    const MnistDataset dataset =
        hasLabels ? MnistDataset::load(cfg.imagesPath, cfg.labelsPath, 0)
                  : MnistDataset::loadImagesOnly(cfg.imagesPath, 0);

    if (cfg.sampleIndex >= dataset.size()) {
        throw std::out_of_range("sample_index out of range for provided MNIST files");
    }

    const MnistSample &sample = dataset[cfg.sampleIndex];
    const std::vector<double> probs = net.forward(sample.pixels);
    const std::uint8_t predicted = net.predictClass(sample.pixels);

    std::cout << "Prediction for sample " << cfg.sampleIndex << ":\n";
    std::cout << "  predicted_class: " << static_cast<int>(predicted) << "\n";
    if (hasLabels) {
        std::cout << "  true_label: " << static_cast<int>(sample.label) << "\n";
    }
    std::cout << "  probabilities:" << std::endl;

    for (std::size_t i = 0; i < probs.size(); ++i) {
        std::cout << "    class " << i << ": " << std::fixed << std::setprecision(6) << probs[i] << std::endl;
    }

    std::cout << std::defaultfloat;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            printUsage(argv[0]);
            return 1;
        }

        const std::string mode = argv[1];
        if (mode == "train") {
            runTrainMode(parseTrainConfig(argc, argv));
            return 0;
        }
        if (mode == "predict") {
            runPredictMode(parsePredictConfig(argc, argv));
            return 0;
        }

        throw std::invalid_argument("Unknown mode: " + mode);
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }
}
