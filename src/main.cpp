#include "global_neuralnetwork.hpp"

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
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
              << "  --seed <int>                   RNG seed for deterministic split/shuffle (default 42)\n\n"
              << "Examples:\n"
              << "  " << exeName
              << " train images labels model.nn 20 0.01 60000 0.1 --target-val-acc 0.995 --early-stop-patience 3\n"
              << "  " << exeName << " train images labels model.nn 10 0.005 0 0.1 --resume model.nn\n"
              << "  " << exeName << " predict model.nn images 42 labels\n"
              << "  " << exeName << " predict model.nn images 42\n";
}

TrainConfig parseTrainConfig(int argc, char **argv) {
    TrainConfig cfg;
    if (argc > 2) {
        cfg.imagesPath = argv[2];
    }
    if (argc > 3) {
        cfg.labelsPath = argv[3];
    }
    if (argc > 4) {
        cfg.modelOutputPath = argv[4];
    }
    if (argc > 5) {
        cfg.epochs = static_cast<std::size_t>(std::stoul(argv[5]));
    }
    if (argc > 6) {
        cfg.learningRate = std::stod(argv[6]);
    }
    if (argc > 7) {
        cfg.sampleLimit = static_cast<std::size_t>(std::stoul(argv[7]));
    }
    if (argc > 8) {
        cfg.validationSplit = std::stod(argv[8]);
    }

    for (int i = 9; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--target-val-acc") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for --target-val-acc");
            }
            cfg.targetValAccuracy = std::stod(argv[++i]);
        } else if (arg == "--early-stop-patience") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for --early-stop-patience");
            }
            cfg.earlyStopPatience = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--checkpoint-every") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for --checkpoint-every");
            }
            cfg.checkpointEvery = static_cast<std::size_t>(std::stoul(argv[++i]));
            if (cfg.checkpointEvery == 0) {
                throw std::invalid_argument("--checkpoint-every must be >= 1");
            }
        } else if (arg == "--resume") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for --resume");
            }
            cfg.resumeModelPath = argv[++i];
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for --seed");
            }
            cfg.seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else {
            throw std::invalid_argument("Unknown train option: " + arg);
        }
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

    const std::vector<uint> topology = {784, 128, 64, 10};
    const std::vector<ActivationType> activations = {
        ActivationType::ReLU,
        ActivationType::ReLU,
        ActivationType::Softmax,
    };

    NeuralNetwork net = cfg.resumeModelPath.empty()
                            ? NeuralNetwork(topology, activations, cfg.learningRate)
                            : NeuralNetwork::loadModel(cfg.resumeModelPath, cfg.learningRate);

    if (!cfg.resumeModelPath.empty()) {
        std::cout << "Resumed from model: '" << cfg.resumeModelPath << "'" << std::endl;
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
            const std::vector<double> target = dataset.oneHotLabel(idx, 10);
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
