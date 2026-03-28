#include "global_neuralnetwork.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace {
std::uint32_t readBigEndianUInt32(std::ifstream &stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char *>(bytes), 4);
    if (!stream) {
        throw std::runtime_error("Failed to read IDX header");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}
} // namespace

MnistDataset MnistDataset::load(const std::string &imagesPath,
                                const std::string &labelsPath,
                                std::size_t limit) {
    std::ifstream images(imagesPath, std::ios::binary);
    if (!images) {
        throw std::runtime_error("Unable to open images file: " + imagesPath);
    }

    std::ifstream labels(labelsPath, std::ios::binary);
    if (!labels) {
        throw std::runtime_error("Unable to open labels file: " + labelsPath);
    }

    const std::uint32_t imageMagic = readBigEndianUInt32(images);
    const std::uint32_t imageCount = readBigEndianUInt32(images);
    const std::uint32_t rows = readBigEndianUInt32(images);
    const std::uint32_t cols = readBigEndianUInt32(images);

    const std::uint32_t labelMagic = readBigEndianUInt32(labels);
    const std::uint32_t labelCount = readBigEndianUInt32(labels);

    if (imageMagic != 2051) {
        throw std::runtime_error("Invalid image file magic number (expected 2051)");
    }
    if (labelMagic != 2049) {
        throw std::runtime_error("Invalid label file magic number (expected 2049)");
    }

    const std::size_t available = static_cast<std::size_t>(std::min(imageCount, labelCount));
    if (available == 0) {
        throw std::runtime_error("MNIST files contain zero samples");
    }

    const std::size_t toRead = (limit == 0 || limit > available) ? available : limit;
    const std::size_t pixelsPerImage = static_cast<std::size_t>(rows) * cols;

    MnistDataset dataset;
    dataset.samples.reserve(toRead);

    for (std::size_t idx = 0; idx < toRead; ++idx) {
        MnistSample sample;
        sample.pixels.resize(pixelsPerImage);

        for (std::size_t p = 0; p < pixelsPerImage; ++p) {
            unsigned char pixel = 0;
            images.read(reinterpret_cast<char *>(&pixel), 1);
            if (!images) {
                throw std::runtime_error("Unexpected end of image file");
            }
            sample.pixels[p] = static_cast<double>(pixel) / 255.0;
        }

        unsigned char label = 0;
        labels.read(reinterpret_cast<char *>(&label), 1);
        if (!labels) {
            throw std::runtime_error("Unexpected end of labels file");
        }

        sample.label = label;
        dataset.samples.push_back(std::move(sample));
    }

    return dataset;
}

MnistDataset MnistDataset::loadImagesOnly(const std::string &imagesPath, std::size_t limit) {
    std::ifstream images(imagesPath, std::ios::binary);
    if (!images) {
        throw std::runtime_error("Unable to open images file: " + imagesPath);
    }

    const std::uint32_t imageMagic = readBigEndianUInt32(images);
    const std::uint32_t imageCount = readBigEndianUInt32(images);
    const std::uint32_t rows = readBigEndianUInt32(images);
    const std::uint32_t cols = readBigEndianUInt32(images);

    if (imageMagic != 2051) {
        throw std::runtime_error("Invalid image file magic number (expected 2051)");
    }

    const std::size_t available = static_cast<std::size_t>(imageCount);
    if (available == 0) {
        throw std::runtime_error("MNIST image file contains zero samples");
    }

    const std::size_t toRead = (limit == 0 || limit > available) ? available : limit;
    const std::size_t pixelsPerImage = static_cast<std::size_t>(rows) * cols;

    MnistDataset dataset;
    dataset.samples.reserve(toRead);

    for (std::size_t idx = 0; idx < toRead; ++idx) {
        MnistSample sample;
        sample.pixels.resize(pixelsPerImage);
        sample.label = 255;

        for (std::size_t p = 0; p < pixelsPerImage; ++p) {
            unsigned char pixel = 0;
            images.read(reinterpret_cast<char *>(&pixel), 1);
            if (!images) {
                throw std::runtime_error("Unexpected end of image file");
            }
            sample.pixels[p] = static_cast<double>(pixel) / 255.0;
        }

        dataset.samples.push_back(std::move(sample));
    }

    return dataset;
}

std::size_t MnistDataset::size() const {
    return samples.size();
}

const MnistSample &MnistDataset::operator[](std::size_t index) const {
    if (index >= samples.size()) {
        throw std::out_of_range("Sample index out of range");
    }
    return samples[index];
}

std::vector<double> MnistDataset::oneHotLabel(std::size_t index, std::size_t classes) const {
    if (classes == 0) {
        throw std::invalid_argument("Class count must be greater than zero");
    }

    const MnistSample &sample = (*this)[index];
    if (sample.label >= classes) {
        throw std::runtime_error("Label exceeds provided class count");
    }

    std::vector<double> target(classes, 0.0);
    target[sample.label] = 1.0;
    return target;
}
