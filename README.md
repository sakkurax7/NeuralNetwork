# NeuralNetwork (MNIST Refactor)

A clean C++17 feed-forward neural network implementation focused on MNIST.

## Features
- Layer-based network implementation (easy to read and extend).
- Pluggable activations per layer:
  - `Sigmoid`
  - `Tanh`
  - `ReLU`
  - `Softmax` (output layer)
- Stable softmax with cross-entropy training.
- IDX MNIST reader for image/label binaries.

## Build
```bash
make
```

## Run
```bash
./bin/main [images_path] [labels_path] [epochs] [learning_rate] [sample_limit]
```

Defaults:
- `images_path`: `images`
- `labels_path`: `labels`
- `epochs`: `3`
- `learning_rate`: `0.01`
- `sample_limit`: `0` (use all available paired samples)

Example:
```bash
./bin/main images labels 5 0.01 10000
```
