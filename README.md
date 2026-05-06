# NeuralNetwork (MNIST Refactor)

A clean C++17 feed-forward neural network implementation focused on MNIST.

## Features
- Layer-based network implementation (easy to read and extend).
- Pluggable activations per layer:
  - `Sigmoid`
  - `Tanh`
  - `ReLU`
  - `Softmax` (output layer)
- Configurable topology and activations from CLI (`--topology`, `--activations`).
- Reusable network API for vector-based training/inference (library-style usage).
- Stable softmax with cross-entropy training.
- IDX MNIST reader for image/label binaries.
- Train/validation split during training.
- Model persistence (save/load learned weights).
- Prediction mode using saved model weights.

## Build
```bash
make
```

## Train (with validation split + save weights)
```bash
./bin/main train [images] [labels] [model_out] [epochs] [learning_rate] [sample_limit] [validation_split] [options]
```

Defaults:
- `images`: `images`
- `labels`: `labels`
- `model_out`: `model.nn`
- `epochs`: `5`
- `learning_rate`: `0.01`
- `sample_limit`: `0` (use all available paired samples)
- `validation_split`: `0.1`

Options:
- `--target-val-acc <float>`: stop early when validation accuracy reaches this threshold (default `1.0`)
- `--early-stop-patience <int>`: stop if validation accuracy does not improve for N epochs (default `0`, disabled)
- `--checkpoint-every <int>`: save model every N epochs (default `1`)
- `--resume <model_path>`: resume training from saved weights
- `--seed <int>`: deterministic split/shuffle seed (default `42`)
- `--topology <csv>`: layer widths, e.g. `784,256,128,10`
- `--activations <csv>`: activation for each non-input layer, e.g. `relu,relu,softmax`

Example:
```bash
./bin/main train images labels model.nn 20 0.01 60000 0.1 --target-val-acc 0.995 --early-stop-patience 3
```

Custom architecture example:
```bash
./bin/main train images labels model.nn 15 0.005 60000 0.1 --topology 784,256,10 --activations relu,softmax
```

Resume example:
```bash
./bin/main train images labels model.nn 10 0.005 0 0.1 --resume model.nn
```

Interrupt handling:
- Press `Ctrl+C` during training to stop gracefully and save weights to `model_out`.

## Predict (using saved model)
```bash
./bin/main predict [model] [images] [sample_index] [labels_optional]
```

Examples:
```bash
# Predict with known labels (prints true label too)
./bin/main predict model.nn images 42 labels

# Predict with images file only
./bin/main predict model.nn images 42
```

## Convert png/jpg to MNIST IDX
Use the helper script in `tools/`:

```bash
python3 tools/image_to_mnist.py <input_image> <output_images_idx> [--label 7] [--output-labels-idx out.labels] [--invert]
```

Examples:
```bash
# Create an images IDX file for prediction
python3 tools/image_to_mnist.py digit.png my_digit.idx --invert

# Create images + labels IDX files for evaluation
python3 tools/image_to_mnist.py digit.jpg my_digit.idx --label 3 --output-labels-idx my_digit.labels --invert

# Predict your converted sample
./bin/main predict model.nn my_digit.idx 0
```

## Notes
- MNIST image pixels are normalized to `[0, 1]`.
- Default network topology is `784 -> 128 -> 64 -> 10`.
- Current model file format is a human-readable text format (`NN_MODEL_V1`).
- The converter script requires Pillow (`pip install pillow`).

## Library-style API
You can now use `NeuralNetwork` directly with generic vectors:

```cpp
NeuralNetwork net({4, 8, 3}, {ActivationType::ReLU, ActivationType::Softmax}, 0.01);
double finalLoss = net.train(trainInputs, trainTargets, 20, true, 42);
std::vector<double> probs = net.run(newInput);
std::uint8_t predicted = net.predictClass(newInput);
net.saveModel("model.nn");
```
