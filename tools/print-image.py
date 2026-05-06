import struct
import numpy as np

# Paths to MNIST test data (adjust if needed)
image_file = '../Data/t10k-images-idx3-ubyte'
label_file = '../Data/t10k-labels-idx1-ubyte'

# Function to read IDX images
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images

# Function to read IDX labels
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load test data
test_images = load_mnist_images(image_file)
test_labels = load_mnist_labels(label_file)

# Pick a single image (e.g., index 0)
idx = 1
image = test_images[idx]
label = test_labels[idx]

# Print image in console using simple ASCII
for row in image:
    print("".join(['{:2}'.format(' ' if pixel < 128 else '#') for pixel in row]))

print(f"Label: {label}")

# Save a single-image MNIST-compatible file
def save_single_mnist_image(image, out_file):
    num_images = 1
    rows, cols = image.shape
    with open(out_file, 'wb') as f:
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))  # 2051 = magic number for images
        f.write(image.tobytes())

save_single_mnist_image(image, 'single_image.idx3-ubyte')
print("Single MNIST-compatible image saved as 'single_image.idx3-ubyte'")