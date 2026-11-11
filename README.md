# Feature-Shrinking CNN

A Convolutional Neural Network built with the **TensorFlow/Keras Sequential API** for 10-class classification of 28×28 grayscale images.

The architecture reduces spatial dimensions through convolution strides and valid padding. It does **not** use a pooling layer.

## CNN Architecture

| Layer | Configuration | Output Shape | Parameters |
|---|---|---:|---:|
| Input | 28×28×1 grayscale image | 28×28×1 | 0 |
| Conv2D 1 | 16 filters, 3×3, stride 2, same padding, ReLU | 14×14×16 | 160 |
| Conv2D 2 | 32 filters, 3×3, stride 1, valid padding, ReLU | 12×12×32 | 4,640 |
| Flatten | — | 4,608 | 0 |
| Dense | 10 units, Softmax | 10 | 46,090 |

**Total parameters:** 50,890  
**Trainable parameters:** 50,890  
**Non-trainable parameters:** 0

## Requirements

- Python 3.9 or later
- TensorFlow

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python model.py
```

The script prints the model summary and performs a smoke test using one randomly generated grayscale image.

Expected final checks:

```text
Output shape: (1, 10)
Sum of class probabilities: approximately 1.0
```

The smoke test verifies that the model runs and produces 10 class probabilities. It does not train the model.

## Project Structure

```text
feature-shrinking-cnn/
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

- Input image shape: `(28, 28, 1)`
- Output shape for a batch of one image: `(1, 10)`
- No MaxPooling or AveragePooling layer is used.
