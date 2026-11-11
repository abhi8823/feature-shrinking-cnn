"""Feature-Shrinking CNN using TensorFlow/Keras Sequential API."""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense


def build_model():
    """Build and return the required Feature-Shrinking CNN."""
    model = Sequential(
        [
            Input(shape=(28, 28, 1), name="grayscale_input"),
            Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                name="conv2d_1",
            ),
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                name="conv2d_2",
            ),
            Flatten(name="flatten"),
            Dense(10, activation="softmax", name="classification_output"),
        ],
        name="feature_shrinking_cnn",
    )
    return model


if __name__ == "__main__":
    cnn = build_model()
    cnn.summary()

    # Smoke test with one normalized grayscale image.
    sample = tf.random.uniform((1, 28, 28, 1), minval=0.0, maxval=1.0)
    probabilities = cnn(sample, training=False)

    print("\nOutput shape:", probabilities.shape)
    print(
        "Sum of class probabilities:",
        float(tf.reduce_sum(probabilities[0])),
    )
