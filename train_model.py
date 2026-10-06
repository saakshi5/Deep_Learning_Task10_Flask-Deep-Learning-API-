import os

# Optional: reduce oneDNN informational messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("=" * 60)
print("MNIST CNN MODEL TRAINING")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load MNIST dataset
# ---------------------------------------------------------

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data:", x_train.shape)
print("Testing data :", x_test.shape)


# ---------------------------------------------------------
# 2. Normalize images
# ---------------------------------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nPreprocessing completed.")


# ---------------------------------------------------------
# 3. Data augmentation
# ---------------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    )
])


# ---------------------------------------------------------
# 4. Create CNN model
# ---------------------------------------------------------

model = models.Sequential([

    layers.Input(shape=(28, 28, 1)),

    # Data augmentation
    data_augmentation,

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),


    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),


    # Classification layers
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# ---------------------------------------------------------
# 5. Compile model
# ---------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------------------------
# 6. Display model
# ---------------------------------------------------------

print("\nModel architecture:")

model.summary()


# ---------------------------------------------------------
# 7. Train model
# ---------------------------------------------------------

print("\nStarting training...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


# ---------------------------------------------------------
# 8. Evaluate model
# ---------------------------------------------------------

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 60)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Test Loss:",
    round(test_loss, 4)
)

print("=" * 60)


# ---------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------

os.makedirs("model", exist_ok=True)

model_path = "model/mnist_cnn.keras"

model.save(model_path)

print("\nModel saved successfully!")

print("Saved at:")
print(model_path)

print("\nTraining completed successfully.")