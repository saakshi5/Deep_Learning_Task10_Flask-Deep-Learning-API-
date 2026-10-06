import os

# Optional: reduce oneDNN informational messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify, render_template

import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps, ImageEnhance


# ---------------------------------------------------------
# Flask application
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

MODEL_PATH = "model/mnist_cnn.keras"

print("Loading deep learning model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ---------------------------------------------------------
# API status
# ---------------------------------------------------------

@app.route("/api")
def api_status():

    return jsonify({

        "application": "Flask Deep Learning API",

        "model": "CNN",

        "dataset": "MNIST",

        "status": "running"

    })


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # -----------------------------------------------------
    # Convert image to NumPy array
    # -----------------------------------------------------

    image_array = np.array(image)

    # -----------------------------------------------------
    # Determine whether image needs inversion
    #
    # MNIST:
    # black background + white digit
    #
    # Normal uploaded image:
    # white background + black digit
    # -----------------------------------------------------

    mean_value = np.mean(image_array)

    if mean_value > 127:

        image = ImageOps.invert(image)

    # -----------------------------------------------------
    # Improve contrast
    # -----------------------------------------------------

    image = ImageEnhance.Contrast(image).enhance(2.0)

    # Convert to NumPy array again
    image_array = np.array(image)

    # -----------------------------------------------------
    # Remove very dark background pixels
    # -----------------------------------------------------

    threshold = 30

    binary = image_array > threshold

    # -----------------------------------------------------
    # Find bounding box of digit
    # -----------------------------------------------------

    if np.any(binary):

        rows = np.any(binary, axis=1)

        cols = np.any(binary, axis=0)

        row_indices = np.where(rows)[0]

        col_indices = np.where(cols)[0]

        top = row_indices[0]

        bottom = row_indices[-1] + 1

        left = col_indices[0]

        right = col_indices[-1] + 1

        image = image.crop(
            (left, top, right, bottom)
        )

    # -----------------------------------------------------
    # Resize while maintaining aspect ratio
    # -----------------------------------------------------

    width, height = image.size

    if width == 0 or height == 0:

        raise ValueError(
            "Could not detect digit in image."
        )

    scale = 20 / max(width, height)

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # -----------------------------------------------------
    # Create 28x28 black canvas
    # -----------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    # -----------------------------------------------------
    # Center digit
    # -----------------------------------------------------

    x_position = (28 - new_width) // 2

    y_position = (28 - new_height) // 2

    canvas.paste(
        image,
        (x_position, y_position)
    )

    # -----------------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------------

    image_array = np.array(canvas)

    # Normalize
    image_array = (
        image_array.astype("float32") / 255.0
    )

    # Add channel dimension
    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ---------------------------------------------------------
# Prediction API
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check file
        if "image" not in request.files:

            return jsonify({

                "error": "No image file provided",

                "status": "failed"

            }), 400


        file = request.files["image"]


        if file.filename == "":

            return jsonify({

                "error": "No image selected",

                "status": "failed"

            }), 400


        # -------------------------------------------------
        # Open uploaded image
        # -------------------------------------------------

        image = Image.open(file)


        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        processed_image = preprocess_image(image)


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(
            processed_image,
            verbose=0
        )


        # -------------------------------------------------
        # Get predicted digit
        # -------------------------------------------------

        predicted_digit = int(
            np.argmax(prediction[0])
        )


        # -------------------------------------------------
        # Get confidence
        # -------------------------------------------------

        confidence = float(
            np.max(prediction[0])
        )


        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return jsonify({

            "prediction": predicted_digit,

            "confidence":
                round(
                    confidence * 100,
                    2
                ),

            "status": "success"

        })


    except Exception as e:

        return jsonify({

            "error": str(e),

            "status": "failed"

        }), 500


# ---------------------------------------------------------
# Start Flask application
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )