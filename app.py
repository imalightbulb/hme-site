import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from model import load_trained_model, get_label_mapping
from main import segment_symbols, preprocess_symbol, predict_symbols, evaluate_expression


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

model = load_trained_model()
label_to_class = get_label_mapping()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode base64 image
    image_data = image_data.split(",")[1]  # Remove header (data:image/png;base64,...)
    img_bytes = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Save the image for processing
    file_path = os.path.join(UPLOAD_FOLDER, "drawn_expression.png")
    cv2.imwrite(file_path, img)

    return jsonify({"message": "Image uploaded successfully", "path": file_path})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        print("Processing image...")  # Debug log

        # Check if the uploaded image exists
        file_path = os.path.join(UPLOAD_FOLDER, "drawn_expression.png")
        if not os.path.exists(file_path):
            print("File not found:", file_path)  # Debug log
            return jsonify({"error": "No uploaded image found. Please upload an image first."}), 400

        # Segment symbols
        print("Segmenting symbols...")  # Debug log
        symbol_imgs = segment_symbols(file_path)
        if not symbol_imgs:
            print("No symbols detected!")  # Debug log
            return jsonify({"error": "No symbols detected in the image."}), 400

        # Predict symbols using `predict_symbols()`
        predictions = predict_symbols(symbol_imgs)

        # Evaluate the expression
        print("Evaluating expression...")  # Debug log
        expression_result = evaluate_expression([(label, probs) for label, probs in predictions])

        # Format predictions for response
        formatted_predictions = [{"label": label, "probs": probs.tolist()} for label, probs in predictions]

        print("Predictions:", formatted_predictions)  # Debug log
        print("Evaluation Result:", expression_result)  # Debug log

        return jsonify({
            "predictions": [pred["label"] for pred in formatted_predictions],
            "evaluation": expression_result
        })

    except Exception as e:
        print("Error in /process:", str(e))  # Debug log
        return jsonify({"error": str(e)}), 500


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)



if __name__ == '__main__':
    app.run(debug=True)
