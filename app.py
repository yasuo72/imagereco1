from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from image_recognition import recognize_image
from chatbot import generate_detailed_response
from chat import chatbot_response

app = Flask(__name__)

# Manual CORS handling
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request format"}), 400

        text = data['message']
        if not text.strip():
            return jsonify({"error": "Empty message received"}), 400

        response = chatbot_response(text)
        return jsonify({
            "status": "success",
            "answer": response
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        # Recognize image content
        recognized_objects = recognize_image(filepath)
        if not recognized_objects:
            return jsonify({"error": "Could not recognize image content"}), 400

        # Format description with cleaned object names
        image_description = ", ".join([
            f"{obj[0].replace('_', ' ').strip()} ({obj[1] * 100:.2f}%)"
            for obj in recognized_objects
        ])

        # Generate enhanced response with definitions
        chatbot_data = generate_detailed_response(image_description)

        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "status": "success",
            "detected_objects": [
                {"name": obj[0], "confidence": float(obj[1])}
                for obj in recognized_objects
            ],
            "analysis": chatbot_data.get("analysis", ""),
            "definitions": chatbot_data.get("definitions", {}),
            "description": image_description
        })

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)