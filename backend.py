from flask import Flask, request, jsonify, send_file
from flask import Flask, request, jsonify
from text_processing import extract_requirements

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_request():
    data = request.json
    user_text = data.get("text", "")

    aspect_ratio, format = extract_requirements(user_text)

    if not aspect_ratio or not format:
        return jsonify({"reply": "I couldn't understand your request. Please specify a valid format and aspect ratio."})

    return jsonify({"reply": f"Processing video as {aspect_ratio} {format}. Need any other changes?"})

if __name__ == "__main__":
    app.run(debug=True)
import os
from video_processing import process_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    return jsonify({"message": "Upload successful", "file_path": filename})

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    input_path = data.get("file_path")
    platform = data.get("platform")
    video_type = data.get("video_type")
    format = data.get("format", "mp4")
    resolution = data.get("resolution", "1080p")
    aspect_ratio = data.get("aspect_ratio", None)
    auto_caption = data.get("auto_caption", False)

    if not input_path or not platform:
        return jsonify({"error": "Missing required parameters"}), 400

    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{os.path.basename(input_path)}")
    process_video(input_path, output_path, platform, video_type, format, resolution, aspect_ratio, auto_caption)

    return jsonify({"message": "Processing complete", "output_path": output_path})

@app.route("/download", methods=["GET"])
def download():
    file_path = request.args.get("file")
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
