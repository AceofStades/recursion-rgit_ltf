from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip
import uuid
import re
import whisper
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# Ensure upload and output directories exist
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["TEMP_FOLDER"] = TEMP_FOLDER

# Load Whisper model
try:
    stt_model = whisper.load_model("base")
except Exception as e:
    stt_model = None
    print(f"Warning: Whisper model could not be loaded. Auto-captioning will be disabled. Error: {e}")

# Load YOLO model for face detection
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    yolo_model = None
    print(f"Warning: YOLO model could not be loaded. Face tracking will be disabled. Error: {e}")

# Helper function to validate file extensions
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload from the frontend."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    return jsonify({
        "message": "File uploaded successfully",
        "file_path": file_path
    })

@app.route("/get_resolution", methods=["POST"])
def get_resolution():
    """Fetches the original resolution of the uploaded video."""
    data = request.json
    video_path = data["file_path"]

    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return jsonify({
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def parse_aspect_ratio(aspect_ratio_str):
    """Parse an aspect ratio string like '16:9' into a tuple of integers."""
    match = re.match(r'(\d+):(\d+)', aspect_ratio_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def resize_video(video_path, output_path, aspect_ratio_str, resolution_percentage):
    """Resizes the video based on the percentage of original resolution while maintaining aspect ratio."""
    try:
        # Get original resolution
        clip = mp.VideoFileClip(video_path)
        original_width, original_height = clip.size

        # Calculate new resolution
        scale_factor = resolution_percentage / 100
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Adjust aspect ratio if provided
        aspect_ratio = parse_aspect_ratio(aspect_ratio_str)
        if aspect_ratio:
            aspect_w, aspect_h = aspect_ratio
            if (new_width / new_height) != (aspect_w / aspect_h):
                new_height = int(new_width * aspect_h / aspect_w)

        print(f"Resizing video to {new_width}x{new_height}")

        # Resize video
        resized_clip = clip.resize(newsize=(new_width, new_height))
        resized_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
    except Exception as e:
        print(f"Error resizing video: {e}")
        return None

def crop_video_to_face(video_path, output_path, aspect_ratio_str, target_width, target_height):
    """Crops the video to track faces and resizes to target dimensions."""
    if yolo_model is None:
        print("YOLO model not loaded. Face tracking is disabled.")
        return None

    try:
        aspect_ratio = parse_aspect_ratio(aspect_ratio_str)
        if not aspect_ratio:
            print(f"Invalid aspect ratio: {aspect_ratio_str}. Using default 16:9")
            aspect_ratio = (16, 9)

        target_ratio = aspect_ratio[0] / aspect_ratio[1]

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Store cropped frames
        cropped_frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends

            frame_count += 1

            # Perform YOLO face detection
            results = yolo_model(frame)

            # Check if a face is detected
            face_detected = False
            for result in results:
                for box in result.boxes:
                    if hasattr(box, 'cls') and len(box.cls) > 0 and int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_detected = True
                        break
                if face_detected:
                    break

            if face_detected:
                # Crop to maintain aspect ratio around the face
                face_width = x2 - x1
                face_height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Adjust crop size to maintain target aspect ratio
                if target_ratio > (face_width / face_height):
                    new_width = int(face_height * target_ratio)
                    new_height = face_height
                else:
                    new_width = face_width
                    new_height = int(face_width / target_ratio)

                # Ensure cropping doesn't exceed frame boundaries
                x1_new = max(0, center_x - new_width // 2)
                x2_new = min(frame_width, center_x + new_width // 2)
                y1_new = max(0, center_y - new_height // 2)
                y2_new = min(frame_height, center_y + new_height // 2)

                # Crop frame
                cropped_frame = frame[y1_new:y2_new, x1_new:x2_new]

                # Resize to target dimensions
                cropped_frame = cv2.resize(cropped_frame, (target_width, target_height))
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_frames.append(cropped_frame)
            else:
                # Center crop when no face detected
                current_ratio = frame_width / frame_height

                if current_ratio > target_ratio:
                    new_width = int(frame_height * target_ratio)
                    start = (frame_width - new_width) // 2
                    cropped_frame = frame[:, start:start+new_width]
                else:
                    new_height = int(frame_width / target_ratio)
                    start = (frame_height - new_height) // 2
                    cropped_frame = frame[start:start+new_height, :]

                # Resize to target dimensions
                cropped_frame = cv2.resize(cropped_frame, (target_width, target_height))
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_frames.append(cropped_frame)

        cap.release()

        if cropped_frames:
            # Create video clip from cropped frames
            cropped_video = ImageSequenceClip(cropped_frames, fps=fps)

            # Add audio from original video
            orig_clip = VideoFileClip(video_path)
            if orig_clip.audio:
                cropped_video = cropped_video.set_audio(orig_clip.audio)

            # Write output
            cropped_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
            return output_path
        else:
            print("No frames processed!")
            return None
    except Exception as e:
        print(f"Error in face tracking: {e}")
        return None

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extracts audio from a video file."""
    try:
        clip = mp.VideoFileClip(video_path)
        if clip.audio:
            clip.audio.write_audiofile(audio_path)
            return audio_path
        return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def generate_captions(video_path):
    """Generates captions using the Whisper STT model."""
    if stt_model is None:
        return "Captions not available. Whisper model not loaded."

    try:
        audio_path = extract_audio(video_path)
        if audio_path:
            result = stt_model.transcribe(audio_path)
            os.remove(audio_path)

            captions = []
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                captions.append(((start_time, end_time), text))

            return captions
        return "No audio detected", None
    except Exception as e:
        print(f"Error generating captions: {e}")
        return "Error generating captions", None

def overlay_captions(video_path, captions, output_path):
    """Overlays captions on the video."""
    try:
        clip = mp.VideoFileClip(video_path)
        original_width, original_height = clip.size

        is_portrait = original_height > original_width
        base_font_size = 30
        subtitle_margin = 300 if is_portrait else 50
        subtitle_width = int(original_width * 0.8)

        def create_subtitle_text_clip(txt):
            return mp.TextClip(
                txt,
                font='Cantarell',
                fontsize=base_font_size * (2 if is_portrait else 1),
                color='white',
                bg_color='black',
                size=(subtitle_width, None),
                method='caption',
                align='center'
            ).on_color(
                color=(0, 0, 0, int(255 * 0.7)),
                col_opacity=0.7
            )

        subtitles = SubtitlesClip(captions, create_subtitle_text_clip)
        subtitle_position = ('center', original_height - subtitle_margin)

        final_clip = mp.CompositeVideoClip([clip, subtitles.set_position(subtitle_position)])
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
    except Exception as e:
        print(f"Error overlaying captions: {e}")
        return None

@app.route("/process_video", methods=["POST"])
def process_video():
    """Processes video based on user selection."""
    data = request.json
    try:
        video_path = data.get("file_path")
        format_type = data.get("format", "mp4")
        aspect_ratio_str = data.get("aspect_ratio", "16:9")
        auto_caption = data.get("auto_caption", False)
        resolution_str = data.get("resolution", "100%")
        use_face_tracking = data.get("use_face_tracking", False)

        # Get original dimensions
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate resolution percentage
        resolution_percentage = float(resolution_str.replace("%", "")) / 100.0

        # Parse aspect ratio and calculate target dimensions
        aspect_ratio = parse_aspect_ratio(aspect_ratio_str)
        if aspect_ratio:
            ratio_w, ratio_h = aspect_ratio
            target_ratio = ratio_w / ratio_h
            new_width = int(original_width * resolution_percentage)
            new_height = int(new_width * ratio_h / ratio_w)
        else:
            new_width = int(original_width * resolution_percentage)
            new_height = int(original_height * resolution_percentage)

        target_width = new_width
        target_height = new_height

        # Generate output path
        output_filename = f"output_{uuid.uuid4().hex}.{format_type}"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

        processed_path = None

        if use_face_tracking and yolo_model is not None:
            processed_path = crop_video_to_face(
                video_path,
                output_path,
                aspect_ratio_str,
                target_width,
                target_height
            )
        else:
            processed_path = resize_video(
                video_path,
                output_path,
                aspect_ratio_str,
                resolution_percentage * 100
            )

        if not processed_path:
            return jsonify({"error": "Failed to process video"}), 500

        result = {"output_path": processed_path}

        # Handle captions
        if auto_caption:
            captions = generate_captions(video_path)
            if captions and isinstance(captions, list):
                captioned_path = processed_path.replace(f".{format_type}", f"_captioned.{format_type}")
                overlay_captions(processed_path, captions, captioned_path)
                result["output_path"] = captioned_path

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/available_features", methods=["GET"])
def available_features():
    """Returns available features status."""
    return jsonify({
        "face_tracking_available": yolo_model is not None,
        "auto_caption_available": stt_model is not None
    })

if __name__ == "__main__":
    app.run(debug=True)