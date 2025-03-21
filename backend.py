
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
        scale_factor = resolution_percentage / 100  # Convert to scale
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

def crop_video_to_face(video_path, output_path, aspect_ratio_str):
    """Crops the video to track faces while maintaining the specified aspect ratio."""
    if yolo_model is None:
        print("YOLO model not loaded. Face tracking is disabled.")
        return None

    try:
        aspect_ratio = parse_aspect_ratio(aspect_ratio_str)
        if not aspect_ratio:
            print(f"Invalid aspect ratio: {aspect_ratio_str}. Using default 16:9")
            aspect_ratio = (16, 9)

        target_width_ratio, target_height_ratio = aspect_ratio
        target_ratio = target_width_ratio / target_height_ratio

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
                    # Only consider boxes that are likely to be faces (class 0)
                    if hasattr(box, 'cls') and len(box.cls) > 0 and int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Face bounding box
                        face_detected = True
                        break  # Process only the first detected face
                if face_detected:
                    break

            if face_detected:
                # Crop to maintain aspect ratio around the face
                face_width = x2 - x1
                face_height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Adjust crop size to maintain the target aspect ratio
                if target_ratio > (face_width / face_height):
                    new_width = int(face_height * target_ratio)
                    new_height = face_height
                else:
                    new_width = face_width
                    new_height = int(face_width / target_ratio)

                # Ensure cropping does not go out of bounds
                x1_new = max(0, center_x - new_width // 2)
                x2_new = min(frame_width, center_x + new_width // 2)
                y1_new = max(0, center_y - new_height // 2)
                y2_new = min(frame_height, center_y + new_height // 2)

                # Crop frame
                cropped_frame = frame[y1_new:y2_new, x1_new:x2_new]

                # Ensure the cropped frame matches the target aspect ratio
                cropped_height, cropped_width = cropped_frame.shape[:2]
                current_ratio = cropped_width / cropped_height

                if abs(current_ratio - target_ratio) > 0.01:  # If not close enough to target ratio
                    if current_ratio > target_ratio:
                        # Too wide, adjust width
                        new_width = int(cropped_height * target_ratio)
                        start = (cropped_width - new_width) // 2
                        cropped_frame = cropped_frame[:, start:start+new_width]
                    else:
                        # Too tall, adjust height
                        new_height = int(cropped_width / target_ratio)
                        start = (cropped_height - new_height) // 2
                        cropped_frame = cropped_frame[start:start+new_height, :]

                # Convert BGR to RGB for MoviePy
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                # Store frame
                cropped_frames.append(cropped_frame)
            else:
                # If no face detected, use the center of the frame
                current_ratio = frame_width / frame_height

                if current_ratio > target_ratio:
                    # Video is wider than target, crop width
                    new_width = int(frame_height * target_ratio)
                    start = (frame_width - new_width) // 2
                    cropped_frame = frame[:, start:start+new_width]
                else:
                    # Video is taller than target, crop height
                    new_height = int(frame_width / target_ratio)
                    start = (frame_height - new_height) // 2
                    cropped_frame = frame[start:start+new_height, :]

                # Convert BGR to RGB for MoviePy
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                # Store frame
                cropped_frames.append(cropped_frame)

        cap.release()

        if cropped_frames:
            # Create video clip from cropped frames
            cropped_video = ImageSequenceClip(cropped_frames, fps=fps)

            # Extract audio from original video
            orig_clip = VideoFileClip(video_path)
            if orig_clip.audio:
                cropped_video = cropped_video.set_audio(orig_clip.audio)

            # Write the final cropped video
            cropped_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

            print(f"Face-tracked video saved as: {output_path}")
            return output_path
        else:
            print("No frames were processed!")
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
            # Transcribe audio using Whisper
            result = stt_model.transcribe(audio_path)
            os.remove(audio_path)

            # Extract segments with timing information
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
    """Overlays captions on the video with uniform size, text wrapping, and semi-transparent background."""
    try:
        clip = mp.VideoFileClip(video_path)
        original_width, original_height = clip.size

        is_portrait = original_height > original_width

        base_font_size = 30  # Base font size for landscape mode
        if is_portrait:
            # Increase font size for portrait mode
            font_size = int(base_font_size * 2)  # Adjust multiplier as needed
            subtitle_margin = 300  # Distance from the bottom of the screen
        else:
            font_size = base_font_size
            subtitle_margin = 50  # Distance from the bottom of the screen

        # Define subtitle width (80% of video width to allow for margins)
        subtitle_width = int(original_width * 0.8)

        # Function to create a TextClip with uniform size and text wrapping
        def create_subtitle_text_clip(txt):
            return mp.TextClip(
                txt,
                font='Cantarell',
                fontsize=font_size,
                color='white',
                bg_color='black',
                size=(subtitle_width, None),  # Fixed width, auto height for wrapping
                method='caption',  # Enables text wrapping
                align='center'  # Center-align text within the subtitle box
            ).on_color(
                color=(0, 0, 0, int(255 * 0.7)),  # 70% opaque black background
                col_opacity=0.7
            )

        # Create subtitles with uniform size and text wrapping
        subtitles = SubtitlesClip(captions, create_subtitle_text_clip)

        # Calculate subtitle position to hover slightly above the bottom
        subtitle_position = ('center', original_height - subtitle_margin)

        # Composite video with subtitles
        final_clip = mp.CompositeVideoClip([clip, subtitles.set_position(subtitle_position)])

        # Write the final video
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
    except Exception as e:
        print(f"Error overlaying captions: {e}")
        return None

@app.route("/process_video", methods=["POST"])
def process_video():
    """Processes video based on user selection and returns the output file."""
    data = request.json
    print("Received request data:", data)  # Add debugging

    try:
        video_path = data.get("file_path")
        format_type = data.get("format", "mp4")
        aspect_ratio = data.get("aspect_ratio", "16:9")
        auto_caption = data.get("auto_caption", False)
        resolution_str = data.get("resolution", "100%")
        use_face_tracking = data.get("use_face_tracking", False)

        # Log the processing parameters
        print(f"Processing video with: format={format_type}, aspect_ratio={aspect_ratio}, "
              f"resolution={resolution_str}, use_face_tracking={use_face_tracking}")

        # Generate unique output filename
        output_filename = f"output_{uuid.uuid4().hex}.{format_type}"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

        processed_path = None

        # Determine if we're using percentage or fixed resolution
        if "%" in resolution_str:
            resolution_percentage = float(resolution_str.replace("%", ""))
        else:
            # Fixed resolution mode, but for simplicity we'll just use 100% and rely on the aspect ratio
            resolution_percentage = 100

        # Process the video based on user selection
        if use_face_tracking and yolo_model is not None:
            print("Using face tracking for processing")
            # Use face tracking for aspect ratio cropping
            processed_path = crop_video_to_face(video_path, output_path, aspect_ratio)
        else:
            print("Using standard resize method")
            # Use traditional resize method
            processed_path = resize_video(video_path, output_path, aspect_ratio, resolution_percentage)

        if not processed_path:
            return jsonify({"error": "Failed to process video"}), 500

        result = {"output_path": processed_path}

        # Generate captions if requested
        if auto_caption:
            captions = generate_captions(video_path)
            if captions and captions != "No audio detected" and captions != "Error generating captions":
                # Overlay captions on the video
                captioned_video_path = processed_path.replace(f".{format_type}", f"_captioned.{format_type}")
                overlay_captions(processed_path, captions, captioned_video_path)
                result["output_path"] = captioned_video_path

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/available_features", methods=["GET"])
def available_features():
    """Returns information about which features are available based on loaded models."""
    return jsonify({
        "face_tracking_available": yolo_model is not None,
        "auto_caption_available": stt_model is not None
    })

if __name__ == "__main__":
    app.run(debug=True)
