from flask import Flask, request, jsonify
import os
import cv2
import moviepy.editor as mp
from transformers import pipeline
import uuid
import re

app = Flask(__name__)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Initialize speech recognition model
try:
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
except:
    stt_pipeline = None
    print("Warning: Speech recognition model could not be loaded. Auto-captioning will be disabled.")

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
        match = re.match(r'(\d+):(\d+)', aspect_ratio_str)
        if match:
            aspect_w, aspect_h = int(match.group(1)), int(match.group(2))
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
    """Generates captions using a speech-to-text model and creates an SRT file."""
    if stt_pipeline is None:
        return "Captions not available. Speech recognition model not loaded."

    try:
        audio_path = extract_audio(video_path)
        if audio_path:
            result = stt_pipeline(audio_path)
            os.remove(audio_path)  # Clean up temp file
            captions = result["text"]

            # Create an SRT file with basic timing (assuming 30s per sentence)
            srt_path = video_path.replace(".mp4", ".srt").replace(".mkv", ".srt")
            with open(srt_path, "w") as f:
                f.write("1\n00:00:00,000 --> 00:00:30,000\n" + captions)

            return captions, srt_path
        return "No audio detected", None
    except Exception as e:
        print(f"Error generating captions: {e}")
        return "Error generating captions", None

@app.route("/process_video", methods=["POST"])
def process_video():
    """Processes video based on user selection and returns the output file."""
    data = request.json

    try:
        video_path = data["file_path"]
        format_type = data["format"]
        aspect_ratio = data["aspect_ratio"]
        auto_caption = data["auto_caption"]
        resolution_str = data["resolution"]

        resolution_percentage = None  # Ensure variable is defined

        if "%" in resolution_str:
            resolution_percentage = float(resolution_str.replace("%", ""))
            custom_mode = False  # Percentage-based scaling
        else:
            custom_mode = True  # Fixed resolution mode
            width, height = map(int, resolution_str.split("x"))

        # Generate unique output filename
        output_filename = f"output_{uuid.uuid4().hex}.{format_type}"
        output_path = os.path.join("output", output_filename)

        # Resize the video
        if not custom_mode:
            processed_path = resize_video(video_path, output_path, aspect_ratio, resolution_percentage)
        else:
            processed_path = resize_video(video_path, output_path, aspect_ratio, 100)  # Default to 100% scaling if custom resolution

        if processed_path:
            result = {"output_path": processed_path}

            # Generate captions if requested
            if auto_caption:
                captions, srt_path = generate_captions(video_path)
                result["captions"] = captions
                if srt_path:
                    result["captions_path"] = srt_path

            return jsonify(result)
        else:
            return jsonify({"error": "Failed to process video"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
