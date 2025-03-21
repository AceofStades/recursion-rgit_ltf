from flask import Flask, request, jsonify
import os
import cv2
import moviepy.editor as mp
from transformers import pipeline
import uuid

app = Flask(__name__)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Initialize speech recognition model
try:
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
except:
    stt_pipeline = None
    print("Warning: Speech recognition model could not be loaded. Auto-captioning will be disabled.")

# Resolution mappings
RESOLUTION_MAP = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160)
}

# Platform aspect ratio settings
ASPECT_RATIOS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "4:5": (4, 5)
}

@app.route("/get_resolution", methods=["POST"])
def get_resolution():
    data = request.json
    video_path = data["file_path"]

    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if width >= 3840 and height >= 2160:
            resolution = "4K"
        elif width >= 1920 and height >= 1080:
            resolution = "1080p"
        else:
            resolution = "720p"

        return jsonify({"resolution": resolution})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def resize_video(video_path, output_path, aspect_ratio, resolution):
    """Resize video to the target aspect ratio and resolution"""
    try:
        # Parse resolution
        if resolution in RESOLUTION_MAP:
            target_width, target_height = RESOLUTION_MAP[resolution]
        else:
            target_width, target_height = RESOLUTION_MAP["1080p"]  # Default to 1080p

        # Parse aspect ratio
        if aspect_ratio in ASPECT_RATIOS:
            ratio_w, ratio_h = ASPECT_RATIOS[aspect_ratio]
        else:
            ratio_w, ratio_h = ASPECT_RATIOS["16:9"]  # Default to 16:9

        # Adjust dimensions to match aspect ratio
        if ratio_w / ratio_h > 1:  # Wider than tall
            new_height = target_height
            new_width = int(new_height * ratio_w / ratio_h)
            if new_width > target_width:
                new_width = target_width
                new_height = int(new_width * ratio_h / ratio_w)
        else:  # Taller than wide
            new_width = target_width
            new_height = int(new_width * ratio_h / ratio_w)
            if new_height > target_height:
                new_height = target_height
                new_width = int(new_height * ratio_w / ratio_h)

        # Create video clip and resize
        clip = mp.VideoFileClip(video_path)
        resized_clip = clip.resize(newsize=(new_width, new_height))
        resized_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
    except Exception as e:
        print(f"Error resizing video: {e}")
        return None

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from video file"""
    try:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def generate_captions(video_path):
    """Generate captions using speech recognition"""
    if stt_pipeline is None:
        return "Captions not available. Speech recognition model not loaded."

    try:
        audio_path = extract_audio(video_path)
        if audio_path:
            result = stt_pipeline(audio_path)
            os.remove(audio_path)  # Clean up temp file
            return result["text"]
        return "No audio detected"
    except Exception as e:
        print(f"Error generating captions: {e}")
        return "Error generating captions"

@app.route("/process_video", methods=["POST"])
def process_video():
    data = request.json

    try:
        video_path = data["file_path"]
        platform = data["platform"]
        video_type = data["video_type"]
        format_type = data["format"]
        resolution = data["resolution"]
        aspect_ratio = data["aspect_ratio"]
        auto_caption = data["auto_caption"]

        # Generate unique output filename
        output_filename = f"output_{uuid.uuid4().hex}.{format_type}"
        output_path = os.path.join("output", output_filename)

        # Process video
        if resize_video(video_path, output_path, aspect_ratio, resolution):
            result = {"output_path": output_path}

            # Generate captions if requested
            if auto_caption:
                captions = generate_captions(video_path)
                result["captions"] = captions

                # Create SRT file
                srt_path = output_path.replace(f".{format_type}", ".srt")
                with open(srt_path, "w") as f:
                    f.write("1\n00:00:00,000 --> 00:00:30,000\n" + captions)

                result["captions_path"] = srt_path

            return jsonify(result)
        else:
            return jsonify({"error": "Failed to process video"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
