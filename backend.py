from flask import Flask, request, jsonify
import os
import cv2
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
import uuid
import re
import whisper

app = Flask(_name_)

os.makedirs("output", exist_ok=True)

# Load Whisper model
try:
    stt_model = whisper.load_model("base")
except Exception as e:
    stt_model = None
    print(f"Warning: Whisper model could not be loaded. Auto-captioning will be disabled. Error: {e}")

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
        resized_clip.write_videofile(output_path, codec="hevc", audio_codec="aac")

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
                captions = generate_captions(video_path)
                if captions:
                    # Overlay captions on the video
                    captioned_video_path = processed_path.replace(f".{format_type}", f"_captioned.{format_type}")
                    overlay_captions(processed_path, captions, captioned_video_path)
                    result["output_path"] = captioned_video_path

            return jsonify(result)
        else:
            return jsonify({"error": "Failed to process video"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if _name_ == "_main_":
    app.run(debug=True)
