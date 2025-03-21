import streamlit as st
import requests
import os
import re
import math

BACKEND_URL = "http://127.0.0.1:5000"

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

st.title("AI Video Resizer & Format Converter")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mkv", "avi"])

if uploaded_file:
    video_path = os.path.join("uploads", uploaded_file.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state["video_path"] = video_path
    st.success("Video uploaded successfully!")

    # Platform selection with Custom option
    platform = st.selectbox("Select Platform", ["YouTube", "TikTok", "Instagram", "Custom"])
    custom_mode = platform == "Custom"

    if not custom_mode:
        video_type = st.radio("Select Video Type", ["Long", "Short"])
        default_aspect_ratio = "9:16" if video_type == "Short" else "16:9"
        predefined_ratios = ["16:9", "9:16", "4:5"]

        aspect_ratio = st.selectbox(
            "Select Aspect Ratio",
            predefined_ratios + ["Custom"],
            index=predefined_ratios.index(default_aspect_ratio) if default_aspect_ratio in predefined_ratios else 0
        )

        if aspect_ratio == "Custom":
            custom_mode = True
    else:
        video_type = "custom"
        aspect_ratio = "Custom"

    if custom_mode:
        st.markdown("### Custom Settings")
        custom_ratio_w = st.number_input("Width ratio", min_value=1, max_value=100, value=16)
        custom_ratio_h = st.number_input("Height ratio", min_value=1, max_value=100, value=9)
        aspect_ratio = f"{custom_ratio_w}:{custom_ratio_h}"

    format = st.selectbox("Select Format", ["mp4", "mkv", "avi"])
    auto_caption = st.checkbox("Enable Auto Captions")

    # Get video resolution from backend
    if "video_resolution" not in st.session_state or "original_dimensions" not in st.session_state:
        try:
            res_response = requests.post(f"{BACKEND_URL}/get_resolution", json={"file_path": video_path})
            if res_response.status_code == 200:
                res_data = res_response.json()
                st.session_state["video_resolution"] = res_data["resolution"]
                st.session_state["original_dimensions"] = (res_data["width"], res_data["height"])
            else:
                st.session_state["video_resolution"] = "1080p"
                st.session_state["original_dimensions"] = (1920, 1080)
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend server. Make sure it's running.")
            st.session_state["video_resolution"] = "1080p"
            st.session_state["original_dimensions"] = (1920, 1080)

    original_width, original_height = st.session_state["original_dimensions"]
    st.info(f"Original video dimensions: {original_width}x{original_height}")

    # Compute aspect ratio values
    if ":" in aspect_ratio:
        ratio_w, ratio_h = map(int, aspect_ratio.split(":"))
        new_height = math.floor(original_width * (ratio_h / ratio_w))
    else:
        new_height = original_height

    # Percentage-based resolution selection
    st.markdown("### Select Output Resolution")
    resolution_percentage = st.slider("Resolution Percentage", 10, 100, 100, step=5)

    # Compute new resolution based on percentage
    new_width = math.floor((original_width * resolution_percentage) / 100)
    new_height = math.floor((original_height * resolution_percentage) / 100)

    # Ensure the new resolution follows the selected aspect ratio
    if ":" in aspect_ratio:
        new_height = math.floor(new_width * (ratio_h / ratio_w))

    # Important resolution markers
    important_resolutions = {
        "4K": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "480p": (854, 480),
        "360p": (640, 360)
    }

    markers = []
    for label, (w, h) in important_resolutions.items():
        if w <= original_width and h <= original_height:
            percent = int((w / original_width) * 100)
            markers.append(f"{label} ~ {percent}%")

    # Display resolution preview
    st.info(f"New Resolution: {new_width}x{new_height} ({resolution_percentage}%)")
    st.markdown(f"### Key Points: {', '.join(markers)}")

    # Process button
    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            try:
                process_payload = {
                    "file_path": video_path,
                    "platform": platform.lower(),
                    "video_type": video_type.lower() if not custom_mode else "custom",
                    "format": format.lower(),
                    "resolution": f"{new_width}x{new_height}",
                    "aspect_ratio": aspect_ratio.lower(),
                    "auto_caption": auto_caption,
                    "custom_mode": custom_mode
                }

                process_response = requests.post(f"{BACKEND_URL}/process_video", json=process_payload)

                if process_response.status_code == 200:
                    output_path = process_response.json()["output_path"]
                    st.success("Video Processing Complete!")
                    st.video(output_path)

                    with open(output_path, "rb") as file:
                        st.download_button("Download Processed Video", file, file_name=os.path.basename(output_path))
                else:
                    st.error(f"Processing failed! Error: {process_response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend server. Make sure it's running.")
