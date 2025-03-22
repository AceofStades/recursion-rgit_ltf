import streamlit as st
import requests
import os
import re
import math

BACKEND_URL = "http://127.0.0.1:5000"

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

st.title("AI Video Resizer & Format Converter")

# Check available features from backend
try:
    features_response = requests.get(f"{BACKEND_URL}/available_features")
    features = features_response.json() if features_response.status_code == 200 else {
        "face_tracking_available": False,
        "auto_caption_available": False
    }
except:
    features = {
        "face_tracking_available": False,
        "auto_caption_available": False
    }

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

    # Add face tracking option if available
    use_face_tracking = False
    if features["face_tracking_available"]:
        use_face_tracking = st.checkbox("Enable Smart Face Tracking", value=True,
                                        help="Intelligently crop video to follow faces (recommended for portrait videos)")

    # Show auto captioning option if available
    auto_caption = False
    if features["auto_caption_available"]:
        auto_caption = st.checkbox("Enable Auto Captions")

        # Caption customization options
        st.markdown("### Caption Customization")
        caption_font = st.selectbox("Caption Font", ["Arial", "Cantarell", "Verdana", "Times New Roman"], index=1)
        caption_size = st.number_input("Caption Size", min_value=10, max_value=100, value=30)
        caption_color = st.color_picker("Caption Color", "#FFFFFF")
        caption_bg_color = st.color_picker("Caption Background Color", "#000000")
        caption_bg_opacity = st.slider("Caption Background Opacity", 0.0, 1.0, 0.7, step=0.1)

        # Caption preview
        st.markdown("### Caption Preview")
        st.markdown(
            f"""
            <div style="
                font-family: {caption_font};
                font-size: {caption_size}px;
                color: {caption_color};
                background-color: {caption_bg_color};
                opacity: {caption_bg_opacity};
                padding: 10px;
                text-align: center;
                margin: 10px 0;
            ">
            Sample Caption Text
            </div>
            """,
            unsafe_allow_html=True
        )

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
    if markers:
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
                    "resolution": f"{resolution_percentage}%",  # Send percentage instead of exact dimensions
                    "aspect_ratio": aspect_ratio.lower(),
                    "auto_caption": auto_caption,
                    "use_face_tracking": use_face_tracking,
                    "custom_mode": custom_mode,
                    "caption_font": caption_font if features["auto_caption_available"] else "Cantarell",
                    "caption_size": caption_size if features["auto_caption_available"] else 30,
                    "caption_color": caption_color if features["auto_caption_available"] else "#FFFFFF",
                    "caption_bg_color": caption_bg_color if features["auto_caption_available"] else "#000000",
                    "caption_bg_opacity": caption_bg_opacity if features["auto_caption_available"] else 0.7
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
