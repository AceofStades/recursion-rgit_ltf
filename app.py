import streamlit as st
import requests
import os

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

    # Platform selection
    platform = st.selectbox("Select Platform", ["YouTube", "TikTok", "Instagram"])

    # Video type selection
    video_type = st.radio("Select Video Type", ["Long", "Short"])

    # Set default aspect ratio based on video type
    default_aspect_ratio = "9:16" if video_type == "Short" else "16:9"
    aspect_ratio = st.selectbox("Select Aspect Ratio", ["16:9", "9:16", "4:5"], index=["16:9", "9:16", "4:5"].index(default_aspect_ratio))

    # Video format selection
    format = st.selectbox("Select Format", ["mp4", "mkv", "avi"])

    # Auto-captions option
    auto_caption = st.checkbox("Enable Auto Captions")

    # Request backend to get video resolution
    if "video_resolution" not in st.session_state:
        try:
            res_response = requests.post(f"{BACKEND_URL}/get_resolution", json={"file_path": video_path})
            if res_response.status_code == 200:
                st.session_state["video_resolution"] = res_response.json()["resolution"]
            else:
                st.session_state["video_resolution"] = "1080p"  # Fallback default
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend server. Make sure it's running.")
            st.session_state["video_resolution"] = "1080p"  # Fallback default

    # Resolution selection (capped at original resolution)
    available_resolutions = ["720p", "1080p", "4K"]
    max_res_index = available_resolutions.index(st.session_state["video_resolution"]) if st.session_state["video_resolution"] in available_resolutions else 1
    resolution = st.selectbox("Select Resolution", available_resolutions[: max_res_index + 1])

    # Process button
    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            try:
                process_payload = {
                    "file_path": video_path,
                    "platform": platform.lower(),
                    "video_type": video_type.lower(),
                    "format": format.lower(),
                    "resolution": resolution.lower(),
                    "aspect_ratio": aspect_ratio.lower(),
                    "auto_caption": auto_caption
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
