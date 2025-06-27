# 📦 Dispatch Management Dashboard with Enhanced UI + YOLO Tracking
import tempfile
import cv2
import pandas as pd
import streamlit as st
import json
import os
import sys
from datetime import datetime
from glob import glob
from modules.detect_and_track import run_yolo_tracking
from modules.extract_feedback_frames import extract_all_feedback_frames
from streamlit_option_menu import option_menu


# ---------- SETTINGS ----------
FEEDBACK_FILE = "feedback.json"
EXPORT_CSV_FILE = "feedback_export.csv"
YOLO_OUTPUT_DIR = "yolo_results"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Dispatch Manager", page_icon="📦", layout="wide")

# ---------- CUSTOM BACKGROUND & FONT ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #fffdf9;
    }
    .main {
        background-color: #fffaf2;
    }
    .stApp {
        background-color: #fffaf2;
    }
    h2, h3, h4, .stMarkdown h2 {
        color: #e65100;
        font-weight: 700;
    }
    .eatlab-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 0;
        border-bottom: 1px solid #ffcc80;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR MENU ----------
with st.sidebar:
    st.markdown("""
        <div class="eatlab-logo">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Orange_circle.svg/120px-Orange_circle.svg.png" width="10" height="10" style="margin-right:6px;">
            <span style='font-size:24px; font-weight:800; color:#e65100;'>EAT</span>
            <span style='font-size:24px; font-weight:800; color:#ff9800;'>LAB</span>
            <span style='font-size:28px; color:#ff9800;'>.</span>
        </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Detection & Tracking", "Submit Feedback",
                 "Feedback Log", "Analytics"],
        icons=["camera-video", "envelope", "file-earmark-text", "bar-chart"],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#fff3e0"},
            "icon": {"color": "#ff6f00", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#ffe0b2",
            },
            "nav-link-selected": {"background-color": "#ffe0b2"},
        },
    )

# ---------- Heading ----------

# Central heading for all pages
if selected in ["Detection & Tracking", "Submit Feedback",
                "Feedback Log", "Analytics"]:
    st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <h1 style='font-size: 42px; color: #e65100; font-weight: 800;'>Dispatch Monitoring System EATLAB</h1>
        </div>
    """, unsafe_allow_html=True)

# ---------- PAGE: Tracking video ----------
if selected == "Detection & Tracking":
    st.markdown("""
        <h2 style='color:#e65100; font-weight:700; border-bottom:2px solid #e65100;'>📹 Track Video</h2>
    """, unsafe_allow_html=True)
    st.caption(
        "Upload a video, adjust parameters, run YOLO tracking, and view results and logs.")

    uploaded_track_video = st.file_uploader(
        "📽️ Upload video for tracking (mp4)", type=["mp4"])

    with st.expander("⚙️ Model parameter settings"):
        col1, col2 = st.columns(2)
        with col1:
            conf_thres = st.slider(
                "🎯 Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
            iou_thres = st.slider("🧩 IoU Threshold", 0.0, 1.0, 0.5, 0.05)
        with col2:
            max_det = st.slider("🔢 Max Detections per Frame", 10, 300, 100, 10)
            class_filter = st.text_input(
                "🎯 Detect classes (e.g., 0 for 'tray', 1 for 'dish')", "")

    run_button = st.button("🚀 Start Tracking")

    if run_button and uploaded_track_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpin:
            tmpin.write(uploaded_track_video.read())
            input_video_path = tmpin.name

        try:
            progress = st.progress(0, text="🔄 Initializing model...")
            progress.progress(20, text="📦 Running YOLO tracking...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tracked_{timestamp}.mp4"
            output_path = os.path.join(YOLO_OUTPUT_DIR, output_filename)

            output_video_path, log_df = run_yolo_tracking(
                input_video_path,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                classes=class_filter,
                output_dir=YOLO_OUTPUT_DIR
            )

            if os.path.exists(output_video_path):
                os.rename(output_video_path, output_path)
                output_video_path = output_path

            progress.progress(100, text="✅ Done!")

            if os.path.exists(output_video_path):
                st.success("🎉 Tracking completed successfully!")
                with open(output_video_path, "rb") as f:
                    st.video(f.read())

                if log_df is not None:
                    st.subheader("📄 Tracking results by frame")
                    st.dataframe(log_df)

                    csv = log_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV", data=csv, file_name="tracking_log.csv", mime="text/csv")
            else:
                st.error("❌ Result video not found.")

        except Exception as e:
            st.error(f"❌ Error during tracking: {str(e)}")

    # 📁 Recently tracked videos
    st.markdown("## 📂 Recently Tracked Videos")
    tracked_videos = sorted(
        glob(os.path.join(YOLO_OUTPUT_DIR, "tracked_*.mp4")), reverse=True)

    if tracked_videos:
        selected_video = st.selectbox(
            "🎬 Select a tracked video", tracked_videos)
        with open(selected_video, "rb") as f:
            st.video(f.read())
        st.download_button("⬇️ Download Video", data=open(selected_video, "rb").read(),
                           file_name=os.path.basename(selected_video), mime="video/mp4")
    else:
        st.info("📭 No tracked videos available.")


# ---------- PAGE: Submit Feedback ----------
elif selected == "Submit Feedback":
    st.markdown("""
        <h2 style='color:#e65100; font-weight:700; border-bottom:2px solid #e65100;'>📤 Submit Feedback</h2>
    """, unsafe_allow_html=True)
    st.caption(
        "Select a tracked video and specify the timestamp to give feedback.")

    # 📂 Get list of tracked videos
    tracked_videos = sorted(
        glob(os.path.join(YOLO_OUTPUT_DIR, "tracked_*.mp4")))
    video_dict = {os.path.basename(v): v for v in tracked_videos}

    if not tracked_videos:
        st.info("📭 No tracked videos available.")
    else:
        selected_video_name = st.selectbox(
            "🎬 Select a tracked video", list(video_dict.keys()))
        selected_video_path = video_dict[selected_video_name]

        # Display video
        with open(selected_video_path, "rb") as f:
            st.video(f.read())

        cap = cv2.VideoCapture(selected_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        st.success(
            f"🎯 Video duration ~{duration:.2f}s — {fps:.2f} FPS — {total_frames} frames")

        # Select timestamp for feedback
        time_sec = st.number_input(
            "⏱️ Enter timestamp in video (seconds)",
            min_value=0.0, max_value=duration,
            value=0.0, step=0.1, format="%.2f"
        )
        estimated_frame = int(time_sec * fps)
        st.write(f"📍 Estimated frame: `{estimated_frame}`")

        # Show frame at selected time
        if st.checkbox("Preview frame at timestamp"):
            cap = cv2.VideoCapture(selected_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, estimated_frame)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.image(
                    frame[:, :, ::-1],
                    caption=f"Frame at {time_sec:.2f}s",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Unable to read frame.")

        # Feedback inputs
        col1, col2 = st.columns(2)
        with col1:
            track_id = st.number_input("🆔 Track ID", min_value=0, step=1)
        with col2:
            feedback_type = st.selectbox(
                "📌 Feedback Type",
                ["Correct", "Wrong ID", "Missing track",
                 "ID switched", "Other"]
            )

        comment = st.text_area(
            "💬 Additional Comments",
            placeholder="Provide detailed description if necessary..."
        )

        if st.button("📤 Submit Feedback"):
            new_entry = {
                "video": selected_video_name,
                "timestamp_sec": round(time_sec, 2),
                "frame": estimated_frame,
                "track_id": int(track_id),
                "feedback": feedback_type,
                "comment": comment
            }

            if os.path.exists(FEEDBACK_FILE):
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    feedback_data = json.load(f)
            else:
                feedback_data = []

            feedback_data.append(new_entry)

            with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(feedback_data, f, indent=4, ensure_ascii=False)

            st.success("✅ Feedback successfully recorded!")


# ---------- PAGE: Feedback Log ----------
elif selected == "Feedback Log":
    st.markdown("""
        <h2 style='color:#e65100; font-weight:700; border-bottom:2px solid #e65100;'>📄 Feedback Log</h2>
    """, unsafe_allow_html=True)

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)

        if feedback_data:
            df = pd.DataFrame(feedback_data)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv,
                               file_name=EXPORT_CSV_FILE, mime="text/csv")

            # 👉 Image extraction button
            st.markdown("---")
            st.subheader("📸 Extract Images from Feedback")
            if st.button("🚀 Start Extraction"):
                extract_all_feedback_frames()
                st.success(
                    "✅ Images extracted to `feedback_frames/` directory.")
        else:
            st.info("📭 No feedback data available.")
    else:
        st.info("📭 No feedback data available.")

    # 📸 Display extracted feedback images
    ROOT_FEEDBACK_IMAGE_DIR = "feedback_frames"
    if os.path.exists(ROOT_FEEDBACK_IMAGE_DIR):
        st.markdown("## 🖼️ Extracted Feedback Images")
        feedback_types = sorted(os.listdir(ROOT_FEEDBACK_IMAGE_DIR))
        selected_type = st.selectbox(
            "🔍 Select feedback type to view images", feedback_types)

        selected_dir = os.path.join(ROOT_FEEDBACK_IMAGE_DIR, selected_type)
        image_files = [os.path.join(selected_dir, f)
                       for f in os.listdir(selected_dir)
                       if f.lower().endswith((".jpg", ".png"))]

        if image_files:
            cols = st.columns(3)
            for i, img_path in enumerate(image_files):
                with cols[i % 3]:
                    st.image(
                        img_path,
                        caption=os.path.basename(img_path),
                        use_container_width=True
                    )
        else:
            st.info("📭 No images available in this feedback category.")

# ---------- PAGE: Feedback Statistics ----------
elif selected == "Statistics":
    st.markdown("""
        <h2 style='color:#e65100; font-weight:700; border-bottom:2px solid #e65100;'>📊 Feedback Statistics</h2>
    """, unsafe_allow_html=True)

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            feedback_data = json.load(f)
        df = pd.DataFrame(feedback_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**By Feedback Type**")
            st.bar_chart(df["feedback"].value_counts())

        with col2:
            st.markdown("**By Video**")
            st.bar_chart(df["video"].value_counts())
    else:
        st.info("📭 No data available for statistics.")
