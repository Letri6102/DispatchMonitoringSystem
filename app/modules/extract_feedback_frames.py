import os
import json
import cv2
import pandas as pd

FEEDBACK_FILE = "feedback.json"
YOLO_RESULTS_DIR = "yolo_results"
OUTPUT_DIR = "feedback_frames"
CSV_SUFFIX = ".csv"


def extract_all_feedback_frames():
    if not os.path.exists(FEEDBACK_FILE):
        print("Cannot find feedback.json")
        return

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        feedback_data = json.load(f)

    if not feedback_data:
        print("No response")
        return

    grouped_feedback = {}
    for entry in feedback_data:
        vid = entry["video"]
        grouped_feedback.setdefault(vid, []).append(entry)

    for video_name, entries in grouped_feedback.items():
        video_path = os.path.join(YOLO_RESULTS_DIR, video_name)
        csv_path = os.path.splitext(video_path)[0] + CSV_SUFFIX

        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            print(f" No video or log {video_name}")
            continue

        print(f"Processing {video_name}")
        df_log = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(video_path)

        for entry in entries:
            frame_idx = int(entry["frame"])
            track_id = int(entry.get("track_id", -1))
            fb_type = entry["feedback"]
            comment = entry.get("comment", "")

            # Move to the required frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(
                    f"⚠️ Cannot read frame {frame_idx} in {video_name}")
                continue

            # Find bounding box from log
            row = df_log[(df_log["frame"] == frame_idx) &
                         (df_log["track_id"] == track_id)]
            if row.empty:
                print(
                    f"⚠️ Could not find track_id={track_id} at frame {frame_idx}")
                continue

            r = row.iloc[0]
            x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
            class_name = r["class_name"]
            sub_label = r.get("sub_label", "unknown")
            crop = frame[y1:y2, x1:x2]

            # Directory to save
            save_dir = os.path.join(OUTPUT_DIR, fb_type, class_name, sub_label)
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{os.path.splitext(video_name)[0]}_f{frame_idx}_id{track_id}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, crop)
            print(f"Extracted: {save_path}")

        cap.release()

    print("🎉 Finished extracting images from feedback.")
