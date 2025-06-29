import os
import json
import cv2
import pandas as pd

FEEDBACK_FILE = "feedback.json"
YOLO_RESULTS_DIR = "/yolo_results"
OUTPUT_DIR = "/feedback_frames"
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
        print("video path", video_path)
        csv_path = os.path.splitext(video_path)[0] + CSV_SUFFIX
        print("csv", csv_path)

        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            print(f"⚠️ No video or log {video_name}")
            continue

        print(f"Processing {video_name}")
        df_log = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(video_path)

        if not os.path.exists(OUTPUT_DIR):
            print(f"⚠️ Output directory does not exist: {OUTPUT_DIR}")
            return
        try:
            test_path = os.path.join(OUTPUT_DIR, "test_write.txt")
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
        except Exception as e:
            print(f"❌ Cannot write to {OUTPUT_DIR}: {e}")
            return

        for entry in entries:
            frame_idx = int(entry["frame"])
            track_id = int(entry.get("track_id", -1))
            fb_type = entry["feedback"]
            comment = entry.get("comment", "")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Cannot read frame {frame_idx} in {video_name}")
                continue

            row = df_log[(df_log["frame"] == frame_idx) &
                         (df_log["track_id"] == track_id)]
            if row.empty:
                print(
                    f"⚠️ Could not find track_id={track_id} at frame {frame_idx}")
                continue

            r = row.iloc[0]
            x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
            h, w, _ = frame.shape
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                print(
                    f"⚠️ Invalid bbox ({x1},{y1},{x2},{y2}) at frame {frame_idx}")
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"⚠️ Empty crop at frame {frame_idx} id {track_id}")
                continue

            save_dir = os.path.join(
                OUTPUT_DIR, fb_type, r["class_name"], r.get("sub_label", "unknown"))
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{os.path.splitext(video_name)[0]}_f{frame_idx}_id{track_id}.jpg"
            save_path = os.path.join(save_dir, filename)

            success = cv2.imwrite(save_path, crop)
            if success:
                print(f"✅ Extracted: {save_path}")
            else:
                print(f"❌ Failed to write image to {save_path}")

        cap.release()

    print("🎉 Finished extracting images from feedback.")


def clean_up_feedback_frames(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.startswith("."):
            os.remove(file_path)
            print(f"⚠️ Removed invalid file: {file_path}")
        elif not os.path.isdir(file_path) and not filename.lower().endswith((".jpg", ".png")):
            os.remove(file_path)
            print(f"⚠️ Removed non-image file: {file_path}")


clean_up_feedback_frames(OUTPUT_DIR)
extract_all_feedback_frames()
