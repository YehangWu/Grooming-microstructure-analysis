# -*- coding: utf-8 -*-
"""
CSV (frame-wise labels) -> TXT (event list) conversion script.
- Input: CSV (frame-wise labels), supports:
    * Common delimiters (comma/tab/semicolon), auto-detected
    * With/without header:
        ["", background, groom1..groom5] or ["frame", background, groom1..groom5]
        or no header: first column is frame index, followed by background, groom1..5
    * Values can be 0/1 or probabilities; if background is dominant, treat as background,
      otherwise, take the maximum value from groom1..5 as the frame label
- Video: used to extract FPS and total frame count (extra frames will be truncated)
- Output: TXT (tab-separated):
    ID  grooming  start_time  end_time  duration
- Interval merging: left-closed, right-open (end_time = (last_frame + 1) / FPS)
Dependencies: pandas, opencv-python
"""

import os
import cv2
import pandas as pd
import numpy as np


# =======================
# Helper functions
# =======================

def get_video_info(video_path: str):
    """
    Get the total frame count and FPS of a video.

    Args:
    - video_path (str): Path to the video file.

    Returns:
    - total_frames (int): Total number of frames in the video.
    - fps (float): Frames per second (FPS) of the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Detected abnormal FPS={fps}, please check the video file: {video_path}")
    return total_frames, fps


def _normalize_columns(cols):
    """
    Normalize column names to lowercase and strip whitespace.

    Args:
    - cols (list): List of column names.

    Returns:
    - out (list): Normalized list of column names.
    """
    out = []
    for c in cols:
        if c is None:
            out.append("")
        else:
            out.append(str(c).strip().lower())
    return out


def _read_csv_flex(csv_path: str, header: bool = True) -> pd.DataFrame:
    """
    Attempt to automatically detect delimiter and encoding for CSV reading.

    Args:
    - csv_path (str): Path to the CSV file.
    - header (bool): Whether the CSV file has a header row.

    Returns:
    - pd.DataFrame: Dataframe containing the CSV data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Try to auto-detect delimiter
    read_kwargs = dict(sep=None, engine="python")

    # Try different encodings
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    for enc in encodings:
        try:
            if header:
                return pd.read_csv(csv_path, encoding=enc, header=0, **read_kwargs)
            else:
                return pd.read_csv(csv_path, encoding=enc, header=None, **read_kwargs)
        except Exception:
            continue
    # If no encoding worked, try UTF-8
    if header:
        return pd.read_csv(csv_path, encoding="utf-8", header=0, **read_kwargs)
    else:
        return pd.read_csv(csv_path, encoding="utf-8", header=None, **read_kwargs)


def read_csv_labels(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and return a DataFrame containing:
        'frame' (int), 'background' (float), 'groom1'..'groom5' (float)

    Handles cases with or without headers.

    Args:
    - csv_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame with event-level labels.
    """
    # Try to read with header
    try:
        df = _read_csv_flex(csv_path, header=True)
        cols = _normalize_columns(df.columns.tolist())
    except Exception:
        df = None
        cols = []

    def has_all(expected, pool):
        return all(x in pool for x in expected)

    expected = ["frame", "background", "groom1", "groom2", "groom3", "groom4", "groom5"]

    if df is not None and has_all(expected, cols):
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        use = expected
    else:
        # Try without header and assume the first 7 columns are frame, background, groom1..5
        df = _read_csv_flex(csv_path, header=False)
        if df.shape[1] < 7:
            raise ValueError(f"CSV has insufficient columns (expected ≥7, found {df.shape[1]}): {csv_path}")
        sub = df.iloc[:, :7].copy()
        sub.columns = ["frame", "background", "groom1", "groom2", "groom3", "groom4", "groom5"]
        df = sub
        use = ["frame", "background", "groom1", "groom2", "groom3", "groom4", "groom5"]

    # Clean up data types
    df = df[use].copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int)
    for c in ["background", "groom1", "groom2", "groom3", "groom4", "groom5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Remove negative frames and sort
    df = df[df["frame"] >= 0].sort_values("frame", kind="mergesort").reset_index(drop=True)
    return df


def frames_to_events(df: pd.DataFrame, total_frames: int, fps: float):
    """
    Convert frame-wise labels to event list:
    - If background is dominant, treat as background (ignore)
    - Otherwise, take the maximum label from groom1..5 for the frame label
    - Merge consecutive frames with the same label into a single event.

    Args:
    - df (pd.DataFrame): DataFrame containing frame-wise labels.
    - total_frames (int): Total number of frames in the video.
    - fps (float): Frames per second (FPS) of the video.

    Returns:
    - List[dict]: List of events with grooming label, start time, end time, and duration.
    """
    # Truncate to video length
    df = df[df["frame"] < total_frames].copy()
    if df.empty:
        return []

    gcols = ["groom1", "groom2", "groom3", "groom4", "groom5"]

    def pick_label(row):
        vals = [float(row[g]) for g in gcols]
        bg = float(row["background"])
        if bg >= max(vals):  # Background dominant: treat as background
            return 0
        # Otherwise, take the maximum groom label
        argmax = int(max(range(5), key=lambda i: vals[i]))
        return argmax + 1

    df["label"] = df.apply(pick_label, axis=1)
    df = df[df["label"] > 0].copy()
    if df.empty:
        return []

    # Merge consecutive frames with the same label
    events = []
    cur_label = None
    cur_start = None
    prev_frame = None

    for _, row in df.iterrows():
        f = int(row["frame"])
        lab = int(row["label"])
        if cur_label is None:
            cur_label, cur_start, prev_frame = lab, f, f
            continue
        if lab == cur_label and f == prev_frame + 1:
            prev_frame = f
        else:
            events.append((cur_label, cur_start, prev_frame))
            cur_label, cur_start, prev_frame = lab, f, f
    events.append((cur_label, cur_start, prev_frame))

    # Convert to time (left-closed, right-open: end_time = (end_frame + 1) / fps)
    out = []
    for (lab, s, e) in events:
        start_time = s / fps
        end_time = (e + 1) / fps
        duration = end_time - start_time
        out.append({
            "grooming": lab,
            "start": start_time,
            "end": end_time,
            "dur": duration
        })
    return out


def write_events_txt(events, txt_path: str):
    """
    Write events to a TXT file (tab-separated), rounded to 3 decimal places.

    Args:
    - events (list): List of event dictionaries.
    - txt_path (str): Path to output the event list.
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("序号\tgrooming\t开始时间\t结束时间\t时长\t\n")
        for i, ev in enumerate(events, 1):
            f.write(f"{i}\t{ev['grooming']}\t{ev['start']:.3f}\t{ev['end']:.3f}\t{ev['dur']:.3f}\t\n")


# =======================
# Single File & Batch Processing
# =======================

def process_single_pair(video_path: str, csv_path: str, out_txt_path: str):
    """Process a single video and CSV, output event list to TXT."""
    print(f"Processing: {os.path.basename(csv_path)}")
    total_frames, fps = get_video_info(video_path)
    df = read_csv_labels(csv_path)
    events = frames_to_events(df, total_frames, fps)
    write_events_txt(events, out_txt_path)
    print(
        f"✓ Generated: {os.path.basename(out_txt_path)} | Events: {len(events)} | FPS: {fps} | Total frames: {total_frames}")


def batch_process(video_folder: str, csv_folder: str, output_folder: str):
    """
    Batch process: Match video files (.mp4) with corresponding CSV files.
    - video/*.mp4
    - csv/*.csv
    - Output to txt/*.txt
    """
    os.makedirs(output_folder, exist_ok=True)
    videos = [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]
    if not videos:
        print(f"No MP4 videos found in {video_folder}.")
        return

    print(f"Found {len(videos)} videos, starting to match CSVs...")
    for vf in videos:
        base = os.path.splitext(vf)[0]
        csv_name = f"{base}_predictions.csv"
        csv_path = os.path.join(csv_folder, csv_name)
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_name} (skipping {vf})")
            continue

        video_path = os.path.join(video_folder, vf)
        out_txt = os.path.join(output_folder, f"{base}.txt")

        try:
            process_single_pair(video_path, csv_path, out_txt)
        except Exception as e:
            print(f"Error processing {vf}: {e}")

    print("Batch processing complete.")


# =======================
# Entry point
# =======================
if __name__ == "__main__":
    # Modify with actual directories
    video_folder = r"video_folder_path"
    csv_folder = r"csv_folder_path"
    output_folder = r"txt_output_path"

    try:
        batch_process(video_folder, csv_folder, output_folder)
    except Exception as e:
        print(f"Error during execution: {e}")
