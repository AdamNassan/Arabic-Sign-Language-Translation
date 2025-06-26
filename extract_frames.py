import cv2
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames(video_path, output_path, resize_shape=(224, 224)):
    """Extract frames from a video and save them as images."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize_shape)
        cv2.imwrite(os.path.join(output_path, f'frame_{frame_count:04d}.jpg'), frame)
        frame_count += 1

    cap.release()
    return frame_count

def process_signer(base_path, output_base_path, signer, modality='color'):
    """Process all videos for a single signer."""
    logs = []
    for split in ['test', 'train']:
        signer_path = os.path.join(base_path, modality, signer, split)
        if not os.path.exists(signer_path):
            logs.append(f"❌ Path does not exist: {signer_path}")
            continue

        for gesture_folder in sorted(os.listdir(signer_path)):
            gesture_path = os.path.join(signer_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            for video_file in os.listdir(gesture_path):
                if not video_file.endswith('.mp4'):
                    continue

                video_path = os.path.join(gesture_path, video_file)
                output_path = os.path.join(
                    output_base_path,
                    modality,
                    signer,
                    split,
                    gesture_folder,
                    video_file.replace('.mp4', '')
                )

                try:
                    count = extract_frames(video_path, output_path)
                    logs.append(f"✅ {video_path}: {count} frames")
                except Exception as e:
                    logs.append(f"❌ {video_path}: Error: {e}")
    return logs

def process_dataset_by_signer(base_path, output_base_path, modality='color', max_workers=6):
    """Use one thread per signer for parallel processing."""
    signers = ['01', '02', '03', '04', '05', '06']
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for signer in signers:
            futures.append(executor.submit(process_signer, base_path, output_base_path, signer, modality))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing signers"):
            logs = future.result()
            for log in logs:
                print(log)

# Run it
dataset_path = "./Dataset"  # Update this to your dataset path
output_path = "./frames"    # Where to save the extracted frames

print("🚀 Processing each signer in parallel...")
process_dataset_by_signer(dataset_path, output_path, modality='color', max_workers=6)
