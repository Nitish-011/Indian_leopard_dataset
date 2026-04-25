# This script is optimized for nvidia gpus contact me or update the script for google collab
# I have added the background image function but it is not working currently 
#creator- Nitish Joshi                     ( 50 % AI as well ;) )

import os
import cv2
import torch
import yt_dlp
import imagehash
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1  CONFIGURATION: SCALE & SPECIES PURITY (update these to get new images :) )
SEARCH_QUERIES = [
    "leopard crossing road india",
    "leopard in uttarakhand village cctv",
    "corbett national park leopard",
    "kumaon garhwal leopard",
    "night vision leopard india"
]

# Increased to 150 to find more hidden gems in the same categories
MAX_VIDEOS_PER_QUERY = 150

OUTPUT_DIR = "Leopard_Dataset"
POS_DIR = os.path.join(OUTPUT_DIR, "leopard_full_frames")
NEG_DIR = os.path.join(OUTPUT_DIR, "background")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "processed_videos.txt")
METADATA_FILE = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

for d in [POS_DIR, NEG_DIR]: os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RTX Engine Active. Filtering out Snow Leopards & Black Panthers. Hardware: {DEVICE.upper()}")

# 2 MODELS
yolo_model = YOLO('yolov8l.pt') 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 3 UTILITIES
def get_unique_urls():
    processed = []
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            processed = f.read().splitlines()
    
    all_urls = []
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for q in SEARCH_QUERIES:
            print(f"Deep Scanning: {q}")
            try:
                results = ydl.extract_info(f"ytsearch{MAX_VIDEOS_PER_QUERY}:{q}", download=False)
                for entry in results.get('entries', []):
                    if entry and entry.get('url') and entry['url'] not in processed:
                        all_urls.append(entry['url'])
            except: continue
    return list(set(all_urls))

# 4 MAIN PIPELINE
def run_production_pipeline():
    urls = get_unique_urls()
    print(f"Found {len(urls)} videos. Starting high-precision extraction...")
    
    metadata = []
    saved_hashes = []
    img_count = 0

    for url in tqdm(urls, desc="Extracting"):
        video_id = url.split('=')[-1] if '=' in url else url[-10:]
        temp_file = f"temp_{video_id}.mp4"
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best', 
            'outtmpl': temp_file,
            'quiet': True, 'no_warnings': True, 'ignoreerrors': True 
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if ydl.download([url]) != 0: continue
        except: continue

        cap = cv2.VideoCapture(temp_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_idx = 0
        last_saved_time = -5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            curr_time_sec = frame_idx / fps
            frame_idx += 1 

            # Accuracy check every 2 seconds
            if frame_idx % (fps * 2) == 0:
                results = yolo_model(frame, classes=[15, 16], conf=0.5, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        if curr_time_sec - last_saved_time < 5: continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                        if crop.size == 0: continue
                        
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        
                        # MULTI-SPECIES DISCRIMINATION
                        labels = [
                            "a spotted yellow indian leopard",       # [0] Target
                            "a thick grey white snow leopard",       # [1] REJECT
                            "a solid black panther",                 # [2] REJECT
                            "a tiger with stripes"                   # [3] REJECT
                        ]
                        
                        inputs = clip_processor(text=labels, images=pil_crop, return_tensors="pt", padding=True).to(DEVICE)
                        with torch.no_grad():
                            probs = clip_model(**inputs).logits_per_image.softmax(dim=1)
                        
                        # High confidence in Target (0.96) if you want more data reduce the confidence
                        # But remeber it will also increase garbage in dataset
                        is_target = probs[0][0] > 0.96
                        not_snow = probs[0][0] > probs[0][1]
                        not_black = probs[0][0] > probs[0][2]

                        if is_target and not_snow and not_black:
                            curr_hash = imagehash.phash(pil_crop)
                            if any(curr_hash - h < 10 for h in saved_hashes): continue
                            
                            saved_hashes.append(curr_hash)
                            img_name = f"leo_{video_id}_{frame_idx}.jpg"
                            save_path = os.path.join(POS_DIR, img_name)
                            
                            cv2.imwrite(save_path, frame)
                            last_saved_time = curr_time_sec
                            
                            metadata.append({"filename": img_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "url": url})
                            img_count += 1
                            print(f"Verified P. p. fusca #{img_count} | REJECTED Snow/Black", end="\r")

        cap.release()
        if os.path.exists(temp_file): os.remove(temp_file)
        with open(CHECKPOINT_FILE, 'a') as f: f.write(url + "\n")
        if metadata: pd.DataFrame(metadata).to_csv(METADATA_FILE, index=False)

    print(f"\nExtraction Complete. {img_count} high-purity frames saved.")

if __name__ == "__main__":
    run_production_pipeline()