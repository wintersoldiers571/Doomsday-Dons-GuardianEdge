#!/usr/bin/env python3
"""
GuardianEdge - Mobile Edge Camera Node
======================================
Designed to run on Android via Pydroid 3, Termux, or secondary devices.
Captures live camera frames and streams them over USB Data Cable / ADB / Wi-Fi
to the GuardianEdge Host Server running on your laptop.

QUICK CONNECTION GUIDE:
-----------------------
Option A (USB Cable + ADB Port Forwarding - RECOMMENDED for zero-latency):
   1. Connect phone to laptop with USB cable (enable USB Debugging in Developer Options).
   2. On your laptop terminal, run:
      adb reverse tcp:5000 tcp:5000
   3. In this script, leave LAPTOP_HOST = "http://127.0.0.1:5000".

Option B (USB Tethering):
   1. Turn on "USB Tethering" in Phone Settings -> Hotspot/Tethering.
   2. Check your laptop's gateway/tether IP (usually 192.168.42.x).
   3. Set LAPTOP_HOST = "http://192.168.42.x:5000".

Option C (Wi-Fi Local Network):
   1. Connect laptop and phone to the same Wi-Fi.
   2. Set LAPTOP_HOST to your laptop's local IP (e.g. "http://192.168.1.15:5000").
"""

import sys
import time
import argparse
import requests

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

# Default configuration
DEFAULT_HOST = "http://127.0.0.1:5000"
UPLOAD_ENDPOINT = "/api/frame_upload"
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
JPEG_QUALITY = 75
TARGET_FPS = 20

def parse_args():
    parser = argparse.ArgumentParser(description="GuardianEdge Mobile Camera Streamer")
    parser.add_argument("--host", default=DEFAULT_HOST, help="GuardianEdge laptop server URL (e.g. http://127.0.0.1:5000)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (0 for rear/default, 1 for front)")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Target FPS to stream")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY, help="JPEG compression quality (1-100)")
    return parser.parse_args()

def draw_hud(frame, fps, frame_count):
    """Draws on-device status overlay on the local preview frame."""
    hud_color = (0, 255, 200) # Cyberpunk cyan
    cv2.putText(frame, "GUARDIAN-EDGE :: CAMERA NODE [ONLINE]", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
    cv2.putText(frame, f"STREAM FPS: {fps:.1f} | FRAMES SENT: {frame_count}", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame

def main():
    args = parse_args()
    upload_url = args.host.rstrip("/") + UPLOAD_ENDPOINT

    print("=" * 60)
    print(" [GUARDIAN-EDGE] MOBILE CAMERA STREAMER")
    print("=" * 60)
    print(f" Target Server URL : {upload_url}")
    print(f" Camera Index      : {args.cam}")
    print(f" Target Frame Rate : {args.fps} FPS")
    print(f" Compression Qual  : {args.quality}%")
    print("=" * 60)

    # Initialize Camera
    print(f"[*] Initializing camera device ({args.cam})...")
    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print(f"[!] Warning: Camera index {args.cam} could not be opened.")
        print("[*] Trying fallback camera index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[!] Camera unavailable. Generating synthetic diagnostic test patterns...")
        mock_mode = True
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        mock_mode = False
        print("[+] Camera initialized successfully.")

    # Frame transmission loop
    session = requests.Session()
    frame_interval = 1.0 / max(1, args.fps)
    frame_count = 0
    last_fps_calc = time.time()
    fps = 0.0
    consecutive_errors = 0

    print("[*] Starting frame upload pipeline. Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle_start = time.time()

            if not mock_mode:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[!] Failed to grab frame from camera. Retrying...")
                    time.sleep(0.1)
                    continue
            else:
                # Generate synthetic test frame with simulated traffic sign
                import numpy as np
                frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                frame[:] = (30, 30, 35) # Dark road background
                # Draw simulated sign circle
                center = (TARGET_WIDTH // 2, TARGET_HEIGHT // 2)
                cv2.circle(frame, center, 80, (0, 0, 220), -1) # Red outer
                cv2.circle(frame, center, 65, (255, 255, 255), -1) # White inner
                cv2.putText(frame, "50", (center[0] - 30, center[1] + 15),
                            cv2.FONT_HERSHEY_BOLD, 1.2, (0, 0, 0), 3)

            # Resize to target resolution if needed
            if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
            success, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                continue

            frame_bytes = encoded_img.tobytes()

            # Upload frame to laptop GuardianEdge host
            try:
                resp = session.post(
                    upload_url,
                    data=frame_bytes,
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=0.8
                )

                if resp.status_code == 200:
                    frame_count += 1
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors % 20 == 0:
                        print(f"[!] Server returned HTTP {resp.status_code}")
            except requests.exceptions.RequestException:
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 30 == 0:
                    print(f"[!] Waiting for GuardianEdge host server at {upload_url}...")
                time.sleep(0.2)

            # Calculate FPS
            now = time.time()
            if now - last_fps_calc >= 1.0:
                fps = frame_count / (now - last_fps_calc)
                frame_count = 0
                last_fps_calc = now
                sys.stdout.write(f"\r[STREAMING] FPS: {fps:.1f} | Server: {resp.status_code if 'resp' in locals() and resp is not None else 'N/A'} | Latency: {(time.time() - cycle_start)*1000:.1f}ms   ")
                sys.stdout.flush()

            # Sleep to maintain target FPS
            elapsed = time.time() - cycle_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[*] Stopping camera streaming node...")
    finally:
        if not mock_mode and cap is not None:
            cap.release()
        session.close()
        print("[+] Stream closed cleanly.")

if __name__ == "__main__":
    main()
