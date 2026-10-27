#!/usr/bin/env python3
"""
GuardianEdge - Edge-AI Behavioral Watchdog & Self-Healing SOC Host Server
========================================================================
Runs on the host laptop. Serves the Cyberpunk Security Operations Center (SOC)
dashboard, processes live camera frames from the USB-connected mobile phone,
executes Dual-Layer Watchdog monitoring, handles live attack simulations,
and orchestrates the 6-Step Autonomous Self-Healing Response Loop.
"""

import os
import sys
import time
import json
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

from watchdog_engine import DualLayerWatchdog, GTSRB_CLASSES

app = Flask(__name__)

# Initialize Core Watchdog & Self-Healing Engine
watchdog = DualLayerWatchdog(model_save_path="gtsrb_clean_model.npz")

# Video Feed Management
class VideoFeedManager:
    """Manages video sources: Phone USB Upload, Laptop Webcam, or Synthetic Road Sim."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active_source = "mobile" # "mobile", "webcam", "simulation"
        self.latest_frame = None
        self.last_mobile_frame_ts = 0.0
        self.webcam_cap = None
        self.sim_tick = 0
        self.sim_sign_idx = 14 # Default Stop sign
        self.sim_signs = [14, 2, 13, 12, 38, 1, 17, 25] # Stop, 50, Yield, Priority, Keep Right, 30, No Entry, Road Work
        
        # Start background simulator generator
        self._init_simulation_stream()

    def update_mobile_frame(self, frame_bytes):
        """Receives JPEG bytes from camera_mobile.py over USB."""
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            with self.lock:
                self.latest_frame = frame
                self.last_mobile_frame_ts = time.time()
                self.active_source = "mobile"
            return True
        return False

    def set_source(self, source_name):
        with self.lock:
            self.active_source = source_name
            if source_name == "webcam" and self.webcam_cap is None:
                try:
                    self.webcam_cap = cv2.VideoCapture(0)
                except Exception:
                    self.webcam_cap = None

    def _init_simulation_stream(self):
        """Background thread generating realistic GTSRB road environment frames."""
        def sim_worker():
            while True:
                time.sleep(0.04) # ~25 FPS
                self.sim_tick += 1
                
                # Switch simulated sign every ~120 frames (approx 5 sec)
                if self.sim_tick % 120 == 0:
                    curr_idx = self.sim_signs.index(self.sim_sign_idx)
                    self.sim_sign_idx = self.sim_signs[(curr_idx + 1) % len(self.sim_signs)]

                # If source is simulation, or mobile hasn't pushed a frame in 3 seconds
                now = time.time()
                if self.active_source == "simulation" or (self.active_source == "mobile" and (now - self.last_mobile_frame_ts > 3.0)):
                    frame = self._generate_simulated_road_frame(self.sim_sign_idx, self.sim_tick)
                    with self.lock:
                        self.latest_frame = frame
                elif self.active_source == "webcam" and self.webcam_cap is not None:
                    ret, w_frame = self.webcam_cap.read()
                    if ret and w_frame is not None:
                        with self.lock:
                            self.latest_frame = w_frame

        threading.Thread(target=sim_worker, daemon=True).start()

    def _generate_simulated_road_frame(self, sign_id, tick):
        """Creates a dynamic driving roadway scene with approaching GTSRB traffic sign."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Horizon & Road Perspective
        horizon_y = 190
        frame[:horizon_y, :] = (45, 30, 20) # Night / Dusk sky
        frame[horizon_y:, :] = (35, 35, 38) # Asphalt road
        
        # Lane divider dash animation
        dash_offset = (tick * 8) % 60
        for y in range(horizon_y + 10 + dash_offset, h, 60):
            # Center dash
            cv2.line(frame, (w//2, y), (w//2, min(h, y + 30)), (0, 220, 255), 3)

        # Draw roadside post and approaching GTSRB traffic sign
        # Center of sign in HUD
        sign_center = (w // 2, h // 2)
        radius = 70
        
        # Sign Pole
        cv2.rectangle(frame, (sign_center[0] - 6, sign_center[1]), (sign_center[0] + 6, h - 40), (120, 120, 125), -1)
        
        # Draw distinctive traffic sign based on sign_id
        if sign_id == 14: # Stop
            # Red Octagon / Circle
            cv2.circle(frame, sign_center, radius, (0, 0, 220), -1)
            cv2.circle(frame, sign_center, radius - 8, (255, 255, 255), 3)
            cv2.putText(frame, "STOP", (sign_center[0] - 50, sign_center[1] + 15),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 3)
        elif sign_id == 2: # Speed 50
            cv2.circle(frame, sign_center, radius, (0, 0, 230), -1) # Red border
            cv2.circle(frame, sign_center, radius - 12, (255, 255, 255), -1) # White center
            cv2.putText(frame, "50", (sign_center[0] - 32, sign_center[1] + 18),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.4, (0, 0, 0), 4)
        elif sign_id == 13: # Yield
            # Inverted triangle
            pt1 = (sign_center[0], sign_center[1] + radius)
            pt2 = (sign_center[0] - radius, sign_center[1] - radius + 15)
            pt3 = (sign_center[0] + radius, sign_center[1] - radius + 15)
            pts = np.array([pt1, pt2, pt3], np.int32)
            cv2.drawContours(frame, [pts], 0, (0, 0, 220), -1)
            # Inner white triangle
            inner_pt1 = (sign_center[0], sign_center[1] + radius - 15)
            inner_pt2 = (sign_center[0] - radius + 18, sign_center[1] - radius + 25)
            inner_pt3 = (sign_center[0] + radius - 18, sign_center[1] - radius + 25)
            cv2.drawContours(frame, [np.array([inner_pt1, inner_pt2, inner_pt3], np.int32)], 0, (255, 255, 255), -1)
        elif sign_id == 12: # Priority Road (Yellow Diamond)
            pts = np.array([
                (sign_center[0], sign_center[1] - radius),
                (sign_center[0] + radius, sign_center[1]),
                (sign_center[0], sign_center[1] + radius),
                (sign_center[0] - radius, sign_center[1])
            ], np.int32)
            cv2.drawContours(frame, [pts], 0, (255, 255, 255), -1)
            inner_r = radius - 14
            inner_pts = np.array([
                (sign_center[0], sign_center[1] - inner_r),
                (sign_center[0] + inner_r, sign_center[1]),
                (sign_center[0], sign_center[1] + inner_r),
                (sign_center[0] - inner_r, sign_center[1])
            ], np.int32)
            cv2.drawContours(frame, [inner_pts], 0, (0, 215, 255), -1) # Yellow
        elif sign_id == 38: # Keep Right (Blue Circle with white arrow)
            cv2.circle(frame, sign_center, radius, (220, 80, 20), -1) # Blue
            cv2.circle(frame, sign_center, radius - 6, (255, 255, 255), 2)
            # White arrow pointing down-right
            cv2.arrowedLine(frame, (sign_center[0] - 25, sign_center[1] - 25),
                            (sign_center[0] + 25, sign_center[1] + 25), (255, 255, 255), 7, tipLength=0.35)
        else: # Generic speed limit
            cv2.circle(frame, sign_center, radius, (0, 0, 230), -1)
            cv2.circle(frame, sign_center, radius - 12, (255, 255, 255), -1)
            cv2.putText(frame, "30", (sign_center[0] - 32, sign_center[1] + 18),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.4, (0, 0, 0), 4)

        return frame

    def get_current_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            # Default standby frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "INITIALIZING VIDEO ENGINE...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
            return frame

video_mgr = VideoFeedManager()

# --- HTTP Endpoints ---

@app.route("/")
def index():
    """Serves the main Cyberpunk SOC Dashboard."""
    return render_template("index.html")

@app.route("/api/frame_upload", methods=["POST"])
def frame_upload():
    """Endpoint for camera_mobile.py running on Android/Pydroid 3 over USB."""
    try:
        frame_bytes = request.get_data()
        if frame_bytes:
            success = video_mgr.update_mobile_frame(frame_bytes)
            if success:
                return jsonify({"status": "OK", "received_bytes": len(frame_bytes)}), 200
        return jsonify({"status": "ERROR", "message": "Invalid frame data"}), 400
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500

@app.route("/api/set_source", methods=["POST"])
def set_source():
    """Selects video source: 'mobile', 'webcam', or 'simulation'."""
    data = request.get_json() or {}
    source = data.get("source", "simulation")
    video_mgr.set_source(source)
    watchdog.log_event("SOURCE_CHANGE", f"Video feed source switched to [{source.upper()}].", "INFO")
    return jsonify({"status": "OK", "active_source": source})

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns full real-time telemetry snapshot for dashboard polling."""
    snapshot = watchdog.get_telemetry_snapshot()
    # Add active camera source info
    time_since_mobile = round(time.time() - video_mgr.last_mobile_frame_ts, 1)
    snapshot['video_source'] = {
        'active': video_mgr.active_source,
        'mobile_connected': time_since_mobile < 3.5,
        'mobile_last_seen_sec': time_since_mobile
    }
    return jsonify(snapshot)

# --- Attack Simulation Endpoints ---

@app.route("/api/attack/weight_corruption", methods=["POST"])
def attack_weight_corruption():
    """Simulates Hostile RAM / Weight Corruption Attack."""
    watchdog.inject_weight_corruption(severity=0.65)
    return jsonify({"status": "ATTACK_TRIGGERED", "type": "WEIGHT_CORRUPTION"})

@app.route("/api/attack/adversarial_noise", methods=["POST"])
def attack_adversarial_noise():
    """Simulates Adversarial Noise Perturbation on live camera frames."""
    watchdog.inject_adversarial_noise(intensity=0.55)
    return jsonify({"status": "ATTACK_TRIGGERED", "type": "ADVERSARIAL_NOISE"})

@app.route("/api/attack/clear", methods=["POST"])
def attack_clear():
    """Deactivates adversarial noise injection."""
    watchdog.clear_adversarial_noise()
    return jsonify({"status": "ATTACKS_CLEARED"})

@app.route("/api/heal/rollback", methods=["POST"])
def manual_rollback():
    """Triggers the 6-Step Autonomous Self-Healing Incident Response Loop."""
    watchdog.trigger_emergency_rollback()
    return jsonify({"status": "SELF_HEALING_INITIALIZED", "loop": "6_STEP_INCIDENT_RESPONSE"})


# --- MJPEG Live Stream Generator with Real-time HUD ---

def generate_video_stream():
    """Encodes processed frames with HUD overlays as MJPEG stream."""
    while True:
        frame = video_mgr.get_current_frame()
        # Run inference and watchdog processing
        result = watchdog.process_frame(frame)
        
        annotated_frame = result['frame_bgr']
        x1, y1, x2, y2 = result['bbox']
        label = result['label']
        conf = result['confidence']
        locked = result['actuator_locked']
        
        # Draw Cyberpunk HUD Corner Brackets & Target Box
        box_color = (0, 0, 255) if (locked or watchdog.system_status != "SECURE_NOMINAL") else (0, 255, 200)
        
        # Bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Corner brackets
        corner_len = 20
        # Top-left
        cv2.line(annotated_frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
        cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
        # Top-right
        cv2.line(annotated_frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
        cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
        # Bottom-left
        cv2.line(annotated_frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
        cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 3)
        # Bottom-right
        cv2.line(annotated_frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
        cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)
        
        # Target crosshair in center of ROI
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.line(annotated_frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(annotated_frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

        # Top HUD Banner: Prediction and Confidence
        cv2.rectangle(annotated_frame, (x1, y1 - 45), (x1 + 320, y1), (20, 20, 25), -1)
        cv2.rectangle(annotated_frame, (x1, y1 - 45), (x1 + 320, y1), box_color, 1)
        
        conf_pct = int(conf * 100)
        cv2.putText(annotated_frame, f"SIGN: {label[:20]}", (x1 + 8, y1 - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"CONF: {conf_pct}% | L2 FLUTTER: {watchdog.layer2_flutter_rate*100:.0f}%",
                    (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

        # Bottom Actuator Status Bar
        if locked:
            cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0] - 35), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 180), -1)
            cv2.putText(annotated_frame, "⚠️ SAFETY OVERRIDE ACTIVE: ACTUATOR LOCKED (EMERGENCY_HOLD)",
                        (30, annotated_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0] - 30), (annotated_frame.shape[1], annotated_frame.shape[0]), (15, 25, 20), -1)
            cv2.putText(annotated_frame, "ACTUATOR: PASS-THROUGH NOMINAL | BEHAVIORAL WATCHDOG ACTIVE",
                        (30, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033) # Cap streaming to ~30 FPS

@app.route("/video_feed")
def video_feed():
    """MJPEG Live Video Stream route."""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print(" [GUARDIAN-EDGE] SOC CONTROL HUB & BEHAVIORAL WATCHDOG")
    print("=" * 70)
    print(f" Web Dashboard URL     : http://127.0.0.1:{port}")
    print(f" Phone Upload Endpoint : http://127.0.0.1:{port}/api/frame_upload")
    print(" GTSRB Golden Baseline : 5 Immutable Reference Signatures Loaded")
    print(" Dual-Layer Watchdog   : ACTIVE (L1 Golden Checks + L2 Stability)")
    print(" Autonomous Healing    : 6-Step Closed Loop Response Standby")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
