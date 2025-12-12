#!/usr/bin/env python3
# eye_collector_v3.py

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime

# ================= KONFIGURASI GLOBAL =================
FILE_NAME = "eye_dataset_wheelchair_continuous.csv"
CAMERA_INDEX = 2

# --- Default Config (Bisa diedit di Runtime/Halaman 3) ---
class AppConfig:
    def __init__(self):
        self.SMOOTH_ALPHA = 0.70      # Smoothing (0.0 - 1.0)
        self.DISTANCE_K = 3.0         # Konstanta Kalibrasi Jarak
        self.BLINK_THRESHOLD = 0.30   # Batas EAR untuk kedip
        self.BRIGHTNESS_LOW = 60      # Batas cahaya
        self.GOOD_MIN_CM = 25.0       # Jarak Min
        self.GOOD_MAX_CM = 70.0       # Jarak Max
        self.REF_DIST_CM = 50.0       # Jarak Referensi User
        self.BURST_LIMIT = 50         # JUMLAH DATA SEKALI TEKAN (Fitur Baru)

config = AppConfig()

# --- Mapping Tombol (SKEMA CONTINUOUS DRIVE) ---
LABEL_KEYS = {
    ord('m'): "CENTER",   # MAJU
    ord('l'): "LEFT",     # BELOK KIRI
    ord('r'): "RIGHT",    # BELOK KANAN
    ord('d'): "DOWN",     # MUNDUR
    ord('b'): "BLINK",    # STOP/SWITCH
    ord('1'): "WINK_L",   
    ord('2'): "WINK_R"    
}

# --- MediaPipe Indices ---
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473
L_OUTER = 33
L_INNER = 133
R_OUTER = 263
R_INNER = 362
LEFT_EAR_IDXS  = [33, 159, 158, 133, 145, 153]
RIGHT_EAR_IDXS = [362, 386, 387, 263, 374, 380]
ALL_EYE_LANDMARKS = [33, 133, 159, 158, 157, 155, 145, 153, 154, 144, 
                     362, 263, 386, 387, 388, 390, 374, 380, 381, 373]

# ================= UTILITIES =================

def init_csv(path):
    if os.path.exists(path): return
    header = [
        "timestamp", 
        "dx_rel", "dy_rel",       
        "ear_left", "ear_right", 
        "blink_l", "blink_r",
        "dist_cm", "brightness", 
        "label"
    ]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)

class SmoothValue:
    def __init__(self):
        self.val = None
    def update(self, new_val):
        if self.val is None: self.val = new_val
        else: self.val = config.SMOOTH_ALPHA * new_val + (1 - config.SMOOTH_ALPHA) * self.val
        return self.val

class EyeNormalizer:
    def normalize(self, frame, landmarks, width, height):
        pts = np.array([(landmarks[i].x * width, landmarks[i].y * height) 
                       for i in ALL_EYE_LANDMARKS], dtype=np.float32)
        x_min, y_min = np.min(pts, axis=0).astype(int)
        x_max, y_max = np.max(pts, axis=0).astype(int)
        
        pad_x, pad_y = 30, 20
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(width, x_max + pad_x)
        y_max = min(height, y_max + pad_y)
        
        if x_max <= x_min or y_max <= y_min:
             return np.zeros((100, 200, 3), dtype=np.uint8)

        eye_img = frame[y_min:y_max, x_min:x_max]
        if eye_img.size == 0: return np.zeros((100, 200, 3), dtype=np.uint8)
        
        target_w = 200
        aspect = eye_img.shape[0] / eye_img.shape[1]
        target_h = int(target_w * aspect)
        return cv2.resize(eye_img, (target_w, target_h))

def calculate_ear(landmarks, idxs, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs])
    v = (np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])) / 2.0
    hor = np.linalg.norm(pts[0]-pts[3]) + 1e-6
    return v / hor

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)
    status = "OK" if avg > config.BRIGHTNESS_LOW else "GELAP"
    col = (0, 255, 0) if avg > config.BRIGHTNESS_LOW else (0, 0, 255)
    return avg, status, col

def get_gaze_vector(lm, w, h, idx_inner, idx_outer, idx_iris):
    p_in = np.array([lm[idx_inner].x * w, lm[idx_inner].y * h])
    p_out = np.array([lm[idx_outer].x * w, lm[idx_outer].y * h])
    p_iris = np.array([lm[idx_iris].x * w, lm[idx_iris].y * h])
    
    eye_center = (p_in + p_out) / 2.0
    eye_width = np.linalg.norm(p_out - p_in) + 1e-6
    
    dx = (p_iris[0] - eye_center[0]) / eye_width
    dy = (p_iris[1] - eye_center[1]) / eye_width
    return dx, dy, eye_center, eye_width

# ================= RENDERING PAGES =================

def draw_page_1_camera(img, data, eye_view, recording_state):
    """Panel 1: View Kamera & Dashboard Visual"""
    h, w, _ = img.shape
    
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (320, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    def txt(t, y, c=(255,255,255), s=0.5):
        cv2.putText(img, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1)

    txt("PANEL 1: KAMERA", 30, (0, 255, 255), 0.6)
    
    # --- INDIKATOR RECORDING (BURST) ---
    is_rec, rec_count, rec_label = recording_state
    if is_rec:
        cv2.circle(img, (280, 30), 10, (0, 0, 255), -1) # Red Dot
        txt(f"REC: {rec_label}", 60, (0, 0, 255), 0.7)
        txt(f"Sisa: {rec_count}", 80, (0, 255, 255), 0.6)
    else:
        dist_val = data['dist']
        dist_ok = data['dist_ok']
        c_dist = (0, 255, 0) if dist_ok else (0, 0, 255)
        txt(f"Jarak: {dist_val:.1f} cm {'(OK)' if dist_ok else '(BURUK)'}", 50, c_dist, 0.5)

    if eye_view is not None:
        eh, ew, _ = eye_view.shape
        if eh > 100: eh = 100
        if 80+eh < h and 20+ew < w:
            img[100:100+eh, 20:20+ew] = eye_view[:eh, :ew]
            cv2.rectangle(img, (20, 100), (20+ew, 100+eh), (255,255,255), 1)

    y_base = 230
    
    bl_L = data['blink_l']
    bl_R = data['blink_r']
    col_L = (0,0,255) if bl_L else (0,255,0)
    col_R = (0,0,255) if bl_R else (0,255,0)
    
    txt(f"Mata Kiri : {'KEDIP' if bl_L else 'BUKA'}", y_base, col_L)
    txt(f"Mata Kanan: {'KEDIP' if bl_R else 'BUKA'}", y_base+20, col_R)

    # Gaze Bar X
    dx = data.get('dx_rel', 0)
    bar_x = int(160 + (dx * 300))
    cv2.rectangle(img, (50, y_base+45), (270, y_base+55), (100,100,100), 1)
    cv2.circle(img, (bar_x, y_base+50), 6, (0,255,255), -1)
    txt(f"X (Ki/Ka): {dx:.3f}", y_base+70)

    # Gaze Bar Y
    dy = data.get('dy_rel', 0)
    bar_y = int(160 + (dy * 300))
    cv2.rectangle(img, (50, y_base+95), (270, y_base+105), (100,100,100), 1)
    cv2.circle(img, (bar_y, y_base+100), 6, (255,0,255), -1)
    txt(f"Y (At/Bw): {dy:.3f}", y_base+120)

    # Kalibrasi Info
    if data['baseline'] is None:
        txt("PERLU KALIBRASI (Tekan 'C')", y_base+150, (0,0,255), 0.7)
    else:
        txt("Kalibrasi OK", y_base+150, (0,255,0))

    txt(f"Log: {data['msg']}", h-20, data['msg_col'])

def draw_page_2_details(img, data, raw_data):
    """Panel 2: Semua Data Numerik Detail"""
    img[:] = (20, 20, 20) 
    
    def row(label, val, y, col=(0, 255, 0)):
        cv2.putText(img, f"{label:<20}: {val}", (30, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    cv2.putText(img, "PANEL 2: DATA DETAIL", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    y = 80
    gap = 25
    
    row("Raw Vector X", f"{raw_data['dx_raw']:.5f}", y); y+=gap
    row("Raw Vector Y", f"{raw_data['dy_raw']:.5f}", y); y+=gap
    
    y += 10
    row("Smooth X", f"{raw_data['dx_s']:.5f}", y); y+=gap
    row("Smooth Y", f"{raw_data['dy_s']:.5f}", y); y+=gap
    
    base_str = "None" if data['baseline'] is None else f"[{data['baseline'][0]:.3f}, {data['baseline'][1]:.3f}]"
    row("Baseline Offset", base_str, y, (0, 255, 255)); y+=gap
    
    row("FINAL Rel X", f"{data['dx_rel']:.5f}", y, (0, 255, 0) if abs(data['dx_rel'])<0.1 else (0,0,255)); y+=gap
    row("FINAL Rel Y", f"{data['dy_rel']:.5f}", y); y+=gap

    y += 10
    thresh = config.BLINK_THRESHOLD
    earL = data['ear_l']
    earR = data['ear_r']
    colL = (0,0,255) if earL < thresh else (0,255,0)
    colR = (0,0,255) if earR < thresh else (0,255,0)
    
    row("EAR Left", f"{earL:.4f}", y, colL); y+=gap
    row("EAR Right", f"{earR:.4f}", y, colR); y+=gap
    
    dist = data['dist']
    row("Est. Distance", f"{dist:.1f} cm", y, (0,255,0) if data['dist_ok'] else (0,0,255)); y+=gap
    row("Brightness", f"{int(raw_data['light_val'])}", y); y+=gap

def draw_page_3_tuning(img, selected_idx):
    """Panel 3: Form Input Tuning Parameter"""
    img[:] = (40, 40, 40)
    
    params = [
        ("REF DISTANCE (cm)", "REF_DIST_CM", 1.0),
        ("DISTANCE K (Auto)", "DISTANCE_K", 0.1),
        ("BURST LIMIT", "BURST_LIMIT", 10), # Config baru
        ("BLINK THRESHOLD", "BLINK_THRESHOLD", 0.01),
        ("SMOOTH ALPHA", "SMOOTH_ALPHA", 0.05),
        ("MIN DIST (cm)", "GOOD_MIN_CM", 1.0),
        ("MAX DIST (cm)", "GOOD_MAX_CM", 1.0),
        ("BRIGHTNESS LIM", "BRIGHTNESS_LOW", 5.0)
    ]
    
    cv2.putText(img, "PANEL 3: TUNING", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(img, "BURST MODE AKTIF", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Tekan sekali -> Rekam beruntun", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    start_y = 130
    
    for i, (label, attr, step) in enumerate(params):
        val = getattr(config, attr)
        is_sel = (i == selected_idx)
        color = (0, 255, 255) if is_sel else (200, 200, 200)
        prefix = "> " if is_sel else "  "
        text = f"{prefix}{label:<18} : {val:.2f}"
        cv2.putText(img, text, (20, start_y + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_sel:
            cv2.circle(img, (10, start_y + (i * 40) - 5), 4, (0, 255, 0), -1)

    return params

# ================= MAIN =================

def main():
    init_csv(FILE_NAME)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    mp_drawing = mp.solutions.drawing_utils
    mesh_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(100, 100, 100))

    sm_dx = SmoothValue()
    sm_dy = SmoothValue()
    normalizer = EyeNormalizer()
    
    baseline = None
    last_msg = "Siap..."
    last_msg_col = (200, 200, 200)
    tuning_idx = 0
    
    # State Recording Burst
    rec_frames_left = 0
    rec_label_current = ""
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        process_frame = frame.copy() 
        process_frame = cv2.flip(process_frame, 1) 
        
        h, w, _ = process_frame.shape
        rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        l_val, l_stat, l_col = check_lighting(process_frame)
        
        results = face_mesh.process(rgb)
        
        dx_rel, dy_rel = 0.0, 0.0
        ear_l, ear_r = 0.0, 0.0
        dist_cm = 0.0
        face_detected = False
        dist_ok = False
        eye_view = None
        
        raw_data = {
            'dx_raw': 0, 'dy_raw': 0,
            'dx_L': 0, 'dx_R': 0,
            'dx_s': 0, 'dy_s': 0,
            'light_val': l_val
        }
        avg_eye_ratio_current = 0.0

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark
            
            mp_drawing.draw_landmarks(
                image=process_frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_spec
            )

            p_iris_L = (int(lm[L_IRIS_CENTER].x * w), int(lm[L_IRIS_CENTER].y * h))
            p_iris_R = (int(lm[R_IRIS_CENTER].x * w), int(lm[R_IRIS_CENTER].y * h))
            cv2.circle(process_frame, p_iris_L, 4, (0, 255, 255), -1)
            cv2.circle(process_frame, p_iris_R, 4, (0, 255, 255), -1)

            eye_view = normalizer.normalize(process_frame, lm, w, h)
            
            dx_L, dy_L, c_L, w_L = get_gaze_vector(lm, w, h, L_INNER, L_OUTER, L_IRIS_CENTER)
            dx_R, dy_R, c_R, w_R = get_gaze_vector(lm, w, h, R_INNER, R_OUTER, R_IRIS_CENTER)
            
            dx_raw = (dx_L + dx_R) / 2.0
            dy_raw = (dy_L + dy_R) / 2.0
            
            avg_width = (w_L + w_R) / 2.0
            avg_eye_ratio_current = avg_width / w 
            dist_cm = config.DISTANCE_K / (avg_eye_ratio_current + 1e-6)
            dist_ok = config.GOOD_MIN_CM <= dist_cm <= config.GOOD_MAX_CM
            
            dx_s = sm_dx.update(dx_raw)
            dy_s = sm_dy.update(dy_raw)
            
            ear_l = calculate_ear(lm, LEFT_EAR_IDXS, w, h)
            ear_r = calculate_ear(lm, RIGHT_EAR_IDXS, w, h)
            
            if baseline is not None:
                dx_rel = dx_s - baseline[0]
                dy_rel = dy_s - baseline[1]
                
                bridge_x = int((c_L[0] + c_R[0]) / 2)
                bridge_y = int((c_L[1] + c_R[1]) / 2)
                end_pt = (int(bridge_x + dx_rel * 200), int(bridge_y + dy_rel * 200))
                cv2.arrowedLine(process_frame, (bridge_x, bridge_y), end_pt, (0, 255, 255), 2)

            raw_data.update({
                'dx_raw': dx_raw, 'dy_raw': dy_raw,
                'dx_L': dx_L, 'dx_R': dx_R,
                'dx_s': dx_s, 'dy_s': dy_s
            })

        blink_L = ear_l < config.BLINK_THRESHOLD
        blink_R = ear_r < config.BLINK_THRESHOLD
        
        ui_data = {
            'baseline': baseline,
            'dx_rel': dx_rel, 'dy_rel': dy_rel,
            'ear_l': ear_l, 'ear_r': ear_r,
            'blink_l': blink_L, 'blink_r': blink_R,
            'dist': dist_cm, 'dist_ok': dist_ok,
            'msg': last_msg, 'msg_col': last_msg_col
        }

        # --- LOGIC BURST RECORDING (AUTO SAVE) ---
        if rec_frames_left > 0:
            if face_detected and dist_ok and baseline is not None:
                row = [
                    datetime.now().isoformat(),
                    f"{dx_rel:.4f}", f"{dy_rel:.4f}",
                    f"{ear_l:.4f}", f"{ear_r:.4f}",
                    1 if blink_L else 0, 1 if blink_R else 0,
                    f"{dist_cm:.1f}", int(l_val),
                    rec_label_current
                ]
                with open(FILE_NAME, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                rec_frames_left -= 1
                if rec_frames_left == 0:
                    last_msg = f"SELESAI: {rec_label_current}"
                    last_msg_col = (0, 255, 0)
            else:
                # Pause recording if face lost, or cancel
                pass 

        # Render UI
        view_cam = process_frame.copy()
        draw_page_1_camera(view_cam, ui_data, eye_view, (rec_frames_left > 0, rec_frames_left, rec_label_current))
        
        view_detail = np.zeros_like(process_frame)
        draw_page_2_details(view_detail, ui_data, raw_data)
        
        view_tuning = np.zeros_like(process_frame)
        params_list = draw_page_3_tuning(view_tuning, tuning_idx)

        combined_view = np.hstack((view_cam, view_detail, view_tuning))

        cv2.imshow("Eye Collector Pro (Dashboard Mode)", combined_view)
        
        # --- INPUT HANDLING ---
        # Gunakan waitKey(1) agar burst recording cepat
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        
        if key == ord('w'): tuning_idx = (tuning_idx - 1) % len(params_list)
        elif key == ord('s'): tuning_idx = (tuning_idx + 1) % len(params_list)
        elif key == ord('a') or key == ord('d'): 
            label, attr, step = params_list[tuning_idx]
            curr_val = getattr(config, attr)
            new_val = curr_val - step if key == ord('a') else curr_val + step
            setattr(config, attr, new_val)
        
        if key == ord('k'):
            if face_detected and avg_eye_ratio_current > 0:
                new_k = config.REF_DIST_CM * avg_eye_ratio_current
                config.DISTANCE_K = new_k
                last_msg = f"K UPDATED: {new_k:.2f}"
                last_msg_col = (0, 255, 255)

        if key == ord('c'):
            if face_detected and dist_ok:
                baseline = np.array([raw_data['dx_s'], raw_data['dy_s']])
                last_msg = "KALIBRASI SUKSES!"
                last_msg_col = (0, 255, 0)
            else:
                last_msg = "GAGAL: Cek Jarak/Wajah"
                last_msg_col = (0,0,255)
        
        # TRIGGER BURST RECORDING
        elif key in LABEL_KEYS and rec_frames_left == 0:
            if not face_detected or baseline is None or not dist_ok:
                last_msg = "ERR: Kalibrasi/Jarak!"
                last_msg_col = (0,0,255)
            else:
                rec_label_current = LABEL_KEYS[key]
                rec_frames_left = config.BURST_LIMIT # Start Burst
                last_msg = f"REC START: {rec_label_current}"
                last_msg_col = (0, 0, 255)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()