#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import cv2
import numpy as np
import mediapipe as mp

# ========= Konfigurasi dasar =========
DATASET_CSV = "eye_gesture_dataset.csv"
CAMERA_INDEX = 2

LABEL_MAP = {
    ord('0'): "neutral",
    ord('1'): "blink",
    ord('2'): "left",
    ord('3'): "right",
    ord('4'): "up",
    ord('5'): "down",
}

mp_face_mesh = mp.solutions.face_mesh

# ========= Landmark index =========
LEFT_EYE_LANDMARKS = [33, 133, 159, 158, 157, 155, 145, 153, 154, 144]
RIGHT_EYE_LANDMARKS = [362, 263, 386, 387, 388, 390, 374, 380, 381, 373]
LEFT_EYE_EAR_IDXS = [33, 159, 158, 133, 145, 153]
RIGHT_EYE_EAR_IDXS = [362, 386, 387, 263, 374, 380]
EYE_FEATURE_LANDMARKS = sorted(list(set(LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS)))

# Iris (MediaPipe FaceMesh)
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

# ========= Parameter jarak (cm) =========
GOOD_MIN_CM = 35.0
GOOD_MAX_CM = 70.0
TOO_CLOSE_CM = 25.0
TOO_FAR_CM = 110.0
DISTANCE_K = 3.0   # konstanta kalibrasi kasar, bisa kamu tuning lagi

# ========= Helper =========
def compute_ear(landmarks, idxs, image_shape):
    h, w, _ = image_shape
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float32)
    p1, p2, p3, p4, p5, p6 = pts
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4) + 1e-6
    ear = (vert1 + vert2) / (2.0 * horiz)
    return float(ear)

class EyeNormalizer:
    def __init__(self, smoothing=0.85):
        self.smoothing = smoothing
        self.center = None
        self.scale = None

    def normalize(self, landmarks, image_shape):
        h, w, _ = image_shape
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

        left_pts = pts[LEFT_EYE_LANDMARKS]
        right_pts = pts[RIGHT_EYE_LANDMARKS]
        left_center = left_pts.mean(axis=0)
        right_center = right_pts.mean(axis=0)

        eyes_center = (left_center + right_center) / 2.0
        eye_dist = np.linalg.norm(right_center - left_center) + 1e-6

        if self.center is None:
            self.center = eyes_center
            self.scale = eye_dist
        else:
            self.center = self.smoothing * self.center + (1.0 - self.smoothing) * eyes_center
            self.scale = self.smoothing * self.scale + (1.0 - self.smoothing) * eye_dist

        norm_pts = (pts - self.center) / float(self.scale)
        return norm_pts, self.center.copy(), float(self.scale)

def draw_normalized_canvas(norm_pts, indices, canvas_size=400):
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    scale = canvas_size * 0.4
    center = np.array([canvas_size / 2, canvas_size / 2])

    for i in indices:
        x, y = norm_pts[i]
        pt = (center + np.array([x, y]) * scale).astype(int)
        cv2.circle(canvas, tuple(pt), 2, (0, 0, 0), -1)

    cv2.line(canvas, (canvas_size // 2, 0),
             (canvas_size // 2, canvas_size), (200, 200, 200), 1)
    cv2.line(canvas, (0, canvas_size // 2),
             (canvas_size, canvas_size // 2), (200, 200, 200), 1)
    cv2.putText(canvas, "Normalized eyes region", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return canvas

def init_csv_if_needed(path):
    if not os.path.exists(path):
        header = ["label", "left_ear", "right_ear"]
        for idx in EYE_FEATURE_LANDMARKS:
            header.append(f"x_{idx}")
            header.append(f"y_{idx}")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def save_sample(path, label, left_ear, right_ear, norm_pts):
    init_csv_if_needed(path)
    row = [label, left_ear, right_ear]
    for idx in EYE_FEATURE_LANDMARKS:
        x, y = norm_pts[idx]
        row.extend([x, y])
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def get_iris_center(landmarks, idxs, image_shape):
    h, w, _ = image_shape
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)

class SmoothValue:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        new_value = np.array(new_value, dtype=np.float32)
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# ========= Main loop =========
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Tidak bisa membuka kamera. Cek index / izin kamera.")
        return

    # Optional set resolusi
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    normalizer = EyeNormalizer(smoothing=0.85)
    smooth_center_eye = SmoothValue(alpha=0.6)
    smooth_pupil = SmoothValue(alpha=0.6)

    prev_time = time.time()

    print("Tekan q / ESC untuk keluar.")
    print("Tekan 0..5 untuk simpan sample dengan label:")
    for k, v in LABEL_MAP.items():
        print(f"   {chr(k)} -> {v}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Sesuaikan orientasi kamera kalau perlu:
        # frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        normalized_canvas = np.ones((400, 400, 3), dtype=np.uint8) * 180

        info_text = "No face detected"
        ready_for_save = False
        left_ear = right_ear = None
        norm_pts = None
        center = None
        eye_scale = None
        distance_cm = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Normalisasi mata
            norm_pts, center, eye_scale = normalizer.normalize(face_landmarks, frame.shape)

            left_ear = compute_ear(face_landmarks, LEFT_EYE_EAR_IDXS, frame.shape)
            right_ear = compute_ear(face_landmarks, RIGHT_EYE_EAR_IDXS, frame.shape)

            # Gambar landmark mata
            for idx in EYE_FEATURE_LANDMARKS:
                x = int(face_landmarks[idx].x * w)
                y = int(face_landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # ==== pupil + gaze vector dengan smoothing ====
            center_eye_raw = center
            center_eye = smooth_center_eye.update(center_eye_raw)

            left_iris = get_iris_center(face_landmarks, LEFT_IRIS, frame.shape)
            right_iris = get_iris_center(face_landmarks, RIGHT_IRIS, frame.shape)
            pupil_raw = (left_iris + right_iris) / 2.0
            pupil = smooth_pupil.update(pupil_raw)

            cx, cy = int(center_eye[0]), int(center_eye[1])
            px, py = int(pupil[0]), int(pupil[1])

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)
            cv2.arrowedLine(frame, (cx, cy), (px, py), (255, 0, 0), 3, tipLength=0.3)

            vec = pupil - center_eye
            dx, dy = vec
            if abs(dx) > abs(dy):
                gaze_dir = "Right" if dx > 0 else "Left"
            else:
                gaze_dir = "Down" if dy > 0 else "Up"

            cv2.putText(frame, f"Gaze: {gaze_dir}", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ==== estimasi jarak ====
            eye_ratio = eye_scale / float(w) if w > 0 else 0.0
            if eye_ratio > 1e-6:
                distance_cm = DISTANCE_K / eye_ratio
            else:
                distance_cm = None

            if distance_cm is None:
                info_text = "Distance unknown"
            else:
                if distance_cm < TOO_CLOSE_CM:
                    info_text = "Too close to camera"
                elif distance_cm > TOO_FAR_CM:
                    info_text = "Too far from camera"
                elif GOOD_MIN_CM <= distance_cm <= GOOD_MAX_CM:
                    info_text = "Good distance"
                    ready_for_save = True
                else:
                    info_text = "OK distance"

            normalized_canvas = draw_normalized_canvas(norm_pts, EYE_FEATURE_LANDMARKS, canvas_size=400)

            cv2.putText(frame, f"EAR L: {left_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR R: {right_ear:.3f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"eye_ratio: {eye_ratio:.3f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if distance_cm is not None:
                cv2.putText(frame, f"Dist: {distance_cm:.1f} cm", (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Info text
        color = (0, 0, 255) if ("Too" in info_text or "No face" in info_text) else (0, 255, 0)
        cv2.putText(frame, info_text, (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Gabung frame & normalized canvas
        disp_frame = cv2.resize(frame, (640, 480))
        disp_canvas = normalized_canvas

        out_h = max(disp_frame.shape[0], disp_canvas.shape[0])
        out_w = disp_frame.shape[1] + disp_canvas.shape[1]
        out = np.ones((out_h, out_w, 3), dtype=np.uint8) * 50
        out[0:disp_frame.shape[0], 0:disp_frame.shape[1]] = disp_frame
        out[0:disp_canvas.shape[0], disp_frame.shape[1]:disp_frame.shape[1] + disp_canvas.shape[1]] = disp_canvas

        cv2.imshow("Eye Gesture Preprocess (Side Camera, Auto Distance)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        if key in LABEL_MAP and ready_for_save and norm_pts is not None:
            label = LABEL_MAP[key]
            save_sample(DATASET_CSV, label, left_ear, right_ear, norm_pts)
            print(f"[SAVED] label={label}, dist={distance_cm:.1f} cm, "
                  f"left_ear={left_ear:.3f}, right_ear={right_ear:.3f}")
        elif key in LABEL_MAP and not ready_for_save:
            print(f"Jarak tidak ideal ({info_text}) → sample TIDAK disimpan.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
