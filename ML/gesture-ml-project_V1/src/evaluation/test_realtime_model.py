import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import time

# ================= KONFIGURASI =================
MODEL_PATH = '/home/ahmad/Documents/Prata/Code/ML/gesture-ml-project_V1/notebooks/wheelchair_gaze_model.h5'
SCALER_PATH = '/home/ahmad/Documents/Prata/Code/ML/gesture-ml-project_V1/notebooks/gaze_scaler.pkl'
ENCODER_PATH = '/home/ahmad/Documents/Prata/Code/ML/gesture-ml-project_V1/notebooks/label_encoder.pkl'

# Parameter (Harus sama dengan saat Training/Collecting)
SEQUENCE_LENGTH = 20          # Panjang jendela waktu LSTM
BLINK_THRESHOLD = 0.30        # Batas EAR untuk kedip (Safety Switch)
SMOOTH_ALPHA = 0.70           # Smoothing pergerakan

# MediaPipe Indices
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473
L_OUTER, L_INNER = 33, 133
R_OUTER, R_INNER = 263, 362
LEFT_EAR_IDXS  = [33, 159, 158, 133, 145, 153]
RIGHT_EAR_IDXS = [362, 386, 387, 263, 374, 380]

# ================= CLASS HELPER =================

class EyeProcessor:
    def __init__(self):
        self.sm_dx = None
        self.sm_dy = None
        self.baseline = None # Titik Nol (0,0) setelah kalibrasi

    def smooth(self, current, prev):
        if prev is None: return current
        return SMOOTH_ALPHA * current + (1 - SMOOTH_ALPHA) * prev

    def calculate_ear(self, lm, w, h, idxs):
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in idxs])
        v = (np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])) / 2.0
        hor = np.linalg.norm(pts[0]-pts[3]) + 1e-6
        return v / hor

    def get_vector(self, lm, w, h, idx_in, idx_out, idx_iris):
        p_in = np.array([lm[idx_in].x * w, lm[idx_in].y * h])
        p_out = np.array([lm[idx_out].x * w, lm[idx_out].y * h])
        p_iris = np.array([lm[idx_iris].x * w, lm[idx_iris].y * h])
        center = (p_in + p_out) / 2.0
        width = np.linalg.norm(p_out - p_in) + 1e-6
        return (p_iris[0] - center[0]) / width, (p_iris[1] - center[1]) / width

    def process(self, lm, w, h):
        # 1. Hitung EAR
        ear_l = self.calculate_ear(lm, w, h, LEFT_EAR_IDXS)
        ear_r = self.calculate_ear(lm, w, h, RIGHT_EAR_IDXS)
        
        # 2. Hitung Vektor Gaze (Rata-rata Kiri Kanan)
        dxL, dyL = self.get_vector(lm, w, h, L_INNER, L_OUTER, L_IRIS_CENTER)
        dxR, dyR = self.get_vector(lm, w, h, R_INNER, R_OUTER, R_IRIS_CENTER)
        
        dx_raw = (dxL + dxR) / 2.0
        dy_raw = (dyL + dyR) / 2.0
        
        # 3. Smoothing
        self.sm_dx = self.smooth(dx_raw, self.sm_dx)
        self.sm_dy = self.smooth(dy_raw, self.sm_dy)
        
        # 4. Relatif terhadap Baseline (Jika sudah kalibrasi)
        dx_rel, dy_rel = 0.0, 0.0
        if self.baseline is not None:
            dx_rel = self.sm_dx - self.baseline[0]
            dy_rel = self.sm_dy - self.baseline[1]
            
        return [dx_rel, dy_rel, ear_l, ear_r] # Format data sesuai Training

# ================= MAIN PROGRAM =================

def main():
    # 1. Load Model & Tools
    print("Loading Model AI...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        labels = encoder.classes_
        print(f"Model siap! Labels: {labels}")
    except Exception as e:
        print(f"GAGAL LOAD MODEL: {e}")
        print("Pastikan file .h5, .pkl ada di folder yang sama.")
        return

    # 2. Setup Kamera
    cap = cv2.VideoCapture(2) # Sesuaikan index kamera (0 atau 1)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Utils untuk gambar Mesh
    mp_drawing = mp.solutions.drawing_utils
    mesh_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(100, 100, 100))
    
    processor = EyeProcessor()
    
    # Buffer untuk LSTM (Sliding Window)
    # deque otomatis membuang data lama jika penuh
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    current_prediction = "MENUNGGU..."
    confidence = 0.0
    status_color = (100, 100, 100)
    
    prev_time = 0 # Untuk hitung FPS
    
    print("\n=== KONTROL KURSI RODA AI ===")
    print("1. Lihat LURUS ke depan (Jalan)")
    print("2. Tekan 'C' untuk Kalibrasi")
    print("3. Sistem Aktif!")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror frame untuk kenyamanan user
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hitung FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # --- 0. VISUALISASI FACE MESH & PUPIL ---
            # Gambar Jaring Wajah
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_spec
            )
            
            # Gambar KEDUA Pupil (Kiri & Kanan)
            cx_l, cy_l = int(lm[L_IRIS_CENTER].x * w), int(lm[L_IRIS_CENTER].y * h)
            cx_r, cy_r = int(lm[R_IRIS_CENTER].x * w), int(lm[R_IRIS_CENTER].y * h)
            cv2.circle(frame, (cx_l, cy_l), 4, (0, 255, 255), -1)
            cv2.circle(frame, (cx_r, cy_r), 4, (0, 255, 255), -1)
            
            # --- 1. PROSES FITUR ---
            # features = [dx_rel, dy_rel, ear_l, ear_r]
            features = processor.process(lm, w, h)
            ear_avg = (features[2] + features[3]) / 2.0
            
            # --- 2. LOGIKA HYBRID (SAFETY FIRST) ---
            
            # A. Cek Kedip (Hardware Interrupt)
            if ear_avg < BLINK_THRESHOLD:
                current_prediction = "STOP (BLINK)"
                status_color = (0, 0, 255) # Merah
                sequence_buffer.clear() # Reset ingatan AI agar tidak bingung
            
            # B. Cek Kalibrasi
            elif processor.baseline is None:
                current_prediction = "PERLU KALIBRASI (C)"
                status_color = (0, 165, 255) # Orange
            
            # C. Jalankan AI (Jika Mata Terbuka & Sudah Kalibrasi)
            else:
                # Normalisasi data menggunakan Scaler yang disimpan saat training
                # Reshape jadi 2D array [1, 4] karena scaler butuh format tabel
                feat_scaled = scaler.transform(np.array([features]))[0]
                
                # Masukkan ke buffer
                sequence_buffer.append(feat_scaled)
                
                # Jika buffer penuh (sudah punya 20 frame / 1 detik data)
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    # Siapkan input LSTM: [1, 20, 4]
                    input_seq = np.array([sequence_buffer])
                    
                    # Prediksi
                    pred_probs = model.predict(input_seq, verbose=0)[0]
                    pred_idx = np.argmax(pred_probs)
                    confidence = pred_probs[pred_idx]
                    
                    # Threshold Confidence (Agar tidak ragu-ragu)
                    if confidence > 0.60:
                        current_prediction = labels[pred_idx]
                        
                        # Warna Status
                        if "CENTER" in current_prediction: status_color = (0, 255, 0) # Hijau
                        elif "LEFT" in current_prediction: status_color = (255, 255, 0) # Cyan
                        elif "RIGHT" in current_prediction: status_color = (255, 255, 0)
                        elif "DOWN" in current_prediction: status_color = (0, 0, 255) # Merah (Mundur)
                    else:
                        current_prediction = "..." # Tidak yakin

        # --- UI DISPLAY ---
        # Kotak Background atas
        cv2.rectangle(frame, (0, 0), (640, 120), (30, 30, 30), -1)
        
        # Teks Status Utama
        cv2.putText(frame, f"STATUS: {current_prediction}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Info Tambahan
        if processor.baseline is not None:
            # Tampilkan X, Y, Confidence, dan FPS
            info = f"Conf: {confidence:.2f} | FPS: {int(fps)}"
            detail = f"X: {features[0]:.3f} | Y: {features[1]:.3f} | EAR: {ear_avg:.2f}"
            cv2.putText(frame, info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, detail, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Lihat Jalan Lurus -> Tekan 'C'", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (500, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Realtime AI Test", frame)
        
        # --- INPUT ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        
        if key == ord('c'):
            if results.multi_face_landmarks:
                # Set Baseline ke posisi saat ini (Smoothed)
                if processor.sm_dx is not None:
                    processor.baseline = np.array([processor.sm_dx, processor.sm_dy])
                    print("Kalibrasi Berhasil!")
                    sequence_buffer.clear() # Reset buffer biar fresh

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()