import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- RTC Config สำหรับการเชื่อมต่อกล้องผ่าน Network ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class MobileMultiTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.p_centers = [] 
        self.smooth_factor = 3 
        self.prev_nx, self.prev_ny = 0, 0
        self.first_frame = True

    def process(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(img_rgb)
        face_results = self.face_detection.process(img_rgb)
        
        current_centers = []
        face_data = [] 

        # --- Hand Tracking ---
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                m2, m5 = hand_lms.landmark[2], hand_lms.landmark[5]
                cx, cy = int(((m2.x + m5.x) / 2) * w), int(((m2.y + m5.y) / 2) * h)
                current_centers.append([cx, cy])
        
        # --- Face Detection ---
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                keypoints = detection.location_data.relative_keypoints
                face_data.append({
                    "bbox": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                    "nose": [keypoints[2].x, keypoints[2].y]
                })

        # --- Smoothing ---
        if not self.p_centers or len(current_centers) != len(self.p_centers):
            self.p_centers = current_centers
        else:
            for i in range(len(current_centers)):
                self.p_centers[i][0] += (current_centers[i][0] - self.p_centers[i][0]) // self.smooth_factor
                self.p_centers[i][1] += (current_centers[i][1] - self.p_centers[i][1]) // self.smooth_factor
                
        return img, self.p_centers, face_data, hand_results

# --- ส่วนของการวาด UI และ Filter ---
tracker = MobileMultiTracker()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # 9:16 Crop สำหรับมือถือ
    target_w = int(h * (9/16))
    start_x = (w - target_w) // 2
    img = img[:, start_x : start_x + target_w].copy()
    
    img, centers, faces, hand_results = tracker.process(img)

    # 1. ดึงสีฉากหลัง (ขยับจุดดึงสีลงมาเล็กน้อย)
    sample_roi = img[10:60, 10:60]
    avg_color = cv2.mean(sample_roi)[:3]
    hex_code = "#{:02x}{:02x}{:02x}".format(int(avg_color[2]), int(avg_color[1]), int(avg_color[0])).upper()

    # 2. วาดการ์ด Pantone
    for face in faces:
        nx, ny = int(face["nose"][0] * target_w), int(face["nose"][1] * h)
        if tracker.first_frame:
            tracker.prev_nx, tracker.prev_ny = nx, ny
            tracker.first_frame = False
        
        smooth_nx = int(tracker.prev_nx * 0.85 + nx * 0.15)
        smooth_ny = int(tracker.prev_ny * 0.85 + ny * 0.15)
        tracker.prev_nx, tracker.prev_ny = smooth_nx, smooth_ny

        scale_factor = (face["bbox"][2] * target_w) / 200.0 
        x_h, y_h, off_u = int(75 * scale_factor), int(75 * scale_factor), int(45 * scale_factor)
        x1, y1, x2, y2 = smooth_nx - x_h, smooth_ny - y_h - off_u, smooth_nx + x_h, smooth_ny + y_h - off_u
        
        border, info_h = int(10 * scale_factor), int(65 * scale_factor)
        cv2.rectangle(img, (x1-border, y1-border), (x2+border, y2+info_h), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), avg_color, -1)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "PANTONE", (x1-border+10, y2+int(25*scale_factor)), font, 0.55*scale_factor, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, hex_code, (x1-border+10, y2+int(45*scale_factor)), font, 0.4*scale_factor, (60,60,60), 1, cv2.LINE_AA)

    # 3. ฟิลเตอร์มือ (Glitch / Blur)
    if hand_results and hand_results.multi_hand_landmarks and len(centers) == 2:
        hand_data = []
        for idx, hand_handedness in enumerate(hand_results.multi_handedness):
            hand_data.append({'label': hand_handedness.classification[0].label, 'y': hand_results.multi_hand_landmarks[idx].landmark[0].y})

        mx1, mx2 = max(0, min(centers[0][0], centers[1][0])), min(img.shape[1], max(centers[0][0], centers[1][0]))
        my1, my2 = max(0, min(centers[0][1], centers[1][1])), min(img.shape[0], max(centers[0][1], centers[1][1]))

        if mx2 - mx1 > 20 and my2 - my1 > 20:
            roi = img[my1:my2, mx1:mx2]
            if roi.size > 0:
                roi = cv2.GaussianBlur(roi, (51, 51), 0)
                ly = next((h['y'] for h in hand_data if h['label'] == 'Left'), 1.0)
                ry = next((h['y'] for h in hand_data if h['label'] == 'Right'), 1.0)
                if ly < ry: roi = 255 - roi
                else:
                    b, g, r = cv2.split(roi)
                    sh = int(30 * (target_w/360))
                    M_r, M_b = np.float32([[1,0,sh],[0,1,0]]), np.float32([[1,0,-sh],[0,1,0]])
                    r = cv2.warpAffine(r, M_r, (r.shape[1], r.shape[0]))
                    b = cv2.warpAffine(b, M_b, (b.shape[1], b.shape[0]))
                    roi = cv2.merge([b, g, r])
                img[my1:my2, mx1:mx2] = roi

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(page_title="Pantone Mobile", layout="centered")
st.title("🎨 Pantone Pro-Smooth")
st.caption("AI Color & Hand Effects (Mobile Ready)")

webrtc_streamer(
    key="pantone-mobile",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True,
)