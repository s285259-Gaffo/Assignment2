#!/usr/bin/env python3
"""
Assignment 2 - GOLD Lane Detection Algorithm + YOLO Obstacles + Sliding Window Tracking
"""

import cv2
import numpy as np
import sys
import glob
import os
import urllib.request
from collections import deque

# ==============================================================================
# 1. PARAMETRI CAMERA (Da Assignment)
# ==============================================================================
IMAGE_SIZE = (1920, 1080)
PRINCIPAL_POINT = (970, 483)
FOCAL_LENGTH = (1970, 1970)
CAMERA_POSITION_Z = 1.6600

# ==============================================================================
# 2. INIZIALIZZAZIONE YOLO
# ==============================================================================
YOLO_CFG = "yolov3-tiny.cfg"
YOLO_WEIGHTS = "yolov3-tiny.weights"
YOLO_CLASSES = {0: 'pedestrian', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}

def load_yolo():
    if not os.path.exists(YOLO_CFG):
        print("Scaricamento yolov3-tiny.cfg...")
        req = urllib.request.Request("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(YOLO_CFG, 'wb') as out_file:
            out_file.write(response.read())
    if not os.path.exists(YOLO_WEIGHTS):
        print("Scaricamento yolov3-tiny.weights...")
        req = urllib.request.Request("https://pjreddie.com/media/files/yolov3-tiny.weights", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(YOLO_WEIGHTS, 'wb') as out_file:
            out_file.write(response.read())
    
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

print("Inizializzazione YOLO...")
YOLO_NET = load_yolo()

# ==============================================================================
# 3. GESTIONE BIRD'S EYE VIEW (IPM)
# ==============================================================================
def get_ipm_matrix():
    z_far, z_near = 25.0, 4.0
    x_width = 2.5
    cx, cy = PRINCIPAL_POINT
    fx, fy = FOCAL_LENGTH
    h = CAMERA_POSITION_Z

    u_tl = fx * (-x_width / z_far) + cx
    v_tl = fy * (h / z_far) + cy
    u_tr = fx * (x_width / z_far) + cx
    v_tr = fy * (h / z_far) + cy
    u_br = fx * (x_width / z_near) + cx
    v_br = fy * (h / z_near) + cy
    u_bl = fx * (-x_width / z_near) + cx
    v_bl = fy * (h / z_near) + cy

    src_points = np.float32([[u_tl, v_tl], [u_tr, v_tr], [u_br, v_br], [u_bl, v_bl]])
    bev_width, bev_height = (800, 800)
    dst_points = np.float32([[0, 0], [bev_width, 0], [bev_width, bev_height], [0, bev_height]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix, (bev_width, bev_height)

IPM_MATRIX, BEV_SIZE = get_ipm_matrix()
_, INV_IPM_MATRIX = cv2.invert(IPM_MATRIX)

def apply_ipm(image):
    return cv2.warpPerspective(image, IPM_MATRIX, BEV_SIZE, flags=cv2.INTER_LINEAR)

# ==============================================================================
# 4. TRACKING CLASSE (EMA Smoothing)
# ==============================================================================
class LineTracker:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.best_fit = None
        self.type_history = deque(maxlen=13)

    def update(self, new_fit, current_type_str):
        if new_fit is not None:
            if self.best_fit is None:
                self.best_fit = new_fit
            else:
                self.best_fit = (self.alpha * new_fit) + ((1.0 - self.alpha) * self.best_fit)
            val = 1 if current_type_str == 'continuous' else 0
            self.type_history.append(val)
        return self.best_fit

    def get_stable_type(self):
        if not self.type_history: return 'continuous'
        return 'continuous' if np.mean(self.type_history) > 0.75 else 'dashed'

left_tracker = LineTracker(alpha=0.20)
right_tracker = LineTracker(alpha=0.20)

# ==============================================================================
# 5. ALGORITMO GOLD (Core Requirement)
# ==============================================================================
def iterative_thresholding(img):
    active_pixels = img[img > 0]
    if active_pixels.size == 0: return 15.0
    th = (float(np.min(active_pixels)) + float(np.max(active_pixels))) / 2.0
    for _ in range(10):
        region_a = active_pixels[active_pixels >= th]
        region_b = active_pixels[active_pixels < th]
        if region_a.size == 0 or region_b.size == 0: break
        new_th = (region_a.mean() + region_b.mean()) / 2.0
        if abs(new_th - th) < 0.5: break
        th = new_th
    return th

def apply_gold_filter(bev_image):
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    tau = 12
    left_diff = np.zeros_like(blur)
    right_diff = np.zeros_like(blur)
    
    # GOLD Spatial Filter: Dark - Light - Dark
    left_diff[:, tau:] = blur[:, tau:] - blur[:, :-tau]
    right_diff[:, :-tau] = blur[:, :-tau] - blur[:, tau:]
    gold_filter = np.minimum(left_diff, right_diff)
    gold_filter[gold_filter < 0] = 0
    
    th_opt = iterative_thresholding(gold_filter)
    _, thresh = cv2.threshold(gold_filter, max(12, th_opt * 0.8), 255, cv2.THRESH_BINARY)
    
    # Pulizia morfologica base
    thresh = thresh.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

# ==============================================================================
# 6. SLIDING WINDOW E CLASSIFICAZIONE INTELLIGENTE
# ==============================================================================
def classify_lane_type(y_coords):
    """
    Analizza i pixel trovati. Se ci sono ampi buchi verticali e il 
    rapporto di riempimento è basso, classifica la linea come tratteggiata.
    """
    if len(y_coords) < 50:
        return "dashed"
        
    unique_y = np.unique(y_coords)
    span = unique_y.max() - unique_y.min()
    if span == 0: 
        return "dashed"
        
    # Calcola il rapporto tra i pixel accesi e la lunghezza totale della linea
    fill_ratio = len(unique_y) / float(span)
    
    # Trova le interruzioni: quanti salti verticali sono maggiori di 15 pixel?
    gaps = np.sum(np.diff(unique_y) > 15) 
    
    # Se la linea è piena per meno del 70% e ha almeno 2 buchi netti, è tratteggiata
    if fill_ratio < 0.70 and gaps >= 2:
        return "dashed"
    return "continuous"

def find_lane_pixels_sliding_window(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 12
    margin = 50
    minpix = 30
    window_height = int(binary_warped.shape[0]//nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix: leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    
    left_fit = np.polyfit(lefty, leftx, 1) if len(lefty) > 50 else None
    right_fit = np.polyfit(righty, rightx, 1) if len(righty) > 50 else None
    
    # === CLASSIFICAZIONE INTELLIGENTE ===
    l_type = classify_lane_type(lefty) if len(lefty) > 0 else "continuous"
    r_type = classify_lane_type(righty) if len(righty) > 0 else "continuous"

    return left_fit, right_fit, l_type, r_type

# ==============================================================================
# 7. RILEVAMENTO OSTACOLI (YOLO)
# ==============================================================================
def detect_obstacles(frame):
    h_frame, w_frame = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    YOLO_NET.setInput(blob)
    layer_names = YOLO_NET.getLayerNames()
    output_layers = [layer_names[i - 1] for i in YOLO_NET.getUnconnectedOutLayers()]
    layer_outputs = YOLO_NET.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and class_id in YOLO_CLASSES:
                center_x, center_y = int(detection[0] * w_frame), int(detection[1] * h_frame)
                w, h = int(detection[2] * w_frame), int(detection[3] * h_frame)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            v_bottom = y + h
            cx, cy = PRINCIPAL_POINT
            fx, fy = FOCAL_LENGTH
            distance = fy * CAMERA_POSITION_Z / float(v_bottom - cy) if v_bottom > cy else -1
            if distance > 0:
                results.append({'class_name': YOLO_CLASSES[class_ids[i]], 'box': (x, y, w, h), 'distance': distance})
    return results

# ==============================================================================
# 8. MAIN LOOP
# ==============================================================================
def main():
    search_path = sys.argv[1] if len(sys.argv) > 1 else 'PandaSetSensorData/archive/008/Camera/front_camera/*.jpg'
    image_files = sorted(glob.glob(search_path))
    if not image_files:
        print("Nessuna immagine trovata.")
        sys.exit(1)

    print(f"Trovate {len(image_files)} immagini. Premi spazio per andare avanti, ESC per uscire.")

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # 1. Prospettiva
        bev = apply_ipm(frame)
        
        # 2. Filtro GOLD
        binary_gold = apply_gold_filter(bev)
        
        # 3. Sliding Window
        l_fit_raw, r_fit_raw, l_type_raw, r_type_raw = find_lane_pixels_sliding_window(binary_gold)
        
        # 4. Smoothing con i Tracker
        l_fit = left_tracker.update(l_fit_raw, l_type_raw)
        r_fit = right_tracker.update(r_fit_raw, r_type_raw)
        
        l_type = left_tracker.get_stable_type()
        r_type = right_tracker.get_stable_type()

        # 5. Disegno Linee su originale
        ploty = np.linspace(0, BEV_SIZE[1]-1, BEV_SIZE[1])
        frame_out = frame.copy()
        
        lanes_found = False
        for fit, ltype, side_label in [(l_fit, l_type, "SX"), (r_fit, r_type, "DX")]:
            if fit is not None:
                lanes_found = True
                fitx = np.polyval(fit, ploty)
                pts_bev = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.float32)
                pts_orig = cv2.perspectiveTransform(pts_bev, INV_IPM_MATRIX)
                pts_orig = np.int32(pts_orig[0])
                
                # Testo per mostrare la classificazione
                text_pos = tuple(pts_orig[-10]) if len(pts_orig) > 10 else tuple(pts_orig[-1])
                label_text = f"{side_label}: Tratt." if ltype == "dashed" else f"{side_label}: Continua"
                cv2.putText(frame_out, label_text, (text_pos[0] - 50, text_pos[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Tratteggiata vs Continua (Disegno visivo)
                if ltype == "dashed":
                    for i in range(0, len(pts_orig)-15, 25): # Salta dei segmenti per l'effetto tratteggiato
                        cv2.line(frame_out, tuple(pts_orig[i]), tuple(pts_orig[i+15]), (0, 255, 0), 5)
                else:
                    cv2.polylines(frame_out, [pts_orig], False, (0, 255, 0), 5)

        if not lanes_found:
            cv2.putText(frame_out, "No lanes found", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # 6. Disegno Ostacoli (YOLO)
        obstacles = detect_obstacles(frame)
        for obs in obstacles:
            x, y, w, h = obs['box']
            dist = obs['distance']
            cv2.rectangle(frame_out, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame_out, f"{obs['class_name']} {dist:.1f}m", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Mostra risultato
        cv2.imshow("GOLD + Sliding Window + YOLO", cv2.resize(frame_out, (1280, 720)))
        cv2.imshow("GOLD Filter (BEV)", cv2.resize(binary_gold, (400, 400)))
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27: break # ESC per uscire

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()