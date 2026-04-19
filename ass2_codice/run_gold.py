#!/usr/bin/env python3
"""
Assignment 2 - GOLD Lane Detection + YOLO (IoA Merged & CIPV) + Sliding Window Tracking
Versione Definitiva: Obstacle Tracking, Mascheratura Auto con Padding, e Line Tracker con Timeout.
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

    u_tl = fx * (-x_width / z_far) + cx; v_tl = fy * (h / z_far) + cy
    u_tr = fx * (x_width / z_far) + cx;  v_tr = fy * (h / z_far) + cy
    u_br = fx * (x_width / z_near) + cx; v_br = fy * (h / z_near) + cy
    u_bl = fx * (-x_width / z_near) + cx; v_bl = fy * (h / z_near) + cy

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
# 4. TRACKING CORSIA CON TIMEOUT (Anti Ghost-Lines per Incroci)
# ==============================================================================
class LineTracker:
    def __init__(self, alpha=0.15, max_unseen=5):
        self.alpha = alpha
        self.best_fit = None
        self.type_history = deque(maxlen=13)
        self.unseen_count = 0  
        self.max_unseen = max_unseen

    def update(self, new_fit, current_type_str):
        if new_fit is not None:
            if self.best_fit is None:
                self.best_fit = new_fit
            else:
                self.best_fit = (self.alpha * new_fit) + ((1.0 - self.alpha) * self.best_fit)
            val = 1 if current_type_str == 'continuous' else 0
            self.type_history.append(val)
            self.unseen_count = 0  
        else:
            self.unseen_count += 1
            if self.unseen_count > self.max_unseen:
                self.best_fit = None
                self.type_history.clear()
                
        return self.best_fit

    def get_stable_type(self):
        if not self.type_history: return 'continuous'
        return 'continuous' if np.mean(self.type_history) > 0.75 else 'dashed'

left_tracker = LineTracker(alpha=0.20, max_unseen=5)
right_tracker = LineTracker(alpha=0.20, max_unseen=5)

# ==============================================================================
# 5. ALGORITMO GOLD E CLASSIFICAZIONE 
# ==============================================================================
def iterative_thresholding(img):
    active_pixels = img[img > 0]
    if active_pixels.size == 0: return 15.0
    if np.max(active_pixels) == np.min(active_pixels): return 15.0
    
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
    
    left_diff[:, tau:] = blur[:, tau:] - blur[:, :-tau]
    right_diff[:, :-tau] = blur[:, :-tau] - blur[:, tau:]
    gold_filter = np.minimum(left_diff, right_diff)
    gold_filter[gold_filter < 0] = 0
    
    th_opt = iterative_thresholding(gold_filter)
    _, thresh = cv2.threshold(gold_filter, max(12, th_opt * 0.8), 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)

    raw_occupancy = np.sum(thresh // 255, axis=0)
    obstacle_mask = cv2.dilate((raw_occupancy > (0.95 * BEV_SIZE[1])).astype(np.uint8).reshape(1, -1), np.ones((1, 15), np.uint8), iterations=1).flatten() > 0
    thresh[:, obstacle_mask] = 0

    row_occupancy = np.sum(thresh // 255, axis=1)
    crosswalk_rows = row_occupancy > int(0.18 * BEV_SIZE[0])
    if np.any(crosswalk_rows):
        crosswalk_rows = cv2.dilate(crosswalk_rows.astype(np.uint8).reshape(-1, 1), np.ones((11, 1), np.uint8), iterations=1).reshape(-1) > 0
        thresh[crosswalk_rows, :] = 0

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    if np.count_nonzero(horizontal_mask) > 0:
        horizontal_mask = cv2.dilate(horizontal_mask, np.ones((5, 5), np.uint8), iterations=1)
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(horizontal_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

def find_lane_pixels_sliding_window(binary_warped):
    raw_hist = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    histogram = np.convolve(raw_hist, np.ones(30), mode='same').astype(np.float32)
    
    midpoint = int(histogram.shape[0]//2)
    left_peak_val = np.max(histogram[:midpoint])
    right_peak_val = np.max(histogram[midpoint:])
    
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    lane_width = rightx_base - leftx_base
    if lane_width < 250 or lane_width > 600:
        if left_peak_val > right_peak_val: rightx_base = None
        else: leftx_base = None

    nwindows = 12; margin = 50; minpix = 30
    window_height = int(binary_warped.shape[0]//nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])
    
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        if leftx_current is not None:
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            if len(good_left_inds) > minpix: leftx_current = int(np.mean(nonzerox[good_left_inds]))
            
        if rightx_current is not None:
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix: rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_fit = right_fit = l_type = r_type = None

    PIXEL_THRESHOLD = 150

    if leftx_base is not None:
        left_lane_inds = np.concatenate(left_lane_inds)
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 1) if len(lefty) > PIXEL_THRESHOLD else None
        
    if rightx_base is not None:
        right_lane_inds = np.concatenate(right_lane_inds)
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 1) if len(righty) > PIXEL_THRESHOLD else None

    MAX_SLOPE = 0.6 
    if left_fit is not None and abs(left_fit[0]) > MAX_SLOPE: left_fit = None
    if right_fit is not None and abs(right_fit[0]) > MAX_SLOPE: right_fit = None

    if left_fit is not None and right_fit is not None:
        slope_diff = abs(left_fit[0] - right_fit[0])
        if slope_diff > 0.35: 
            if abs(left_fit[0]) > abs(right_fit[0]): left_fit = None
            else: right_fit = None

    def classify_lane_type(y_coords):
        if len(y_coords) < 50: return "dashed"
        unique_y = np.unique(y_coords)
        span = unique_y.max() - unique_y.min()
        if span == 0: return "dashed"
        fill_ratio = len(unique_y) / float(span)
        gaps = np.sum(np.diff(unique_y) > 15) 
        if fill_ratio < 0.70 and gaps >= 2: return "dashed"
        return "continuous"

    l_type = classify_lane_type(lefty) if 'lefty' in locals() and len(lefty) > 0 else "continuous"
    r_type = classify_lane_type(righty) if 'righty' in locals() and len(righty) > 0 else "continuous"

    return left_fit, right_fit, l_type, r_type

# ==============================================================================
# 6. RILEVAMENTO OSTACOLI E TRACKING (YOLO + IoA + Centroid)
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
    raw_results = []
    
    cx, cy = PRINCIPAL_POINT
    fx, fy = FOCAL_LENGTH
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            v_bottom = y + h
            distance = fy * CAMERA_POSITION_Z / float(v_bottom - cy) if v_bottom > cy else -1
            raw_results.append({'class_name': YOLO_CLASSES[class_ids[i]], 'confidence': confidences[i], 'box': (x, y, w, h), 'distance': distance})
            
    merged_results = []
    for r in raw_results:
        x, y, w, h = r['box']
        area_r = w * h
        is_merged = False
        
        for mr in merged_results:
            mx, my, mw, mh = mr['box']
            area_m = mw * mh
            ix = max(x, mx); iy = max(y, my)
            iw = min(x + w, mx + mw) - ix; ih = min(y + h, my + mh) - iy
            
            if iw > 0 and ih > 0:
                inter_area = iw * ih
                if inter_area / min(area_r, area_m) > 0.6: 
                    new_x = min(x, mx); new_y = min(y, my)
                    new_w = max(x + w, mx + mw) - new_x; new_h = max(y + h, my + mh) - new_y
                    mr['box'] = (new_x, new_y, new_w, new_h)
                    v_bottom_new = new_y + new_h
                    if v_bottom_new > cy:
                        mr['distance'] = fy * CAMERA_POSITION_Z / float(v_bottom_new - cy)
                    mr['confidence'] = max(r['confidence'], mr['confidence'])
                    is_merged = True
                    break
                    
        if not is_merged:
            merged_results.append(r)
            
    return merged_results

class ObstacleTracker:
    def __init__(self, max_unseen=5, distance_threshold=80, alpha=0.4):
        self.tracked_obstacles = [] 
        self.next_id = 0
        self.max_unseen = max_unseen
        self.distance_threshold = distance_threshold
        self.alpha = alpha

    def update(self, new_detections):
        updated_obstacles = []
        unassigned_detections = new_detections.copy()

        for tracked in self.tracked_obstacles:
            best_match = None
            best_dist = float('inf')
            
            tx, ty, tw, th = tracked['box']
            tcx, tcy = tx + tw / 2.0, ty + th / 2.0

            for det in unassigned_detections:
                nx, ny, nw, nh = det['box']
                ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
                dist = np.sqrt((tcx - ncx)**2 + (tcy - ncy)**2)
                
                if dist < self.distance_threshold and dist < best_dist:
                    best_dist = dist
                    best_match = det
            
            if best_match:
                nx, ny, nw, nh = best_match['box']
                tracked['box'] = (
                    int(self.alpha * nx + (1 - self.alpha) * tx),
                    int(self.alpha * ny + (1 - self.alpha) * ty),
                    int(self.alpha * nw + (1 - self.alpha) * tw),
                    int(self.alpha * nh + (1 - self.alpha) * th)
                )
                tracked['distance'] = best_match['distance'] 
                tracked['unseen'] = 0
                updated_obstacles.append(tracked)
                unassigned_detections.remove(best_match)
            else:
                tracked['unseen'] += 1
                if tracked['unseen'] <= self.max_unseen:
                    updated_obstacles.append(tracked)

        for det in unassigned_detections:
            det['id'] = self.next_id
            det['unseen'] = 0
            self.next_id += 1
            updated_obstacles.append(det)

        self.tracked_obstacles = updated_obstacles
        return self.tracked_obstacles

# ==============================================================================
# 7. MAIN LOOP
# ==============================================================================
def main():
    search_path = sys.argv[1] if len(sys.argv) > 1 else 'PandaSetSensorData/archive/008/Camera/front_camera/*.jpg'
    image_files = sorted(glob.glob(search_path))
    if not image_files:
        print("Nessuna immagine trovata.")
        sys.exit(1)

    print(f"Trovate {len(image_files)} immagini. Premi SPAZIO (o un tasto qualsiasi) per andare avanti, ESC per uscire.")

    obstacle_tracker = ObstacleTracker(max_unseen=4, distance_threshold=100, alpha=0.5)

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # --- 1. RILEVAMENTO OSTACOLI (YOLO + Tracking) ---
        raw_obstacles = detect_obstacles(frame)
        tracked_obstacles = obstacle_tracker.update(raw_obstacles)

        # --- 2. PROSPETTIVA E GOLD ---
        bev = apply_ipm(frame)
        binary_gold = apply_gold_filter(bev)
        
        # --- 3. MASCHERATURA AUTO (Gomma da cancellare con PADDING) ---
        cx, cy = PRINCIPAL_POINT
        for obs in tracked_obstacles:
            x, y, w, h = obs['box']
            
            # PADDING: Allarghiamo la gomma da cancellare di 40 pixel per lato
            # in modo da assorbire cofani, specchietti o ruote fuori dal box
            PAD = 40
            x_pad = max(0, x - PAD)
            y_pad = max(0, y - PAD)
            w_pad = w + (PAD * 2)
            h_pad = h + (PAD * 2)
            
            y_top_safe = max(y_pad, cy + 10)
            y_bottom = y_pad + h_pad
            
            if y_bottom <= y_top_safe: continue
            
            # Usiamo le coordinate "paddate"
            pts = np.float32([[[x_pad, y_top_safe], [x_pad+w_pad, y_top_safe], [x_pad+w_pad, y_bottom], [x_pad, y_bottom]]])
            bev_pts = cv2.perspectiveTransform(pts, IPM_MATRIX)
            cv2.fillConvexPoly(binary_gold, np.int32(bev_pts[0]), 0)

        # --- 4. TRACKING E TIPI LINEE ---
        l_fit_raw, r_fit_raw, l_type_raw, r_type_raw = find_lane_pixels_sliding_window(binary_gold)
        l_fit = left_tracker.update(l_fit_raw, l_type_raw)
        r_fit = right_tracker.update(r_fit_raw, r_type_raw)
        
        l_type = left_tracker.get_stable_type() if l_fit is not None else None
        r_type = right_tracker.get_stable_type() if r_fit is not None else None

        # --- 5. DISEGNO ---
        ploty = np.linspace(0, BEV_SIZE[1]-1, BEV_SIZE[1])
        frame_out = frame.copy()
        bev_vis = bev.copy()
        
        # 5a. Disegno Linee
        lanes_found = False
        for fit, ltype, side_label in [(l_fit, l_type, "SX"), (r_fit, r_type, "DX")]:
            if fit is not None and ltype is not None:
                lanes_found = True
                fitx = np.polyval(fit, ploty)
                pts_bev = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.float32)
                
                pts_orig = cv2.perspectiveTransform(pts_bev, INV_IPM_MATRIX)
                pts_orig = np.int32(pts_orig[0])
                pts_bev_int = np.int32(pts_bev[0])
                
                text_pos = tuple(pts_orig[-10]) if len(pts_orig) > 10 else tuple(pts_orig[-1])
                label_text = f"{side_label}: Tratt." if ltype == "dashed" else f"{side_label}: Continua"
                cv2.putText(frame_out, label_text, (text_pos[0] - 50, text_pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                if ltype == "dashed":
                    for i in range(0, len(pts_orig)-10, 50): 
                        cv2.line(frame_out, tuple(pts_orig[i]), tuple(pts_orig[i+10]), (0, 255, 0), 5)
                    for i in range(0, len(pts_bev_int)-20, 60):
                        cv2.line(bev_vis, tuple(pts_bev_int[i]), tuple(pts_bev_int[i+20]), (0, 255, 0), 6)
                else:
                    cv2.polylines(frame_out, [pts_orig], False, (0, 255, 0), 5)
                    cv2.polylines(bev_vis, [pts_bev_int], False, (0, 255, 0), 6)

        if not lanes_found:
            cv2.putText(frame_out, "No lanes found", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(bev_vis, "No lanes found", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 5b. Disegno Ostacoli e CIPV
        ego_obstacles = []
        for obs in tracked_obstacles:
            x, y, w, h = obs['box']
            dist = obs['distance']
            unseen = obs['unseen']
            
            color = (180, 180, 180) if unseen == 0 else (100, 100, 100)
            thickness = 2 if unseen == 0 else 1
            
            cv2.rectangle(frame_out, (x, y), (x+w, y+h), color, thickness)
            
            if dist > 0:
                u_center = x + w / 2.0; v_bottom = y + h
                pts_orig_bottom = np.array([[[u_center, float(v_bottom)]]], dtype=np.float32)
                pts_bev_bottom = cv2.perspectiveTransform(pts_orig_bottom, IPM_MATRIX)
                bev_x = pts_bev_bottom[0][0][0]; bev_y = pts_bev_bottom[0][0][1]
                
                margin = -10 
                in_lane = False
                
                if l_fit is not None and r_fit is not None:
                    left_bound = np.polyval(l_fit, bev_y); right_bound = np.polyval(r_fit, bev_y)
                    if (left_bound - margin) <= bev_x <= (right_bound + margin): in_lane = True
                elif l_fit is not None:
                    left_bound = np.polyval(l_fit, bev_y)
                    if (left_bound - margin) <= bev_x <= (left_bound + 350 + margin): in_lane = True
                elif r_fit is not None:
                    right_bound = np.polyval(r_fit, bev_y)
                    if (right_bound - 350 - margin) <= bev_x <= (right_bound + margin): in_lane = True
                else:
                    if 280 <= bev_x <= 520: in_lane = True

                if in_lane:
                    ego_obstacles.append(obs)

        if ego_obstacles:
            closest_obs = min(ego_obstacles, key=lambda o: o['distance'])
            cx, cy, cw, ch = closest_obs['box']
            cdist = closest_obs['distance']
            
            cv2.rectangle(frame_out, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 3)
            cv2.putText(frame_out, f"WARNING: OSTACOLO A {cdist:.1f}m!", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

        # --- 6. AFFIANCAMENTO ---
        TARGET_HEIGHT = 540 
        
        h_orig, w_orig = frame_out.shape[:2]
        w_orig_new = int(w_orig * (TARGET_HEIGHT / float(h_orig)))
        frame_resized = cv2.resize(frame_out, (w_orig_new, TARGET_HEIGHT))
        
        h_bev, w_bev = bev_vis.shape[:2]
        w_bev_new = int(w_bev * (TARGET_HEIGHT / float(h_bev)))
        bev_resized = cv2.resize(bev_vis, (w_bev_new, TARGET_HEIGHT))
        
        combined_view = np.hstack((frame_resized, bev_resized))
        
        cv2.namedWindow("Driver View (sx) & Bird's Eye View (dx)", cv2.WINDOW_NORMAL)
        cv2.imshow("Driver View (sx) & Bird's Eye View (dx)", combined_view)
        
        key = cv2.waitKey(0) & 0xFF 
        if key == 27: break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()