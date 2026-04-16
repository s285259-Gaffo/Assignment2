import cv2
import numpy as np
import sys
import glob

# -
# Parametri della telecamera forniti dall'assignment
IMAGE_SIZE = (1920, 1080) # (width, height)
PRINCIPAL_POINT = (970, 483) # (cx, cy)
FOCAL_LENGTH = (1970, 1970) # (fx, fy)
CAMERA_POSITION_Z = 1.6600 # Altezza della camera dal suolo (metri)

def get_ipm_matrix():
    """
    Calcola la matrice di trasformazione prospettica per l'IPM.
    Mappiamo 4 punti 3D dal piano della strada (Z_world = profondità, X_world = larghezza)
    sul piano immagine (u, v) e definiamo un rettangolo di destinazione (Bird's Eye View).
    """
    # Definiamo i confini dell'area da guardare sulla strada (in metri rispetto alla telecamera)
    z_far = 25.0   # Riduciamo la profondità a 25 metri per evitare distorsioni e orizzonte
    z_near = 4.0   # Partiamo a guardare da 4 metri dal muso dell'auto
    x_width = 2.5  # Guardiamo -2.5m a sx e +2.5m a dx (5 metri totali)

    # Formula della prospettiva:
    # u = fx * (X / Z) + cx
    # v = fy * (Y / Z) + cy  (Y è l'altezza dal suolo: CAMERA_POSITION_Z)
    cx, cy = PRINCIPAL_POINT
    fx, fy = FOCAL_LENGTH
    h = CAMERA_POSITION_Z

    # Calcoliamo le coordinate (u, v) nell'immagine dei nostri 4 estremi stradali
    # Ordine punti: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    u_tl = fx * (-x_width / z_far) + cx
    v_tl = fy * (h / z_far) + cy
    
    u_tr = fx * (x_width / z_far) + cx
    v_tr = fy * (h / z_far) + cy
    
    u_br = fx * (x_width / z_near) + cx
    v_br = fy * (h / z_near) + cy
    
    u_bl = fx * (-x_width / z_near) + cx
    v_bl = fy * (h / z_near) + cy

    src_points = np.float32([
        [u_tl, v_tl],
        [u_tr, v_tr],
        [u_br, v_br],
        [u_bl, v_bl]
    ])

    # Definiamo la dimensione della Bird's Eye View (es. 800x800 pixel)
    bev_width, bev_height = (800, 800)
    
    dst_points = np.float32([
        [0, 0],                      # Top-Left dest
        [bev_width, 0],              # Top-Right dest
        [bev_width, bev_height],     # Bottom-Right dest
        [0, bev_height]              # Bottom-Left dest
    ])

    # Calcolo matrice di trasformazione
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return matrix, (bev_width, bev_height)

# Calcoliamo la matrice una volta sola, insieme alla sua inversa
IPM_MATRIX, BEV_SIZE = get_ipm_matrix()
_, INV_IPM_MATRIX = cv2.invert(IPM_MATRIX)

def apply_ipm(image):
    return cv2.warpPerspective(image, IPM_MATRIX, BEV_SIZE, flags=cv2.INTER_LINEAR)

def iterative_thresholding(img):
    """Calcola la soglia ottima dinamicamente in base alle luci/ombre dell'immagine. (Formula Prof)"""
    # Ignoriamo gli zeri assoluti, altrimenti la media crolla verso il basso considerandoli parte dell'asfalto
    active_pixels = img[img > 0]
    if active_pixels.size == 0:
        return 15.0
    if np.max(active_pixels) == np.min(active_pixels): 
        return 15.0
        
    th = (float(np.min(active_pixels)) + float(np.max(active_pixels))) / 2.0
    for _ in range(10):
        region_a = active_pixels[active_pixels >= th]
        region_b = active_pixels[active_pixels < th]
        if region_a.size == 0 or region_b.size == 0: 
            break
        new_th = (region_a.mean() + region_b.mean()) / 2.0
        if abs(new_th - th) < 0.5: 
            break
        th = new_th
    return th

def detect_lanes_gold(bev_image):
    """
    Rilevamento linee percorsa sulla BEV. Implementazione filtri GOLD (dark-light-dark).
    """
    # 1. Scala di grigi e smoothing
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    
    # 2. Filtro spaziale GOLD (Esalta transizioni "Scuro - Chiaro - Scuro")
    # Tau = 12 pixel corrisponde a circa 15cm in BEV.
    # Un ridge filter rileva oggetti di spessore circa 2*tau.
    tau = 12
    left_diff = np.zeros_like(blur)
    right_diff = np.zeros_like(blur)
    
    # Calcoliamo il salto di luce tra il centro e i rispettivi bordi sx/dx
    left_diff[:, tau:] = blur[:, tau:] - blur[:, :-tau]
    right_diff[:, :-tau] = blur[:, :-tau] - blur[:, tau:]
    
    # IL MIRACOLO MATEMATICO (Vero Ridge Filter):
    # Un pixel fa parte di una striscia stradale SOLO SE è più chiaro SIA rispetto a sinistra, SIA a destra.
    gold_filter = np.minimum(left_diff, right_diff)
    gold_filter[gold_filter < 0] = 0
    
    # 3. Soglia iterativa (Adattiva alla luce) invece di fissa a 20
    # Usiamo un margine (es. 0.8) per essere sicuri di prendere anche linee sbiadite all'ombra
    th_opt = iterative_thresholding(gold_filter)
    th_final = max(12, th_opt * 0.8) # PARAMETRO: abbassa (es. 0.6) se le linee nelle zone d'ombra spariscono

    _, thresh = cv2.threshold(gold_filter, th_final, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    # FILTRO SULLO SPESSORE E ORIENTAMENTO (Morfologia Selettiva)
    # 1. Eliminiamo rumore orizzontale o spilli troppo sottili (crepe)
    # PARAMETRO: Se una linea sbiadita sparisce dopo questo, riduci a (3, 1) o (2, 1)
    kernel_thickness = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_thickness)
    
    # 2. Rinforziamo la verticalità (fondamentale per linee sbiadite)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_vertical)
    
    # Puliamo piccolo rumore residuo
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    # FILTRO PER ESCLUDERE OGGETTI GRANDI (veicoli, pullman)
    # Identifichiamo zone "blackout" causate da oggetti grandi che riempiono la BEV verticalmente.
    # Un veicolo davanti proiettato in BEV appare come una colonna scura verticale di pixel.
    # Contiamo i pixel "accesi" per ogni colonna.
    # PARAMETRO CRITICO: 0.60 era troppo basso! Una linea continua dritta occupa il 100% della colonna
    # (è verticale da cima a fondo). Se si supera l'110% è impossibile, ma se usiamo 0.95
    # evitiamo di auto-cancellare le linee rettilinee scambiandole per grandi ostacoli neri.
    raw_occupancy = np.sum(thresh // 255, axis=0)
    max_occupancy = 800  # Altezza della BEV
    occupancy_threshold = 0.95 * max_occupancy
    
    # Creiamo una maschera che scarta colonne in cui ci sono troppi pixel bianchi
    # (tipicamente linee spurie causate dagli oggetti)
    obstacle_mask = raw_occupancy > occupancy_threshold
    
    # Dilatiamo leggermente la maschera per non perdere linee ai bordi dell'ostacolo
    obstacle_mask = cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((1, 15)), iterations=1) > 0
    
    # Applichiamo la maschera all'immagine threshold
    # Dove c'è un ostacolo, poniamo i valori a 0 (nero)
    for x in range(BEV_SIZE[0]):
        if obstacle_mask[x]:
            thresh[:, x] = 0

    # FILTRO ATTRAVERSAMENTI PEDONALI:
    # Le strisce pedonali accendono molte colonne nella stessa riga.
    # Una vera lane invece occupa poche colonne per riga.
    row_occupancy = np.sum(thresh // 255, axis=1)
    row_occ_threshold = int(0.18 * BEV_SIZE[0])
    crosswalk_rows = row_occupancy > row_occ_threshold
    if np.any(crosswalk_rows):
        # Allarghiamo leggermente la banda rimossa per includere il contorno della striscia.
        crosswalk_rows = cv2.dilate(
            crosswalk_rows.astype(np.uint8).reshape(-1, 1),
            np.ones((11, 1), np.uint8),
            iterations=1
        ).reshape(-1) > 0
        thresh[crosswalk_rows, :] = 0

    # FILTRO ORIZZONTALE AGGIUNTIVO:
    # Estrae blocchi prevalentemente orizzontali (tipici delle zebrature) e li rimuove.
    # Le lane verticali vere non sopravvivono a questo kernel orizzontale lungo.
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    if np.count_nonzero(horizontal_mask) > 0:
        # Dilatazione lieve per togliere anche i bordi delle strisce pedonali.
        horizontal_mask = cv2.dilate(horizontal_mask, np.ones((5, 5), np.uint8), iterations=1)
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(horizontal_mask))
    
    # 4. Istogramma per trovare le linee: sommiamo le colonne
    # Usiamo solo la parte ALTA/MEDIA della BEV per evitare che i bordi verticali
    # dell'auto davanti (molto vicina) vengano presi come linee corsia.
    histogram_h = int(0.72 * BEV_SIZE[1])
    raw_histogram = np.sum(thresh[:histogram_h, :] // 255, axis=0)
    
    # Tagliamo via i margini estremi molto meno!
    # L'auto davanti a noi ci costringe ad avere linee molto laterali.
    raw_histogram[:10] = 0
    raw_histogram[-20:] = 0 # Prima tagliava via 60px! Se la destra è sul bordo la perdevamo.
    
    # Raggruppiamo pixel vicini con una convoluzione PIÙ LARGA (da 10 a 40)
    # PARAMETRO: Se la linea è "sghemba" (storta/diagonale) i pixel si sparpagliano su tante X diverse.
    # Una convoluzione più larga raggruppa pixel anche se la linea non è perfettamente dritta verticale!
    histogram = np.convolve(raw_histogram, np.ones(40), mode='same').astype(np.float32)
    
    # 5. Cerca i picchi nell'istogramma (corsia sinistra e destra)
    midpoint = BEV_SIZE[0] // 2
    
    # Validazione larghezza corsia
    min_lane_width = 250 # Aumentato! Le ruote dell'auto o riflessi interni sono più vicini di 250px!
    max_lane_width = 600 # [] distanza massima tra due linee rilevabili (distanza tra quella di sx e quella di dx)

    # Soglia minima per l'istogramma
    # PARAMETRO: rialzato a 6 per non prendere polvere sull'asfalto come linea
    threshold_hist = 6

    lanes = [] 

    def lane_metrics(x):
        # ROI più larga per seguire le CURVE senza perderle (da +-25 a +-55)
        # PARAMETRO: Aumentato a 75 perchè la strada curva in alcune foto e la linea "esce" dall'area di scansione
        strip = thresh[:, max(0, x-75):min(BEV_SIZE[0], x+75)]
        row_on = np.sum(strip, axis=1) > 0
        ratio = np.count_nonzero(row_on) / float(len(row_on))

        # Chiusura per gap piccoli (usura della vernice)
        row_on_uint8 = row_on.astype(np.uint8) * 255
        # PARAMETRO: Se ti rileva una tratteggiata quando è continua rovinata, aumenta (40, 1) per fondere meglio i buchi
        # Alzato a 25,1 per coprire buchi leggermente più grandi (es. strisce o crepe nel mezzo della continua)
        kernel_fill = np.ones((25, 1), np.uint8)
        row_on_filled = cv2.morphologyEx(row_on_uint8, cv2.MORPH_CLOSE, kernel_fill) > 0

        # Conteggio segmenti + lunghezza del tratto continuo più lungo.
        # Serve a non classificare "tratteggiata" una continua con pochi buchi locali.
        runs = row_on_filled.astype(np.int8)
        padded = np.pad(runs, (1, 1), mode='constant', constant_values=0)
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        segments = int(len(starts))
        longest_run = int(np.max(ends - starts)) if segments > 0 else 0
        longest_ratio = longest_run / float(len(row_on_filled))

        # Continuita locale nel tratto vicino all'auto (parte bassa BEV),
        # piu affidabile quando in alto ci sono ombre/occlusioni.
        near_start = int(0.55 * len(row_on_filled))
        near_ratio = np.count_nonzero(row_on_filled[near_start:]) / float(len(row_on_filled) - near_start)

        # Presenza nel tratto medio-basso (evita falsi da bordi auto/attraversamenti,
        # che tipicamente sono localizzati e non attraversano verticalmente la scena).
        mid_start = int(0.30 * len(row_on_filled))
        mid_end = int(0.80 * len(row_on_filled))
        mid_ratio = np.count_nonzero(row_on_filled[mid_start:mid_end]) / float(mid_end - mid_start)
        return ratio, segments, longest_ratio, near_ratio, mid_ratio

    def is_valid_lane_candidate(x, ratio, longest_ratio, mid_ratio):
        # Requisiti base: un minimo di presenza globale e nel tratto medio-basso.
        if ratio <= 0.05 or mid_ratio <= 0.28:
            return False

        # Ai bordi estremi richiediamo evidenza più forte.
        # Riduce i falsi da fiancate/gomme auto parcheggiate che in BEV sembrano "linee".
        edge_margin = 80
        if x < edge_margin or x > (BEV_SIZE[0] - edge_margin):
            if ratio <= 0.12:
                return False
            if mid_ratio <= 0.42:
                return False
            if longest_ratio <= 0.22:
                return False

        return True

    # Cerchiamo il picco massimo nella METÀ SINISTRA della BEV
    left_half = histogram[50:midpoint]      # [] GUARDA SOLO DA 50PX IN POI A SX
    if left_half.size > 0:
        left_peak = float(np.max(left_half))
        if left_peak > threshold_hist:
            left_x = int(np.argmax(left_half)) + 50
            ratio, segments, longest_ratio, near_ratio, mid_ratio = lane_metrics(left_x)
            if is_valid_lane_candidate(left_x, ratio, longest_ratio, mid_ratio):
                # Se nel tratto vicino all'auto la linea è molto continua,
                # privilegiarla come continua anche con qualche buco lontano.
                is_dashed = (near_ratio < 0.72) and ((ratio < 0.56) or ((segments > 4) and (longest_ratio < 0.40)))
                lane_type = "tratteggiata" if is_dashed else "continua"
                lanes.append({'x': left_x, 'type': lane_type, 'score': left_peak})

    # Cerchiamo il picco massimo nella METÀ DESTRA della BEV
    right_half = histogram[midpoint:750]
    if right_half.size > 0:
        right_peak = float(np.max(right_half))
        if right_peak > threshold_hist:
            right_x = int(np.argmax(right_half)) + midpoint
            ratio, segments, longest_ratio, near_ratio, mid_ratio = lane_metrics(right_x)
            if is_valid_lane_candidate(right_x, ratio, longest_ratio, mid_ratio): 
                is_dashed = (near_ratio < 0.72) and ((ratio < 0.56) or ((segments > 4) and (longest_ratio < 0.40)))
                lane_type = "tratteggiata" if is_dashed else "continua"
                lanes.append({'x': right_x, 'type': lane_type, 'score': right_peak})

    # Filtro: scartiamo il picco più debole se è TROPPO inferiore all'altro
    # PARAMETRO: abbassato da 0.20 a 0.05. Se la linea tratteggiata spariva perché
    # l'altra continua era "troppo forte", così gli permettiamo di sopravvivere!
    if len(lanes) == 2:
        s0, s1 = lanes[0]['score'], lanes[1]['score']
        if min(s0, s1) < 0.10 * max(s0, s1):            # [] DA MODIFICARE A 0.05 OPPURE ABBASSARE SE NON RILEVA LINEE
            keep_idx = 0 if s0 >= s1 else 1
            lanes = [lanes[keep_idx]]

    # VALIDAZIONE: la larghezza tra le linee deve essere plausibile per una corsia
    if len(lanes) == 2:
        width = lanes[1]['x'] - lanes[0]['x']
        if width < min_lane_width or width > max_lane_width:
            # SE SONO TROPPO VICINE TRA LORO E IN MEZZO ALLA CARREGGIATA -> E' SICURAMENTE UN'AUTO CHE ATTRAVERSA!
            # Rimuoviamole entrambe per evitare linee sui cerchioni
            if width < min_lane_width and lanes[0]['x'] > 200 and lanes[1]['x'] < 600:
                lanes = []
            else:
                # INVECE DI CANCELLARE ENTRAMBE LE LINEE, TENIAMO QUELLA PIÙ FORTE
                # Spesso una linea è vera e forte, l'altra è un finto posizionamento
                if lanes[0]['score'] >= lanes[1]['score']:
                    lanes = [lanes[0]]
                else:
                    lanes = [lanes[1]]

    # Ripulisci campi di debug non necessari.
    for lane in lanes:
        lane.pop('score', None)
        lane.pop('ratio', None)
        
    return lanes, thresh

import urllib.request
import os

# YOLO CONFIGURATION
YOLO_CFG = "yolov3-tiny.cfg"
YOLO_WEIGHTS = "yolov3-tiny.weights"

def load_yolo():
    """Carica il modello YOLOv3-tiny. Scarica i pesi se non presenti."""
    if not os.path.exists(YOLO_CFG):
        print("Scaricamento yolov3-tiny.cfg...")
        req = urllib.request.Request("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(YOLO_CFG, 'wb') as out_file:
            out_file.write(response.read())
    if not os.path.exists(YOLO_WEIGHTS):
        print("Scaricamento yolov3-tiny.weights (potrebbe richiedere qualche secondo)...")
        req = urllib.request.Request("https://pjreddie.com/media/files/yolov3-tiny.weights", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(YOLO_WEIGHTS, 'wb') as out_file:
            out_file.write(response.read())
    
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# Caricamento del modello YOLO globale per efficienza
print("Inizializzazione modello YOLO...")
YOLO_NET = load_yolo()
YOLO_CLASSES = {0: 'pedestrian', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}
print("YOLO pronto.")

def detect_obstacles(frame):
    """
    Rileva ostacoli (es. auto, pedoni) utilizzando YOLO (You Only Look Once) 
    sull'estrazione delle bounding boxes (x, y, w, h) per come spiegato nelle slide della Professoressa.
    Applica poi la conversione matematica per trovare la profondità/distanza sull'asse Z.
    """
    h_frame, w_frame = frame.shape[:2]
    
    # 1. Conversione dell'immagine in un blob per passarla alla rete neurale convoluzionale
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    YOLO_NET.setInput(blob)
    
    # Nomi dei livelli di output
    layer_names = YOLO_NET.getLayerNames()
    output_layers = [layer_names[i - 1] for i in YOLO_NET.getUnconnectedOutLayers()]
    
    # 2. Forward pass: estraiamo le previsioni del modello (tensori)
    layer_outputs = YOLO_NET.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # 3. Estrazione Bounding Box (x, y, w, h) e Confidence (C) come descritto nelle slide (Formulazione 5 predizioni)
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filtriamo in base alle classi che ci interessano (veicoli e pedoni) e probabilità > 30%
            if confidence > 0.3 and class_id in YOLO_CLASSES:
                # Coordinate YOLO restituite come percentuali rispetto all'immagine
                center_x = int(detection[0] * w_frame)
                center_y = int(detection[1] * h_frame)
                w = int(detection[2] * w_frame)
                h = int(detection[3] * h_frame)
                
                # Ricaviamo x, y top-left
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # 4. Applichiamo Non-Maximum Suppression (NMS) per rimuovere i bounding box ridondanti sullo stesso oggetto
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            
            # Formule di Distanza in Z (Ground Plane Projection)
            # Dalla formula della prospettiva: v = fy * (Y / Z) + cy
            # Pertanto: Z = fy * Y / (v - cy), dove v è il pixel in basso, Y è l'altezza della telecamera
            v_bottom = y + h
            cx, cy = PRINCIPAL_POINT
            fx, fy = FOCAL_LENGTH
            
            distance = -1
            # Valutiamo la distanza solo se l'auto tocca l'asfalto sotto la linea dell'orizzonte (cy)
            if v_bottom > cy:
                distance = fy * CAMERA_POSITION_Z / float(v_bottom - cy)
            
            results.append({
                'class_name': YOLO_CLASSES[class_ids[i]],
                'confidence': confidences[i],
                'box': (x, y, w, h),
                'distance': distance
            })
            
    # FUSIONE DEI BOUNDING BOX MULTIPLI SULLO STESSO OSTACOLO
    # Risolve il problema del "cerchione" o della "targa" rilevata come auto separata
    # Quando l'Intersection over Minimum Area (IoA) è alta (> 60%), YOLO ha rilevato un pezzo dell'auto e l'auto intera.
    merged_results = []
    for r in results:
        x, y, w, h = r['box']
        area_r = w * h
        is_merged = False
        
        for mr in merged_results:
            mx, my, mw, mh = mr['box']
            area_m = mw * mh
            
            # Calcolo dell'intersezione
            ix = max(x, mx)
            iy = max(y, my)
            iw = min(x + w, mx + mw) - ix
            ih = min(y + h, my + mh) - iy
            
            if iw > 0 and ih > 0:
                inter_area = iw * ih
                # Calcola l'overlap basato sull'area PIÙ PICCOLA (IoA) anziché l'Unione (IoU)
                if inter_area / min(area_r, area_m) > 0.6:
                    # Fonde i due rettangoli (prendendo i bordi più estremi)
                    new_x = min(x, mx)
                    new_y = min(y, my)
                    new_w = max(x + w, mx + mw) - new_x
                    new_h = max(y + h, my + mh) - new_y
                    
                    mr['box'] = (new_x, new_y, new_w, new_h)
                    
                    # Ricalcola la distanza sul bounding box fuso
                    v_bottom_new = new_y + new_h
                    if v_bottom_new > cy:
                        mr['distance'] = fy * CAMERA_POSITION_Z / float(v_bottom_new - cy)
                    
                    mr['confidence'] = max(r['confidence'], mr['confidence'])
                    is_merged = True
                    break
        
        if not is_merged:
            merged_results.append(r)
            
    return merged_results

def draw_yolo_obstacles(frame, obstacles):
    """
    Disegna i bounding box di YOLO e la distanza calcolata sul frame originale.
    """
    frame_out = frame.copy()
    for obs in obstacles:
        x, y, w, h = obs['box']
        dist = obs['distance']
        
        # Gestione visualizzazione info
        label = f"{obs['class_name']} {obs['confidence']:.2f}"
        dist_label = f"{dist:.1f}m" if dist > 0 else "N/A"
        text = f"{label} | {dist_label}"
        
        # Colore identificativo (rosso)
        color = (0, 0, 255)
        
        # Disegna il rettangolo
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), color, 2)
        
        # Sfondo per il testo
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame_out, (x, y - 25), (x + text_size[0], y), color, -1)
        # Testo
        cv2.putText(frame_out, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return frame_out

def draw_lanes_on_original(frame, lanes, inv_matrix):
    """
    Riporta le linee delimitarici dalla BEV sull'immagine originale 3D (Requirement 2 & 1)
    """
    frame_lanes = frame.copy()
    
    # Scende lungo la Y della BEV per disegnare la linea a segmenti
    y_step = 20
    for lane in lanes:
        x = lane['x']
        l_type = lane['type']
        
        # Tutte le strisce devono essere verdi (su richiesta)
        color = (0, 255, 0) 
        
        # Creiamo i punti della linea nella BEV partendo dal basso all'alto
        pts_bev = []
        for y in range(BEV_SIZE[1], 0, -y_step):
            # Se la linea è tratteggiata, saltiamo dei pezzi per renderla visivamente "dashed"
            if l_type == "tratteggiata" and (y // 40) % 2 == 0:
                continue
            pts_bev.append([x, y])
            
        if not pts_bev:
            continue
            
        # Trasformiamo l'array di punti (N, 1, 2) format per cv2
        pts_bev = np.array(pts_bev, dtype=np.float32).reshape(-1, 1, 2)
        
        # Mappiamo i punti dalla BEV all'immagine prospettica originale
        pts_orig = cv2.perspectiveTransform(pts_bev, inv_matrix)
        
        # Disegniamo la poli-linea (nel caso sia un array di punti)
        pts_orig = np.int32(pts_orig)
        for i in range(len(pts_orig) - 1):
            pt1 = tuple(pts_orig[i][0])
            pt2 = tuple(pts_orig[i+1][0])
            # Ignoriamo salti enormi per linee tratteggiate (se distanti + di 1 segmento non colleghiamo)
            if l_type == "tratteggiata" and abs(pts_bev[i][0][1] - pts_bev[i+1][0][1]) > y_step + 5:
                continue
            cv2.line(frame_lanes, pt1, pt2, color, thickness=8, lineType=cv2.LINE_AA)
            
    return frame_lanes

def main():
    if len(sys.argv) < 2:
        print("Uso: python run_gold.py '<path_to_images>'")
        sys.exit(1)
        
    search_path = sys.argv[1]
    image_files = sorted(glob.glob(search_path))
    
    if not image_files:
        print(f"Nessuna immagine trovata nel percorso: {search_path}")
        sys.exit(1)
        
    print(f"Trovate {len(image_files)} immagini. Inizio elaborazione...")

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        bev = apply_ipm(frame)
        bev_gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        detected_lanes, thresh = detect_lanes_gold(bev)
        
        # Mandatory Requirement 3.b: Se non ci sono corsie stampiamo "No lanes found" sulla BEV
        if not detected_lanes:
            cv2.putText(bev, "No lanes found", (200, BEV_SIZE[1] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            frame_lanes = frame.copy()
            cv2.putText(frame_lanes, "No lanes found", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        else:
            frame_lanes = frame.copy()

            # Disegniamo solo linee reali rilevate (specifica assignment).
            frame_lanes = draw_lanes_on_original(frame_lanes, detected_lanes, INV_IPM_MATRIX)
            for lane in detected_lanes:
                x = lane['x']
                c = (0, 255, 0)
                tipo = "Continua" if lane['type'] == "continua" else "Tratt."
                cv2.putText(bev, tipo, (x + 10, BEV_SIZE[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
                if lane['type'] == "tratteggiata":
                    for y in range(0, BEV_SIZE[1], 40):
                        if (y // 40) % 2 != 0:
                            cv2.line(bev, (x, y), (x, y + 40), c, 4)
                else:
                    cv2.line(bev, (x, 0), (x, BEV_SIZE[1]), c, 4)
                
        # Opt 3: Ostacoli con YOLO (Logica delle slide)
        detected_obstacles = detect_obstacles(frame)
        
        # Filtriamo solo gli ostacoli "davanti a noi" (nella nostra corsia)
        # mappando il centro-base del bounding box nella Bird's Eye View.
        ego_obstacles = []
        if detected_obstacles:
            # Calcoliamo i limiti della nostra corsia nella BEV
            # Di default consideriamo un corridoio centrale (da 200 a 600 in una BEV da 800)
            left_bound = 200
            right_bound = 600
            if len(detected_lanes) == 2:
                left_bound = min(detected_lanes[0]['x'], detected_lanes[1]['x'])
                right_bound = max(detected_lanes[0]['x'], detected_lanes[1]['x'])
            elif len(detected_lanes) == 1:
                lx = detected_lanes[0]['x']
                if lx < 400: # Linea sinistra trovata
                    left_bound = lx
                    right_bound = lx + 350
                else:        # Linea destra trovata
                    right_bound = lx
                    left_bound = lx - 350
                    
            # Manteniamo un margine bassissimo o nullo: le auto in corsia stanno centrali (CIPV).
            # Evitiamo di allargare troppo altrimenti includiamo auto della corsia a fianco
            # (soprattutto in curva o per colpa del bounding box largo dovuto alla prospettiva).
            left_bound -= 0
            right_bound += 0

            for obs in detected_obstacles:
                if obs['distance'] > 0: # Saltiamo ostacoli messi sopra l'orizzonte
                    x, y, w, h = obs['box']
                    # Usiamo il centro in basso del bounding box
                    u_center = x + w / 2.0
                    v_bottom = y + h
                    # Proiettiamo il punto (u, v) nella BEV per trovare la posizione (X, Y) reale sulla strada
                    pts = np.array([[[u_center, float(v_bottom)]]], dtype=np.float32)
                    bev_pts = cv2.perspectiveTransform(pts, IPM_MATRIX)
                    bev_x = bev_pts[0][0][0]
                    
                    # Verifichiamo che il centroide appartenga *strettamente* alla nostra corsia
                    if left_bound <= bev_x <= right_bound:
                        ego_obstacles.append(obs)

        if ego_obstacles:
            # Troviamo l'ostacolo EGO più vicino (Closest In-Path Vehicle - CIPV)
            # Solo lui è il nostro vero ostacolo primario.
            closest_obs = min(ego_obstacles, key=lambda o: o['distance'])
            
            # Disegniamo i bounding box e le info a schermo SOLO per l'ostacolo più vicino (CIPV)
            frame_lanes = draw_yolo_obstacles(frame_lanes, [closest_obs])
            
            # Stampa l'allarme
            min_distance = closest_obs['distance']
            cv2.putText(frame_lanes, f"WARNING: OSTACOLO A {min_distance:.1f}m!", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
                
        print(f"[{os.path.basename(img_path)}] Lanes: {len(detected_lanes)} | Obstacles: {len(ego_obstacles)}")

        # Per visualizzare meglio ridimensioniamo
        frame_resized = cv2.resize(frame_lanes, (960, 540))
        bev_resized = cv2.resize(bev, (540, 540)) # Facciamo in modo che entrino bene su schermi bassi
        
        # Mettiamo affiancati il frame originale "aumentato" e la BEV
        # L'interfaccia divisa a metá risulta molto più professionale
        # Original (960x540) accostato a BEV (540x540). Padding BEV: 960+540 = 1500
        combined = np.zeros((540, 1500, 3), dtype=np.uint8)
        combined[:, :960] = frame_resized
        combined[:, 960:] = bev_resized
        
        cv2.imshow("Driver View + Bird's Eye (Press 'q' or Space)", combined)
        
        # Premi 'q' per uscire, barra spaziatrice o freccia destra per passare all'immagine successiva
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
