import cv2
import numpy as np
import sys
import glob

# Parametri della telecamera forniti dall'assignment
IMAGE_SIZE = (1920, 1080) # (width, height)
PRINCIPAL_POINT = (970, 483) # (cx, cy)
FOCAL_LENGTH = (1970, 1970) # (fx, fy)
CAMERA_POSITION_Z = 1.6600 # Altezza della camera dal suolo (metri)
PITCH = 0 # Radianti

def get_ipm_matrix():
    """
    Calcola la matrice di trasformazione prospettica per l'IPM.
    Mappiamo 4 punti 3D dal piano della strada (Z_world = profondità, X_world = larghezza)
    sul piano immagine (u, v) e definiamo un rettangolo di destinazione (Bird's Eye View).
    """
    # Definiamo i confini dell'area da guardare sulla strada (in metri rispetto alla telecamera)
    z_far = 25.0   # Riduciamo la profondità a 25 metri per evitare distorsioni e orizzonte
    z_near = 6.0   # Partiamo a guardare da 6 metri dal muso dell'auto
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
INV_IPM_MATRIX = np.linalg.inv(IPM_MATRIX)

def apply_ipm(image):
    return cv2.warpPerspective(image, IPM_MATRIX, BEV_SIZE, flags=cv2.INTER_LINEAR)

def detect_lanes_gold(bev_image):
    """
    Rilevamento linee percorsa sulla BEV. Implementazione filtri GOLD (dark-light-dark).
    """
    # 1. Scala di grigi e smoothing
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    
    # 2. Filtro spaziale GOLD (Esalta transizioni "Scuro - Chiaro - Scuro")
    # Ignora in automatico strisce pedonali larghe e oggetti enormi
    # Nella BEV larga 800px e 5 metri di strada, una striscia di 15cm occupa circa 24 pixel.
    # Quindi usiamo una distanza spaziale tau di circa 12 pixel
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
    
    # 3. Soglia sul filtro per ottenere un'immagine binaria chiara
    # Più permissiva per linee tratteggiate deboli in scena urbana.
    _, thresh = cv2.threshold(gold_filter, 18, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    # Puliamo piccolo rumore residuo
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    # 4. Istogramma per trovare le linee: sommiamo le colonne
    raw_histogram = np.sum(thresh // 255, axis=0)
    
    # Tagliamo via i margini estremi (ancora meno aggressivo a sinistra)
    raw_histogram[:20] = 0
    raw_histogram[-80:] = 0
    
    # Raggruppiamo pixel vicini
    histogram = np.convolve(raw_histogram, np.ones(15), mode='same')
    
    # 5. Cerca i picchi nell'istogramma (corsia sinistra e destra)
    midpoint = BEV_SIZE[0] // 2
    
    # Soglia minima base molto permissiva.
    threshold_hist = 10

    lanes = [] # Conterrà dizionari con x e 'type'

    def lane_metrics(x):
        # Analisi verticale su ROI: aumentiamo la finestra di analisi per catturare più tratti.
        strip = thresh[:, max(0, x-10):min(BEV_SIZE[0], x+10)]
        roi = strip[100:750, :] 
        row_on = np.sum(roi, axis=1) > 0
        ratio = np.count_nonzero(row_on) / float(len(row_on))

        # Numero di segmenti verticali accesi.
        changes = np.diff(row_on.astype(np.uint8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if row_on[0]:
            starts = np.r_[0, starts]
        if row_on[-1]:
            ends = np.r_[ends, len(row_on)]

        seg_lengths = (ends - starts) if len(starts) == len(ends) and len(starts) > 0 else np.array([], dtype=np.int32)
        segments = int(len(seg_lengths))
        avg_seg_len = float(np.mean(seg_lengths)) if segments > 0 else 0.0
        return ratio, segments, avg_seg_len

    # Lato sinistro
    left_side = histogram[20:midpoint]
    left_peak = float(np.max(left_side)) if left_side.size else 0.0
    left_thr = max(threshold_hist, 0.25 * left_peak)
    if left_peak > left_thr:
        left_x = int(np.argmax(left_side)) + 20
        ratio, segments, avg_seg_len = lane_metrics(left_x)
        # Logica ultra-permissiva per tratteggiata: basta avere un ratio basso (< 55%) 
        # o più di un segmento (evita che piccoli buchi convertano tutto in continua).
        lane_type = "tratteggiata" if (ratio < 0.60 or segments >= 2) else "continua"
        lanes.append({'x': left_x, 'type': lane_type, 'score': left_peak, 'ratio': ratio})

    # Lato destro: estendiamo ricerca fino a 760 per non perdere corsie vicine al bordo.
    right_side = histogram[midpoint:760]
    right_peak = float(np.max(right_side)) if right_side.size else 0.0
    right_thr = max(threshold_hist, 0.25 * right_peak)
    if right_peak > right_thr:
        right_x = int(np.argmax(right_side)) + midpoint
        ratio, segments, avg_seg_len = lane_metrics(right_x)  
        lane_type = "tratteggiata" if (ratio < 0.60 or segments >= 2) else "continua"
        lanes.append({'x': right_x, 'type': lane_type, 'score': right_peak, 'ratio': ratio})

    # Se un candidato e' debolissimo rispetto al lato opposto, scartalo (tipico rumore locale).
    if len(lanes) == 2:
        s0, s1 = lanes[0]['score'], lanes[1]['score']
        if min(s0, s1) < 0.15 * max(s0, s1):
            keep_idx = 0 if s0 >= s1 else 1
            lanes = [lanes[keep_idx]]

    # VALIDAZIONE GEOMETRICA più permissiva su scene urbane/intersezioni.
    if len(lanes) == 2:
        width = lanes[1]['x'] - lanes[0]['x']
        if width < 220 or width > 760:
            lanes = []

    # Ripulisci campi di debug non necessari.
    for lane in lanes:
        lane.pop('score', None)
        lane.pop('ratio', None)
        
    return lanes, thresh

def detect_obstacles(bev_gray, lanes):
    """
    Rileva ostacoli (es. un'auto) all'interno del corridoio formato dalle due corsie.
    Se le corsie mancano o siamo ad un incrocio nudo, proietta un corridoio virtuale di salvataggio.
    Restituisce la distanza stimata in metri, se c'è un ostacolo.
    """
    lane_x_positions = [int(l['x']) for l in lanes]

    if len(lanes) == 2:
        left_x = lanes[0]['x']
        right_x = lanes[1]['x']
    elif len(lanes) == 1:
        # Se ce n'è una sola in BEV cerchiamo di indovinare la seconda distanziata di 560px (~3.5m)
        if lanes[0]['x'] < BEV_SIZE[0] // 2:
            left_x = lanes[0]['x']
            right_x = left_x + 560 
        else:
            right_x = lanes[0]['x']
            left_x = right_x - 560
    else:
        # Nessuna linea (es. incrocio): CORRIDOIO VIRTUALE CENTRALE DI SALVATAGGIO:
        left_x = 120
        right_x = 680
        
    # Limiti validi per il bounding box BEV
    left_x = max(0, left_x)
    right_x = min(BEV_SIZE[0], right_x)
    
    corridor_width = right_x - left_x - 60
    if corridor_width <= 0:
        return None, None, None
        
    corridor = bev_gray[:, int(left_x + 30): int(right_x - 30)]
    
    # Maschera ostacoli robusta: combiniamo bordi + anomalia fotometrica rispetto alla riga stradale.
    # Questo riduce i falsi positivi su crepe sottili e recupera pedoni/auto in BEV distorta.
    blur_corridor = cv2.GaussianBlur(corridor, (5, 5), 0)
    edges = cv2.Canny(blur_corridor, 40, 120)

    row_median = np.median(blur_corridor, axis=1, keepdims=True)
    photometric_anomaly = (np.abs(blur_corridor.astype(np.int16) - row_median.astype(np.int16)) > 18).astype(np.uint8) * 255

    combined_mask = cv2.bitwise_or(edges, photometric_anomaly)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Sopprimiamo bande orizzontali (es. strisce pedonali) mantenendo strutture verticali.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, vertical_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask)

    obstacle_y = None
    obstacle_x = None
    best_bottom = -1
    min_area = int(corridor_width * 8)
    min_w = int(corridor_width * 0.12)
    min_h = 35

    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]

        # Filtri geometrici anti-falsi positivi: ignoriamo segni sottili dell'asfalto.
        if area < min_area:
            continue
        if w < min_w:
            continue
        if h < min_h:
            continue

        # Scarta pattern da linea: blob troppo slanciato verticalmente.
        if h > 3 * w:
            continue

        # Scarta componenti troppo "piatte": tipico pattern di strisce orizzontali a terra.
        if w > int(corridor_width * 0.75) and h < int(corridor_width * 0.18):
            continue

        # Scarta blob troppo vicino alle x delle linee rilevate (tipico falso positivo su lane marking).
        center_x_bev = int(left_x + 30 + x + (w // 2))
        if any(abs(center_x_bev - lx) < 50 for lx in lane_x_positions):
            continue

        bottom = y + h
        if bottom > best_bottom:
            best_bottom = bottom
            obstacle_x = x + (w // 2)

    if best_bottom > 0:
        obstacle_y = int(min(BEV_SIZE[1] - 1, best_bottom))
            
    if obstacle_y is not None:
        # Calcolo distanza. 
        # BEV mappa Y=800 a 6.0m e Y=0 a 25.0m (come definito in get_ipm_matrix)
        z_far = 25.0
        z_near = 6.0
        # Proporzione lineare nella BEV!
        distance = z_far - (obstacle_y / float(BEV_SIZE[1])) * (z_far - z_near)
        # Convertiamo x dal sistema del corridoio al sistema BEV completo.
        obstacle_x_bev = int(left_x + 30 + obstacle_x) if obstacle_x is not None else BEV_SIZE[0] // 2
        return obstacle_x_bev, obstacle_y, distance
        
    return None, None, None

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
                
        # Opt 3: Ostacoli (sempre attivo! Sia con, sia senza corsie visibili)
        obstacle_x, obstacle_y, distance = detect_obstacles(bev_gray, detected_lanes)
        if obstacle_y is not None:
            cv2.line(bev, (0, obstacle_y), (BEV_SIZE[0], obstacle_y), (0, 0, 255), 3)
            cv2.putText(bev, f"Ostacolo: {distance:.1f} m", (50, obstacle_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Usiamo il centro del blob ostacolo rilevato per proiezione precisa.
            pt_bev = np.array([[[obstacle_x, obstacle_y]]], dtype=np.float32)
            pt_orig = cv2.perspectiveTransform(pt_bev, INV_IPM_MATRIX)
            ou, ov = int(pt_orig[0][0][0]), int(pt_orig[0][0][1])

            # Testo di allarme sul frame (alto a sinistra) spostanto un po in basso
            cv2.putText(frame_lanes, f"WARNING: OSTACOLO A {distance:.1f}m!", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.circle(frame_lanes, (ou, ov), 20, (0, 0, 255), -1)
                
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
