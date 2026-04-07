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

# Calcoliamo la matrice una volta sola
IPM_MATRIX, BEV_SIZE = get_ipm_matrix()

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
    # Usando np.minimum, se una fiancata di un'auto fa "Scuro-Chiaro-Chiaro" (bordo e poi carrozzeria bianca netta)...
    # ...il salto di luce dal lato dritto sarà 0, e np.minimum porterà tutto a ZERO distruggendo l'auto!
    gold_filter = np.minimum(left_diff, right_diff)
    gold_filter[gold_filter < 0] = 0
    
    # 3. Soglia sul filtro per ottenere un'immagine binaria chiara
    _, thresh = cv2.threshold(gold_filter, 40, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    # Puliamo piccolo rumore residuo
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    # 4. Istogramma per trovare le linee: sommiamo le colonne
    raw_histogram = np.sum(thresh // 255, axis=0)
    
    # Tagliamo via i margini estremi sinistro e destro della visuale:
    # Lì ci finiscono spalmate ruote, palazzi e fari. Sappiamo che la corsia deve essere verso il centro.
    raw_histogram[:80] = 0
    raw_histogram[-80:] = 0
    
    # Applichiamo una finestra mobile (convoluzione) per raggruppare i pixel di colonne adiacenti (es. 15 pixel di spessore).
    # Questo salva le strisce vere che, a causa di leggere sterzate, "sbavano" su più colonne X adiacenti.
    histogram = np.convolve(raw_histogram, np.ones(15), mode='same')
    
    # 5. Cerca i picchi nell'istogramma (corsia sinistra e destra)
    midpoint = BEV_SIZE[0] // 2
    
    # Essendo corazzati dal filtro GOLD "spaziale", ora possiamo far passare la corsia anche se sbiadita!
    threshold_hist = 50 
    
    lanes_x = []
    
    # Lato sinistro (0 -> midpoint)
    left_side = histogram[:midpoint]
    if np.max(left_side) > threshold_hist:
        left_x = np.argmax(left_side)
        lanes_x.append(int(left_x))
        
    # Lato destro (midpoint -> larghezza totale)
    right_side = histogram[midpoint:]
    if np.max(right_side) > threshold_hist:
        right_x = np.argmax(right_side) + midpoint
        lanes_x.append(int(right_x))
        
    return lanes_x, thresh

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
        detected_lanes_x, thresh = detect_lanes_gold(bev)
        
        if not detected_lanes_x:
            cv2.putText(bev, "No lanes found", (200, BEV_SIZE[1] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            for x in detected_lanes_x:
                cv2.line(bev, (x, 0), (x, BEV_SIZE[1]), (0, 255, 0), 4)
                
        # Per visualizzare meglio ridimensiono il frame originale (è in 1080p, un po' grande)
        frame_resized = cv2.resize(frame, (960, 540))
        cv2.imshow("1. Original", frame_resized)
        cv2.imshow("2. Bird's Eye View (BEV)", bev)
        cv2.imshow("3. Threshold (Filtro B/N)", thresh)
        
        # Premi 'q' per uscire, barra spaziatrice per l'immagine successiva
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
