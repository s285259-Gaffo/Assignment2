# Guida alla Modifica dei Parametri (Algoritmo GOLD)

Questo file contiene le indicazioni su quali parametri modificare nel file `run_gold.py` quando l'algoritmo di rilevamento fallisce in specifiche situazioni (es. curve, ombre, usura dell'asfalto).

---

### 1. La linea sparisce improvvisamente (specialmente all'ombra o al buio)
Questo accade perché la luminosità globale cala e il contrasto non è sufficiente per superare la soglia.
*   **Cosa cercare nel codice:**
    ```python
    th_final = max(12, th_opt * 0.8) # PARAMETRO (soglia iterativa)
    ```
*   **Come modificarlo:** Abbassa il moltiplicatore (es. da `0.8` a `0.6` o `0.5`). 
*   **Effetto:** Stai dicendo al programma: *"Anche se la soglia ideale calcolata è alta, accetta anche segnali più deboli (fino al 50%-60%), perché l'ombra sta nascondendo la linea"*.

---

### 2. Le linee tratteggiate spariscono o diventano troppo deboli
Questo avviene quando la linea è molto sottile o estremamente sbiadita.
*   **Cosa cercare nel codice:**
    ```python
    tau = 12 # Spessore linea atteso
    ```
*   **Come modificarlo:** Abbassa `tau` (es. `8` o `10`). Questo fa sì che il filtro GOLD cerchi linee di spessore inferiore.
*   **Altro parametro da controllare:**
    ```python
    kernel_thickness = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    ```
*   **Come modificarlo:** Questa morfologia cancella il rumore orizzontale. Se la linea laterale sbiadita sparisce dopo questo passaggio, riducilo a `(2, 1)` per essere più permissivo (oppure commenta la riga per testare).

---

### 3. Vede una linea continua molto usurata e crede sia tratteggiata
C'è un'usura fortissima sulla riga che ne mangia dei pezzi verticalmente, il codice la legge come "buchi di tratteggio".
*   **Cosa cercare nel codice:** (All'interno della funzione `lane_metrics(x)`)
    ```python
    kernel_fill = np.ones((40, 1), np.uint8)
    ```
*   **Come modificarlo:** Aumenta l'altezza del kernel, ad esempio a `(60, 1)` o `(80, 1)`. 
*   **Effetto:** Spalma virtualmente la linea verso l'alto e verso il basso prima di misurarla, fondendo i "buchi" creati dall'usura.

---

### 4. Nelle curve a gomito o curve molto strette perde la linea
L'algoritmo guarda una "striscia" verticale dritta. Se la curva è molto pronunciata, la linea esce da questa zona di osservazione.
*   **Cosa cercare nel codice:** (All'interno della funzione `lane_metrics(x)`)
    ```python
    strip = thresh[:, max(0, x-55):min(BEV_SIZE[0], x+55)]
    ```
*   **Come modificarlo:** Allarga l'intervallo di ricerca attorno al centro `x`. Cambia `x-55` e `x+55` in valori più ampi, ad esempio `x-80` e `x+80` o anche `100`.
*   **Effetto:** Il corridoio di verifica diventa molto più largo, permettendo alla curva di restare all'interno del campo visivo.

---

### 5. Problemi con gli Ostacoli (Pedoni, Veicoli)

**Caso A: L'algoritmo vede ostacoli che non ci sono (Falsi Positivi - es. crepe scure, cartelli, ombre nette)**
*   **Cosa cercare nel codice:** (In `detect_obstacles()`)
    ```python
    photometric_anomaly = (np.abs(...) > 25)
    ...
    min_area = 400 
    ```
*   **Come modificarlo:** Aumenta il valore `25` (es. `35` o `40`) in modo che pretenda uno stacco di colore nettissimo con l'asfalto. Aumenta `min_area` da `400` a `800` (l'oggetto deve essere enorme per essere considerato un ostacolo).

**Caso B: L'algoritmo NON vede un ostacolo reale (Falsi Negativi - es. pedone scuro su asfalto scuro)**
*   **Come modificarlo:** Fai l'esatto contrario. Riduci l'anomalia fotometrica (es. da `25` a `15`) per cogliere anche leggere variazioni di colore, e abbassa `min_area` (es. a `200`) per rilevare oggetti più piccoli o sottili.
