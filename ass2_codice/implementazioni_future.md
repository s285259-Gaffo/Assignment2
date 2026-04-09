# Limitazioni Architetturali e Implementazioni Future

Questo documento descrive i limiti strutturali dell'attuale versione dell'algoritmo GOLD e spiega quali **modifiche architetturali** (non solo cambi di parametri) bisognerà implementare se il rilevamento dovesse fallire sistematicamente in scenari complessi.

---

### 1. Assenza della "Sliding Window" (Punto debole sulle curve strette)
**Il Problema:**
Attualmente, l'algoritmo crea un istogramma sommando interamente i pixel sull'asse verticale (Y da 0 a 800) della Bird's Eye View (BEV). Se la strada fa una curva molto stretta, la linea si storta orizzontalmente: sommandola dall'alto in basso, i pixel bianchi si spalmano su troppe colonne X diverse, disintegrando il picco dell'istogramma (la linea sparisce).

**La Soluzione (Cosa implementare):**
Sostituire la logica dell'istogramma totale con l'algoritmo **Sliding Window**.
1. Dividere l'immagine verticalmente in "fette" (es. 10 finestre orizzontali da 80 pixel ciascuna).
2. Trovato il picco (la posizione X della linea) nella finestra più in basso, si passa alla finestra appena sopra.
3. Nella nuova finestra, si cerca il picco *solo* in un piccolo range (es. X ± 40) attorno al picco trovato sotto.
4. Si ripete fino in cima, "inseguendo" fedelmente la curvatura della linea.

---

### 2. Nessun Tracking Temporale (Sfarfallio sui video)
**Il Problema:**
Il codice elabora ogni immagine (frame) in maniera totalmente indipendente. Se in un singolo frame un riflesso fortissimo nasconde la corsia o genera un falso ostacolo, l'algoritmo lo mostrerà, causando un fastidioso sfarfallio visivo quando le immagini vengono riprodotte in sequenza come in un video.

**La Soluzione (Cosa implementare):**
Introdurre una logica di **Media Mobile (o Filtro di Kalman)** per conservare la "memoria" del frame precedente.
*   Quando calcoli la posizione `x` della linea attuale, non usi il valore nudo e crudo, ma lo filtri: 
    `x_finale = (0.7 * x_vecchio_frame) + (0.3 * x_calcolato_ora)`
*   In questo modo, se un frame perde la linea per un'ombra o un errore di 1 millisecondo, la linea non sparirà di colpo ma manterrà la sua ultima posizione nota in attesa del frame successivo.

---

### 3. Matrice IPM e il "Finto" parametro PITCH
**Il Problema:**
Nelle costanti in alto hai definito `PITCH = 0`. Tuttavia, nelle formule usate per costruire i punti sorgente `(u, v)` nella funzione `get_ipm_matrix()`, tu usi la formula prospettica semplice `u = fx * (x/z) + cx`.
Questa formula matematica **non prevede alcuna rotazione**: assume implicitamente che la telecamera sia matematicamente parallela al suolo.

**La Soluzione (Cosa implementare):**
Se il professore nei test futuri fornirà un dataset dove la telecamera è esplicitamente inclinata verso il basso (pitch non nullo, es. `PITCH = 0.05 radianti`), l'attuale calcolo della BEV sformerà pesantemente la strada.
*Bisognerà sostituire il calcolo dei 4 punti sorgente* applicando la molitplicazione per la **Matrice di Rotazione 3D sull'asse X (Pitch)** prima di proiettare in 2D tramite $f_x$ e $f_y$.
