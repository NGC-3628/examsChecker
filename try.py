#""" 
#import cv2
#
#cap = cv2.VideoCapture(0)
#
#
#while True:
#    ret, frame = cap.read()
#    cv2.imshow('Camera', frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()
#
#"""




import cv2
import numpy as np

# Tamaño normalizado — equivale a carta a ~150dpi
# Siempre el mismo, sin importar la cámara
OUTPUT_W, OUTPUT_H = 1240, 1754

def find_markers(image):
    """Detecta los 4 cuadros negros de las esquinas."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Umbral: píxeles muy oscuros = marcadores
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 8000:               # tamaño esperado de marcador
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h
            if 0.7 < aspect < 1.3:          # forma cuadrada
                markers.append((x + w//2, y + h//2))  # centro del marcador

    return markers  # debe retornar exactamente 4 puntos

def order_corners(pts):
    """Ordena: top-left, top-right, bottom-right, bottom-left"""
    pts = np.array(pts)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.float32([
        pts[np.argmin(s)],   # TL: suma mínima
        pts[np.argmin(diff)],# TR: diferencia mínima
        pts[np.argmax(s)],   # BR: suma máxima
        pts[np.argmax(diff)] # BL: diferencia máxima
    ])

def normalize_sheet(frame):
    """Toma el frame crudo y retorna imagen normalizada 1240x1754."""
    markers = find_markers(frame)
    if len(markers) != 4:
        return None  # hoja no detectada

    src = order_corners(markers)
    dst = np.float32([
        [0, 0],
        [OUTPUT_W, 0],
        [OUTPUT_W, OUTPUT_H],
        [0, OUTPUT_H]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (OUTPUT_W, OUTPUT_H))
    return warped  # siempre 1240x1754, sin importar la cámara    




# TEMPORAL: prueba de cámara y detección de hoja

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        exit()
    
    print(f"Camara abierta. Resolución: {int(cap.get(3))}x{int(cap.get(4))}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: No se pudo leer frame")
            break
        
        # Intenta normalizar
        warped = normalize_sheet(frame)
        
        if warped is not None:
            cv2.putText(frame, "HOJA DETECTADA", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Normalizada", warped)
        else:
            cv2.putText(frame, f"Buscando hoja...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()