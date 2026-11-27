# SquatLab — Análisis de sentadilla

Proyecto de visión por computadora que toma un video lateral de una sentadilla y estima, sin marcadores corporales, ángulos, fases y repeticiones en 2D.

> Los resultados son estimaciones para apoyar la observación. **No son un diagnóstico.**

## Qué hace

Flujo: video lateral → YOLOv8n-Pose (17 puntos del cuerpo) → ByteTrack (identifica a cada persona) → cálculo 2D → video anotado, CSV, JSON y gráficas.

- Seguimiento de 1 o 2 personas.
- Flexión de rodilla y cadera, inclinación de tronco y tibia.
- Fase de bajada (excéntrica) y subida (concéntrica), duración y repeticiones.
- Cuentas e historial guardados localmente en el computador (sin nube).

## Criterio de alerta

Se evalúa cuando la flexión de rodilla es de 90° o más. Δ = inclinación del tronco − inclinación de la tibia. Objetivo predeterminado: neutral.

| Δ | Resultado |
|---|---|
| Mayor que +10° | Dominancia de cadera |
| Entre −10° y +10° | Sin alerta |
| Menor que −10° | Dominancia de rodilla |

El umbral de ±10° es un criterio heurístico del prototipo, no una norma clínica.

## Archivos

| Archivo | Para qué sirve | Estado |
|---|---|---|
| `web_app.py` | Aplicación web para PC y Android: cargar o grabar videos y analizarlos | Actual |
| `squat_analysis.py` | Ángulos, fases y repeticiones | Actual |
| `app_seguimiento.py` | Aplicación de escritorio antigua: muestra tablas y gráficas de reportes de sentadilla guardados en `outputs/` | Antigua |
| `kinematics.py` | Cinemática horizontal de la marcha (posición, distancia, velocidad), del enfoque inicial. No participa en el análisis actual | Referencia |
| `docs/presentacion.pdf` | Presentación del proyecto | — |

## Archivos pendientes

`web_app.py` necesita estos elementos, que **todavía no están en el repositorio**:

- `pose_tracker.py` (detecta la pose y genera los resultados)
- carpeta `web/` (`index.html`, `manifest.webmanifest`, `sw.js`, `assets/`)

Sin ellos la aplicación web no funciona. `requirements.txt` tampoco incluye aún las librerías de `pose_tracker.py`.

## Instalación y uso (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python web_app.py --host 127.0.0.1
```

Abre `http://127.0.0.1:8000` en el navegador.

Para entrar desde un teléfono en la misma red Wi-Fi, ejecuta `python web_app.py` (sin `--host`). La cámara y la instalación en Android necesitan HTTPS, con las opciones `--cert-file` y `--key-file`.

**Seguridad:** sin HTTPS, las contraseñas viajan sin cifrar por la red. Usa el acceso desde el teléfono solo en una red de confianza y nunca subas certificados ni claves al repositorio.

Los datos se guardan en `%LOCALAPPDATA%\SquatLab`.

## Límites

- Una sola cámara lateral y fija: no evalúa valgo frontal, profundidad real ni simetría completa.
- La ropa, las oclusiones, la iluminación, el encuadre o una cámara inclinada pueden cambiar los resultados.
- No calcula fuerzas ni torques.

## Próximos pasos

Validar ángulos y fases contra una medición de referencia, y evaluar una cámara GoPro y sensores de profundidad (Kinect / RGB-D). Todavía no están integrados.

## Autores

- Carlos Andrés Puentes Chica
- Juan José Mayorga Vergano
- Andrés Felipe Tinjacá Segura
- José David Vargas Vargas
- Juan Manuel Olaya Manrique
- Brayan Esteban Santamaría Pulgarim

## Fuentes

- Straub RK, Powers CM. *A Biomechanical Review of the Squat Exercise.* Int J Sports Phys Ther. 2024;19(4):490–501. https://doi.org/10.26603/001c.94600
- Ultralytics YOLOv8: https://docs.ultralytics.com/models/yolov8
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
