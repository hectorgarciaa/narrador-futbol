import cv2

input_video = "./partido.mp4"
output_video = "./partido_medio.mp4"

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el códec y crear el objeto de escritura
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Leer y guardar los primeros 10 frames
for i in range(200):
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Liberar recursos
cap.release()
out.release()
