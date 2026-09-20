from ultralytics import YOLO
import cv2
import time

model = YOLO("field.pt")

cap = cv2.VideoCapture(0)

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    img = frame.copy()

    # Draw keypoints only
    if result.keypoints is not None:

        points = result.keypoints.xy.cpu().numpy()

        if len(points) > 0:

            for x, y in points[0]:

                cv2.circle(
                    img,
                    (int(x), int(y)),
                    6,
                    (0, 255, 0),
                    -1
                )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Football Field Keypoints", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()