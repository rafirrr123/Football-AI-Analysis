import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("pss.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam")
        break

    # Detect + Track
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    # Draw bounding boxes, class names and tracking IDs
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Football YOLO Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()