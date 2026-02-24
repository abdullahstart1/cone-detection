import cv2
import torch
from inference import detect_cones_in_frame

def run_realtime(model, device):

    cap = cv2.VideoCapture(0)

    model.eval()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, scores, _ = detect_cones_in_frame(
            model,
            frame,
            device
        )

        for box, cls, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)

            name = "yellow" if cls == 1 else "blue"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame,
                        f"{name}:{score:.2f}",
                        (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)

        cv2.imshow("Real-Time Cone Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()