import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input


def predict_frame(model, frame, top_k=3):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (224, 224))
    image_array = np.expand_dims(resized.astype(np.float32), axis=0)
    image_array = preprocess_input(image_array)
    predictions = model.predict(image_array, verbose=0)
    return decode_predictions(predictions, top=top_k)[0]


def main():
    model = MobileNetV2(weights="imagenet")
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("Camera not available. Check permissions or camera index.")

    last_predictions = []
    frame_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            last_predictions = predict_frame(model, frame)

        y = 30
        overlay = frame.copy()
        for _, label, probability in last_predictions:
            text = f"{label}: {probability * 100:.1f}%"
            cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        cv2.imshow("Live object test - press q to quit", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
