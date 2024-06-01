import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array
import train
import os

model_data_folder = 'model_data'
dataset_folder = 'dataset'

haar_cascade_file_name = 'haarcascade_frontalface_default.xml'
haar_cascade_file_path = f'{model_data_folder}/{haar_cascade_file_name}'

model_file_name = 'sentiment_analysis_model.keras'
model_file_path = f'{model_data_folder}/{model_file_name}'

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
image_shape = (48, 48)


def start_video_capture():
    video_capture = cv2.VideoCapture(0)

    while True:
        read_frame, frame = video_capture.read()

        if not read_frame:
            continue

        detect_faces(frame)

        cv2.imshow('Face Sentiment Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


def detect_faces(frame):
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(
        grayscale_image,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = extract_roi(grayscale_image, x, y, w, h)

        predicted_emotions = predict_emotion(roi)

        render_face_detection_frame(frame, predicted_emotions, x, y, w, h)

def render_face_detection_frame(frame, predicted_emotions, x, y, w, h):
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2)

    for i, (emotion, prob) in enumerate(predicted_emotions[:2]):
        text = f"{emotion}: {prob * 100:.2f}%"
        y_text = y + h + 30 + i * 30
        cv2.putText(
            frame,
            text,
            (x, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2)


def extract_roi(grayscale_image, x, y, w, h):
    roi_gray = grayscale_image[y: y + h, x: x + w]

    roi_gray = cv2.resize(roi_gray, image_shape)
    roi = roi_gray.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    return roi


def predict_emotion(roi):
    predictions = emotion_classifier.predict(roi)[0]
    emotion_probabilities = {emotion_labels[i]: predictions[i] for i in range(len(predictions))}
    sorted_emotions = sorted(emotion_probabilities.items(), key=lambda e: e[1], reverse=True)

    return sorted_emotions


def build_model():
    print("Starting Face Emotion Recognition model build...")

    fer_model_builder = train.FERModelBuilder(
        dataset_path=f"{dataset_folder}/fer2013.csv",
        training_label="Training",
        test_label="PublicTest",
        input_shape=(*image_shape, 1),
        output_model_path=model_file_path)

    fer_model_builder.start_build()

    print("Model built successfully.")


if __name__ == '__main__':
    model_exists: bool = os.path.exists(model_file_path)

    if not model_exists:
        build_model()
    else:
        print("Model already exists.")

    haar_cascade = cv2.CascadeClassifier(haar_cascade_file_path)
    emotion_classifier = load_model(model_file_path)

    print("Starting video capture...\nPress 'q' to exit.")

    start_video_capture()