import cv2
import os
import numpy as np

# Path data
DATA_PATHS = {
    'penghuni': 1,
    'bukanpenghuni': 0
}

CASCADE_PATH = 'haarcascade_frontalface_default.xml'
OUTPUT_MODEL = 'trained_model.xml'

def load_images_from_folder(folder_path, label):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            detected = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in detected:
                face = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (200, 200))
                faces.append(face_resized)
                labels.append(label)

    return faces, labels

def prepare_dataset():
    all_faces = []
    all_labels = []
    for folder_name, label in DATA_PATHS.items():
        folder_path = os.path.join('.', folder_name)
        faces, labels = load_images_from_folder(folder_path, label)
        all_faces.extend(faces)
        all_labels.extend(labels)
    return all_faces, all_labels

def train_and_save_model():
    print("[INFO] Loading dataset...")
    faces, labels = prepare_dataset()
    print(f"[INFO] Total data wajah: {len(faces)}")

    print("[INFO] Training LBPH model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save(OUTPUT_MODEL)
    print(f"[INFO] Model disimpan sebagai: {OUTPUT_MODEL}")

if __name__ == "__main__":
    train_and_save_model()
