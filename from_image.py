import cv2
import os

MODEL_PATH = 'trained_model.xml'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
UJICOBA_PATH = 'ujicoba/'
OUTPUT_DIKENALI = 'hasil/dikenali/'
OUTPUT_TIDAKDIKENALI = 'hasil/tidakdikenali/'

os.makedirs(OUTPUT_DIKENALI, exist_ok=True)
os.makedirs(OUTPUT_TIDAKDIKENALI, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

for filename in os.listdir(UJICOBA_PATH):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    path = os.path.join(UJICOBA_PATH, filename)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))
        label, conf = recognizer.predict(face_resized)

        if label == 1:  # penghuni
            label_text = "penghuni"
            save_path = os.path.join(OUTPUT_DIKENALI, filename)
        else:
            label_text = "bukan"
            save_path = os.path.join(OUTPUT_TIDAKDIKENALI, filename)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(save_path, img)
        print(f"{filename} → {label_text} (confidence: {conf:.2f})")

cv2.destroyAllWindows()
