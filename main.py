import cv2
import face_recognition
import os

# Load known images
known_faces = []
known_names = []

dataset_path = "dataset"

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = face_recognition.load_image_file(image_path)

        face_encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(face_encoding)
        known_names.append(person_name)

print("Face data loaded successfully")

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):

        matches = face_recognition.compare_faces(
            known_faces,
            face_encoding
        )

        name = "Unknown"

        if True in matches:

            match_index = matches.index(True)

            name = known_names[match_index]

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()