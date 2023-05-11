import face_recognition
import os
import os
import sys
import cv2
import numpy as np
import math
import time
from deepface import DeepFace


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Helper


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            print(os.path.splitext(image)[0])
        print(self.known_face_names)

    def run_recognition(self):
        return_values = {
            "isMatching": False,
            "age": "0",
            "emotion": "",
            "image": np.ndarray,
            "userName" : ""
        }

        isMatching = False
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        t_end = time.time() + 30

        while time.time() < t_end:
            print(time.time())
            print(t_end)
            print(return_values)
            ret, frame = video_capture.read()
            ret, imgFrame = video_capture.read()
            result = DeepFace.analyze(img_path=frame, actions=[
                                      'emotion'], enforce_detection=False)
            resultAge = DeepFace.analyze(img_path=frame, actions=[
                                         'age'], enforce_detection=False)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            imageFrame = cv2.cvtColor(imgFrame,cv2.COLOR_BGR2RGB)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            emotion = result[0]["dominant_emotion"][:]
            age = resultAge[0]["age"]

            txtEmotion = str(emotion)
            txtAge = str(age)

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '0%'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]  
                        print(os.path.splitext(name)[0])
                        confidence = face_confidence(
                            face_distances[best_match_index])

                    confidence_without_percent = confidence[:-1]
                    print(confidence_without_percent)

                    if isMatching == False:
                        if float(confidence_without_percent) > 90:
                            print("Tanimlama Başarili")
                            return_values['age'] = txtAge
                            return_values['emotion'] = txtEmotion
                            return_values['isMatching'] = True
                            return_values['image'] = imageFrame
                            return_values['userName']=os.path.splitext(name)[0]
                            isMatching = True
                        else:
                            print("Tanimlama Başarisiz!!!")
                            # isMatching = False

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, txtEmotion, (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, txtAge, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        return return_values


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
