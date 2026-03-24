import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import json
import socket

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (720, 480)}))
picam2.start()

# Initialize variables
cv_scaler = 4  # Scaling factor for frame processing
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Setup socket for communication with Node Helper
def send_message_to_node_helper(message):
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    node_helper_ip = "127.0.0.1"  # Adjust as needed
    node_helper_port = 5000  # Ensure this matches the Node.js listener port
    client.sendto(message.encode('utf-8'), (node_helper_ip, node_helper_port))

def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances else -1
        if best_match_index != -1 and matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    return frame, matches, best_match_index, face_distances

def draw_results(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    frame = picam2.capture_array()
    processed_frame, matches, best_match_index, face_distances = process_frame(frame)
    display_frame = draw_results(processed_frame, face_locations, face_names)
    
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Video', display_frame)
    
    if best_match_index != -1 and matches[best_match_index]:
        name = known_face_names[best_match_index]
        confidence = 1 - face_distances[best_match_index]
        message = {'login': {'user': best_match_index + 1, 'confidence': confidence}}
        send_message_to_node_helper(json.dumps(message))
        print(f"User {name} logged in with confidence {confidence:.2f}")
    
    if len(face_encodings) == 0:
        message = {'logout': {'user': 0}}
        send_message_to_node_helper(json.dumps(message))
        print("No faces detected. Logging out.")
    
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
