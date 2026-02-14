import cv2
import face_recognition
import pickle
import numpy as np

# Load the trained model
print("[INFO] Loading model...")
with open("face_model.pkl", "rb") as f:
    model_data = pickle.load(f)

method = model_data.get('method', 'svm')
print(f"[INFO] Model type: {method}")

if method == 'svm':
    clf = model_data['classifier']
    label_names = model_data['label_names']
    print(f"[INFO] SVM model loaded with classes: {label_names}")
else:
    known_encodings = model_data['encodings']
    known_names = model_data['names']
    print(f"[INFO] Encoding model loaded with {len(known_encodings)} encodings")
    print(f"[INFO] People in model: {list(set(known_names))}")

# Configurable thresholds
CONFIDENCE_THRESHOLD = 60  # Lowered from 70
DISTANCE_THRESHOLD = 0.55  # Slightly adjusted

print(f"[INFO] Using confidence threshold: {CONFIDENCE_THRESHOLD}%")

# Start webcam
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam")
    exit()

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            if method == 'svm':
                # Use SVM classifier with improved confidence handling
                try:
                    probability = clf.predict_proba([face_encoding])[0]
                    confidence = max(probability) * 100
                    prediction = clf.predict([face_encoding])[0]
                    predicted_idx = clf.classes_.tolist().index(prediction)
                    
                    # Get the actual probability for the predicted class
                    actual_confidence = probability[predicted_idx] * 100
                    
                    # Check if confidence meets threshold
                    if actual_confidence < CONFIDENCE_THRESHOLD:
                        name = f"Unknown ({actual_confidence:.1f}%)"
                        color = (0, 0, 255)  # Red for unknown
                    else:
                        name = f"{prediction} ({actual_confidence:.1f}%)"
                        color = (0, 255, 0)  # Green for known
                        
                except Exception as e:
                    name = "Unknown (Error)"
                    color = (0, 0, 255)
                    print(f"[WARNING] Prediction error: {e}")
                
            else:
                # Use distance matching for single person
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = min(distances)
                
                if min_distance < DISTANCE_THRESHOLD:
                    best_match_idx = np.argmin(distances)
                    name = known_names[best_match_idx]
                    confidence = (1 - min_distance) * 100
                    name = f"{name} ({confidence:.1f}%)"
                    color = (0, 255, 0)  # Green
                else:
                    name = f"Unknown ({min_distance:.3f})"
                    color = (0, 0, 255)  # Red
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Put name text
            cv2.putText(frame, name, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    # Display info
    cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD}%", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Face Recognition Test", frame)
    
    # Key controls for adjusting threshold in real-time
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        CONFIDENCE_THRESHOLD = min(95, CONFIDENCE_THRESHOLD + 5)
        print(f"[INFO] Confidence threshold increased to: {CONFIDENCE_THRESHOLD}%")
    elif key == ord('-'):
        CONFIDENCE_THRESHOLD = max(40, CONFIDENCE_THRESHOLD - 5)
        print(f"[INFO] Confidence threshold decreased to: {CONFIDENCE_THRESHOLD}%")

cap.release()
cv2.imshow("Face Recognition Test", frame)
    
    # Key controls for adjusting threshold in real-time
key = cv2.waitKey(1) & 0xFF

cap.release()
cv2.destroyAllWindows()
print("[INFO] Test complete!")