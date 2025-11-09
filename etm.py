import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Initialize webcam
cam = cv2.VideoCapture(0)
# Initialize face mesh detector with high confidence threshold
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Get screen dimensions
screen_w, screen_h = pyautogui.size()
# Set mouse movement sensitivity
sensitivity = 5
# Smoothing factor (higher = smoother but more lag)
smoothing = 0.5
# Initial position
prev_x, prev_y = 0, 0

# Blink detection parameters
EYE_AR_THRESHOLD = 0.18
EYE_AR_CONSEC_FRAMES = 3
# Both eyes closed counter
BOTH_EYES_COUNTER = 0
# Left eye closed counter
LEFT_EYE_COUNTER = 0
# Right eye closed counter
RIGHT_EYE_COUNTER = 0
# Time trackers
LAST_CLICK_TIME = 0
CLICK_COOLDOWN = 1.0  # seconds between click events

def calculate_ear(eye_points, landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for the given eye points
    """
    # Get the vertical distances between eye landmarks
    v1 = np.linalg.norm(
        np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]) -
        np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
    )
    v2 = np.linalg.norm(
        np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]) -
        np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    )
    # Get the horizontal distance
    h = np.linalg.norm(
        np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]) -
        np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])
    )
    # Calculate EAR
    return (v1 + v2) / (2.0 * h)

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cam.read()
    if not ret:
        break
        
    # Flip frame horizontally for a more intuitive experience
    frame = cv2.flip(frame, 1)
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame to detect face landmarks
    output = face_mesh.process(rgb_frame)
    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape
    
    # Add UI elements
    cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
    cv2.putText(frame, "Eye Tracking Mouse (Press 'q' to quit)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Check if face landmarks were detected
    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        
        # Right eye iris landmarks for mouse control
        right_iris = []
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            right_iris.append([x, y])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Calculate the center of right iris
        right_iris = np.array(right_iris)
        right_iris_center = right_iris.mean(axis=0).astype(int)
        cv2.circle(frame, tuple(right_iris_center), 4, (0, 255, 0), -1)
        
        # Map eye position to screen coordinates with sensitivity adjustment
        x_ratio = landmarks[476].x
        y_ratio = landmarks[476].y
        screen_x = screen_w * (x_ratio * sensitivity - (sensitivity - 1) / 2)
        screen_y = screen_h * (y_ratio * sensitivity - (sensitivity - 1) / 2)
        
        # Apply smoothing
        curr_x = prev_x * smoothing + screen_x * (1 - smoothing)
        curr_y = prev_y * smoothing + screen_y * (1 - smoothing)
        prev_x, prev_y = curr_x, curr_y
        
        # Ensure coordinates are within screen bounds
        curr_x = max(0, min(curr_x, screen_w))
        curr_y = max(0, min(curr_y, screen_h))
        
        # Move mouse to the calculated position
        pyautogui.moveTo(int(curr_x), int(curr_y))
        
        # Define eye landmark indices
        LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks)
        right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks)
        
        # Draw eye landmarks
        for idx in LEFT_EYE_INDICES:
            x, y = int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        for idx in RIGHT_EYE_INDICES:
            x, y = int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # Display EAR values
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, frame_h - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, frame_h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_time = time.time()
        # Check if enough time has passed since last click
        if current_time - LAST_CLICK_TIME >= CLICK_COOLDOWN:
            # Both eyes closed (double click)
            if left_ear < EYE_AR_THRESHOLD and right_ear < EYE_AR_THRESHOLD:
                BOTH_EYES_COUNTER += 1
                LEFT_EYE_COUNTER = 0
                RIGHT_EYE_COUNTER = 0
                
                if BOTH_EYES_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # Double click
                    pyautogui.doubleClick()
                    cv2.putText(frame, "Double Click", (frame_w - 180, frame_h - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    LAST_CLICK_TIME = current_time
                    BOTH_EYES_COUNTER = 0
                
            # Left eye closed (left click)
            elif left_ear < EYE_AR_THRESHOLD and right_ear >= EYE_AR_THRESHOLD:
                LEFT_EYE_COUNTER += 1
                RIGHT_EYE_COUNTER = 0
                BOTH_EYES_COUNTER = 0
                
                if LEFT_EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # Left click
                    pyautogui.click(button='left')
                    cv2.putText(frame, "Left Click", (frame_w - 180, frame_h - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    LAST_CLICK_TIME = current_time
                    LEFT_EYE_COUNTER = 0
                
            # Right eye closed (right click)
            elif right_ear < EYE_AR_THRESHOLD and left_ear >= EYE_AR_THRESHOLD:
                RIGHT_EYE_COUNTER += 1
                LEFT_EYE_COUNTER = 0
                BOTH_EYES_COUNTER = 0
                
                if RIGHT_EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # Right click
                    pyautogui.click(button='right')
                    cv2.putText(frame, "Right Click", (frame_w - 180, frame_h - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    LAST_CLICK_TIME = current_time
                    RIGHT_EYE_COUNTER = 0
            
            # Both eyes open (reset counters)
            else:
                LEFT_EYE_COUNTER = 0
                RIGHT_EYE_COUNTER = 0
                BOTH_EYES_COUNTER = 0
    
    # Display the frame with landmarks
    cv2.imshow('Eye Tracking Mouse', frame)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows() 
