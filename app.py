import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables for rep counting and tracking state
rep_count = 0
rep_started = False
is_tracking = False
cap = None
current_stage = ""  # Variable to track the current stage of the lift

# Function to calculate the vertical position of the wrists
def get_wrist_positions(landmarks):
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    return left_wrist, right_wrist

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Knee
    b = np.array(b)  # Hip
    c = np.array(c)  # Ankle
    ab = b - a
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure value is within range
    return np.degrees(angle)

# Function to start tracking
def start_tracking():
    global rep_count, rep_started, is_tracking, cap, current_stage
    is_tracking = True
    cap = cv2.VideoCapture(0)

    while is_tracking and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Initialize form_quality with a default value
        form_quality = "Not determined"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant landmarks
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            left_wrist, right_wrist = get_wrist_positions(landmarks)

            # Average the wrist positions for simplicity
            average_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
            knee_y = left_knee.y  # Get the knee height

            # Invert the y-coordinates based on the frame height
            frame_height = frame.shape[0]
            average_wrist_y = frame_height * average_wrist_y
            knee_y = frame_height * knee_y

            # Calculate angle for form quality assessment
            knee_coords = [frame_height * left_knee.x, frame_height * left_knee.y]
            hip_coords = [frame_height * left_hip.x, frame_height * left_hip.y]
            ankle_coords = [frame_height * left_ankle.x, frame_height * left_ankle.y]
            angle = calculate_angle(knee_coords, hip_coords, ankle_coords)

            # Debugging output for angle and stage
            print(f"Angle: {angle:.2f}, Stage: {current_stage}, Form: {form_quality}")

            # Detect down movement with increased tolerance for wrist height comparison
            if average_wrist_y > knee_y + 10:
                if not rep_started:
                    rep_started = True
                    current_stage = "down"  # Update current stage to "down"

                # Update form quality based on angle during the "down" stage
                if angle < 150:  # Threshold for poor form
                    form_quality = "Poor Form"
                elif 150 <= angle <= 180:  # Threshold for good form
                    form_quality = "Good Form"

            # Detect up movement
            elif average_wrist_y < knee_y - 10 and rep_started:  # Increased tolerance
                rep_count += 1
                rep_started = False
                current_stage = "up"  # Update current stage to "up"

            # Display rep count, stage, angle, and quality on the frame
            cv2.putText(frame, f"Reps: {rep_count}", 
                        (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Stage: {current_stage}", 
                        (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Angle: {angle:.2f}", 
                        (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Form: {form_quality}", 
                        (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Debugging information: display wrist and knee heights
            cv2.putText(frame, f"Wrist Y: {average_wrist_y:.2f}", 
                        (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Knee Y: {knee_y:.2f}", 
                        (50, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw pose landmarks with enhanced visualization
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp.solutions.drawing_utils.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5), 
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

        # Show the frame
        cv2.imshow("Deadlift Tracker", frame)

        # Listen for key inputs
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Press 'r' to reset the rep count
            rep_count = 0
            current_stage = ""  # Reset current stage

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    start_tracking()
