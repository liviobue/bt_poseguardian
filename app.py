import cv2
import mediapipe as mp

# Initialize MediaPipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose detection
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame color space to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose landmarks
    result = pose.process(rgb_frame)

    # If pose landmarks are found
    if result.pose_landmarks:
        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract and print keypoints
        for i, landmark in enumerate(result.pose_landmarks.landmark):
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # Print or use landmark data as needed
            print(f"Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

    # Display the frame
    cv2.imshow("Pose Estimation", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
