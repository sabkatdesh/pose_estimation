import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Constants
REP_GOAL = 12
UP_ANGLE_THRESHOLD = 70
DOWN_ANGLE_THRESHOLD = 120

# Globals for tracking
l_reps = 0
l_sets = 0
r_reps = 0
r_sets = 0
l_stage = None
r_stage = None


def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def update_stage(angle, stage, reps, sets, low_thresh=UP_ANGLE_THRESHOLD, high_thresh=DOWN_ANGLE_THRESHOLD):
    """Update stage and count reps/sets based on angle."""
    if angle > high_thresh:
        stage = "down"
    elif angle < low_thresh and stage == "down":
        stage = "up"
        reps += 1
        if reps == REP_GOAL:
            sets += 1
            reps = 0
    return stage, reps, sets


def main():
    global l_stage, l_reps, l_sets, r_stage, r_reps, r_sets

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_tracking_confidence=0.6, min_detection_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)
                )

                landmarks = results.pose_landmarks.landmark

                # Webcam is mirrored: RIGHT = viewer's left
                l_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angles
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                # Display angles
                l_elbow_coords = np.multiply(l_elbow, [image.shape[1], image.shape[0]]).astype(int)
                r_elbow_coords = np.multiply(r_elbow, [image.shape[1], image.shape[0]]).astype(int)

                cv2.putText(image, str(int(l_angle)), tuple(l_elbow_coords),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(r_angle)), tuple(r_elbow_coords),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Update stages and counters
                l_stage, l_reps, l_sets = update_stage(l_angle, l_stage, l_reps, l_sets)
                r_stage, r_reps, r_sets = update_stage(r_angle, r_stage, r_reps, r_sets)

            # Background info panel
            cv2.rectangle(image, (10, 10), (390, 120), (30, 30, 30), -1)

            # Display counters
            cv2.putText(image, f"LEFT REPS: {l_reps}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2, cv2.LINE_AA)
            cv2.putText(image, f"LEFT SETS: {l_sets}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2, cv2.LINE_AA)
            cv2.putText(image, f"L-Stage: {l_stage}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA)

            cv2.putText(image, f"RIGHT REPS: {r_reps}", (200, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(image, f"RIGHT SETS: {r_sets}", (200, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"R-Stage: {r_stage}", (200, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)

            # Show video feed
            cv2.imshow("My Video Feed", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
