import cv2
import mediapipe as mp
from posture_detector import PostureDetector
from pose_analyzer import PoseAnalyzer
import config

def main():
    """Main function to run the yoga posture tracker"""
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize detectors
    posture_detector = PostureDetector()
    pose_analyzer = PoseAnalyzer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    frame_count = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            height, width, _ = frame.shape
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            results = pose.process(rgb_frame)
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Analyze posture
                landmarks = results.pose_landmarks.landmark
                posture_feedback = pose_analyzer.analyze_pose(landmarks)
                
                # Display feedback
                y_offset = 30
                for i, feedback in enumerate(posture_feedback):
                    color = (0, 255, 0) if feedback['status'] == 'good' else (0, 165, 255)
                    cv2.putText(frame, feedback['message'], (10, y_offset + i * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Display pose confidence
                cv2.putText(frame, f"Pose Detected: {results.pose_landmarks is not None}",
                          (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected. Please ensure full body is visible.",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Yoga Posture Tracker', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()