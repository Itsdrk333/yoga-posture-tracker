"""
Yoga Posture Detector
Detects and analyzes yoga poses using computer vision and pose estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class YogaPose(Enum):
    """Enumeration of common yoga poses."""
    MOUNTAIN_POSE = "Mountain Pose"
    DOWNWARD_DOG = "Downward Dog"
    WARRIOR_I = "Warrior I"
    WARRIOR_II = "Warrior II"
    TREE_POSE = "Tree Pose"
    COBRA_POSE = "Cobra Pose"
    CHILD_POSE = "Child Pose"
    LOTUS_POSE = "Lotus Pose"
    PLANK_POSE = "Plank Pose"
    UNKNOWN = "Unknown Pose"


@dataclass
class PoseConfidence:
    """Data class for pose detection confidence."""
    pose: YogaPose
    confidence: float
    landmarks: Dict[str, np.ndarray]
    angles: Dict[str, float]


class PoseDetector:
    """
    Detects and classifies yoga poses using MediaPipe pose estimation.
    """

    def __init__(self, static_image_mode: bool = False, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        Initialize the pose detector.
        
        Args:
            static_image_mode: If true, uses static image mode
            min_detection_confidence: Minimum confidence threshold for pose detection
            min_tracking_confidence: Minimum confidence threshold for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_names = [landmark.name for landmark in self.mp_pose.PoseLandmark]
        
    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect pose landmarks in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary containing detected landmarks and pose info, or None if no pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = self._extract_landmarks(results.pose_landmarks, image.shape)
            return {
                'landmarks': landmarks,
                'pose_world_landmarks': results.pose_world_landmarks,
                'visibility': self._extract_visibility(results.pose_landmarks)
            }
        
        return None
    
    def _extract_landmarks(self, pose_landmarks, image_shape: Tuple) -> Dict[str, np.ndarray]:
        """Extract landmark coordinates from pose landmarks."""
        landmarks = {}
        h, w, c = image_shape
        
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks[self.landmark_names[idx]] = np.array([
                landmark.x * w,
                landmark.y * h,
                landmark.z
            ])
        
        return landmarks
    
    def _extract_visibility(self, pose_landmarks) -> Dict[str, float]:
        """Extract visibility scores for each landmark."""
        visibility = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            visibility[self.landmark_names[idx]] = landmark.visibility
        
        return visibility
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, 
                       point3: np.ndarray) -> float:
        """
        Calculate angle between three points.
        
        Args:
            point1: First point coordinates
            point2: Middle point coordinates
            point3: Third point coordinates
            
        Returns:
            Angle in degrees
        """
        # Vector from point2 to point1
        vector1 = point1[:2] - point2[:2]
        # Vector from point2 to point3
        vector2 = point3[:2] - point2[:2]
        
        # Calculate angle
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point coordinates
            point2: Second point coordinates
            
        Returns:
            Distance between the two points
        """
        return np.linalg.norm(point1[:2] - point2[:2])
    
    def classify_pose(self, landmarks: Dict[str, np.ndarray]) -> PoseConfidence:
        """
        Classify the detected pose.
        
        Args:
            landmarks: Dictionary of landmark positions
            
        Returns:
            PoseConfidence object with detected pose and confidence
        """
        angles = self._calculate_pose_angles(landmarks)
        
        # Detect pose based on key angles and positions
        if self._is_mountain_pose(landmarks, angles):
            return PoseConfidence(YogaPose.MOUNTAIN_POSE, 0.9, landmarks, angles)
        elif self._is_downward_dog(landmarks, angles):
            return PoseConfidence(YogaPose.DOWNWARD_DOG, 0.85, landmarks, angles)
        elif self._is_warrior_i(landmarks, angles):
            return PoseConfidence(YogaPose.WARRIOR_I, 0.8, landmarks, angles)
        elif self._is_warrior_ii(landmarks, angles):
            return PoseConfidence(YogaPose.WARRIOR_II, 0.8, landmarks, angles)
        elif self._is_tree_pose(landmarks, angles):
            return PoseConfidence(YogaPose.TREE_POSE, 0.75, landmarks, angles)
        elif self._is_plank_pose(landmarks, angles):
            return PoseConfidence(YogaPose.PLANK_POSE, 0.85, landmarks, angles)
        elif self._is_cobra_pose(landmarks, angles):
            return PoseConfidence(YogaPose.COBRA_POSE, 0.8, landmarks, angles)
        else:
            return PoseConfidence(YogaPose.UNKNOWN, 0.5, landmarks, angles)
    
    def _calculate_pose_angles(self, landmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate key angles for pose classification."""
        angles = {}
        
        # Shoulder angles
        angles['left_shoulder'] = self.calculate_angle(
            landmarks.get('LEFT_ELBOW', np.array([0, 0])),
            landmarks.get('LEFT_SHOULDER', np.array([0, 0])),
            landmarks.get('LEFT_HIP', np.array([0, 0]))
        )
        angles['right_shoulder'] = self.calculate_angle(
            landmarks.get('RIGHT_ELBOW', np.array([0, 0])),
            landmarks.get('RIGHT_SHOULDER', np.array([0, 0])),
            landmarks.get('RIGHT_HIP', np.array([0, 0]))
        )
        
        # Elbow angles
        angles['left_elbow'] = self.calculate_angle(
            landmarks.get('LEFT_SHOULDER', np.array([0, 0])),
            landmarks.get('LEFT_ELBOW', np.array([0, 0])),
            landmarks.get('LEFT_WRIST', np.array([0, 0]))
        )
        angles['right_elbow'] = self.calculate_angle(
            landmarks.get('RIGHT_SHOULDER', np.array([0, 0])),
            landmarks.get('RIGHT_ELBOW', np.array([0, 0])),
            landmarks.get('RIGHT_WRIST', np.array([0, 0]))
        )
        
        # Hip angles
        angles['left_hip'] = self.calculate_angle(
            landmarks.get('LEFT_SHOULDER', np.array([0, 0])),
            landmarks.get('LEFT_HIP', np.array([0, 0])),
            landmarks.get('LEFT_KNEE', np.array([0, 0]))
        )
        angles['right_hip'] = self.calculate_angle(
            landmarks.get('RIGHT_SHOULDER', np.array([0, 0])),
            landmarks.get('RIGHT_HIP', np.array([0, 0])),
            landmarks.get('RIGHT_KNEE', np.array([0, 0]))
        )
        
        # Knee angles
        angles['left_knee'] = self.calculate_angle(
            landmarks.get('LEFT_HIP', np.array([0, 0])),
            landmarks.get('LEFT_KNEE', np.array([0, 0])),
            landmarks.get('LEFT_ANKLE', np.array([0, 0]))
        )
        angles['right_knee'] = self.calculate_angle(
            landmarks.get('RIGHT_HIP', np.array([0, 0])),
            landmarks.get('RIGHT_KNEE', np.array([0, 0])),
            landmarks.get('RIGHT_ANKLE', np.array([0, 0]))
        )
        
        return angles
    
    def _is_mountain_pose(self, landmarks: Dict[str, np.ndarray], 
                         angles: Dict[str, float]) -> bool:
        """Check if pose is mountain pose (Tadasana)."""
        # In mountain pose, body is upright, arms at sides
        try:
            shoulder_distance = self.calculate_distance(
                landmarks.get('LEFT_SHOULDER', np.array([0, 0])),
                landmarks.get('RIGHT_SHOULDER', np.array([0, 0]))
            )
            hip_distance = self.calculate_distance(
                landmarks.get('LEFT_HIP', np.array([0, 0])),
                landmarks.get('RIGHT_HIP', np.array([0, 0]))
            )
            
            # Check if knees are relatively straight
            knee_angle = (angles.get('left_knee', 0) + angles.get('right_knee', 0)) / 2
            
            return (knee_angle > 160 and 
                    abs(shoulder_distance - hip_distance) < 50)
        except:
            return False
    
    def _is_downward_dog(self, landmarks: Dict[str, np.ndarray], 
                        angles: Dict[str, float]) -> bool:
        """Check if pose is downward dog."""
        try:
            # Head should be between hands, hips high
            head = landmarks.get('NOSE', np.array([0, 0]))
            left_wrist = landmarks.get('LEFT_WRIST', np.array([0, 0]))
            right_wrist = landmarks.get('RIGHT_WRIST', np.array([0, 0]))
            left_hip = landmarks.get('LEFT_HIP', np.array([0, 0]))
            right_hip = landmarks.get('RIGHT_HIP', np.array([0, 0]))
            
            # Hip should be higher than head
            hip_height = (left_hip[1] + right_hip[1]) / 2
            head_height = head[1]
            
            # Hands should be relatively wide apart and below hips
            hand_distance = self.calculate_distance(left_wrist, right_wrist)
            
            return (hip_height < head_height and 
                    hand_distance > 100 and
                    angles.get('left_elbow', 0) < 120 and
                    angles.get('right_elbow', 0) < 120)
        except:
            return False
    
    def _is_warrior_i(self, landmarks: Dict[str, np.ndarray], 
                     angles: Dict[str, float]) -> bool:
        """Check if pose is Warrior I."""
        try:
            # Front leg bent, back leg straight
            left_knee = angles.get('left_knee', 0)
            right_knee = angles.get('right_knee', 0)
            
            one_knee_bent = (left_knee < 120 or right_knee < 120)
            one_knee_straight = (left_knee > 160 or right_knee > 160)
            
            return one_knee_bent and one_knee_straight
        except:
            return False
    
    def _is_warrior_ii(self, landmarks: Dict[str, np.ndarray], 
                      angles: Dict[str, float]) -> bool:
        """Check if pose is Warrior II."""
        try:
            # One leg bent, torso twisted
            left_knee = angles.get('left_knee', 0)
            right_knee = angles.get('right_knee', 0)
            
            one_knee_bent = (left_knee < 120 or right_knee < 120)
            arms_extended = (angles.get('left_shoulder', 0) > 160 or
                           angles.get('right_shoulder', 0) > 160)
            
            return one_knee_bent and arms_extended
        except:
            return False
    
    def _is_tree_pose(self, landmarks: Dict[str, np.ndarray], 
                     angles: Dict[str, float]) -> bool:
        """Check if pose is tree pose."""
        try:
            left_knee = angles.get('left_knee', 0)
            right_knee = angles.get('right_knee', 0)
            
            # One leg should be bent significantly
            one_leg_bent = (left_knee < 100 or right_knee < 100)
            one_leg_straight = (left_knee > 160 or right_knee > 160)
            
            return one_leg_bent and one_leg_straight
        except:
            return False
    
    def _is_plank_pose(self, landmarks: Dict[str, np.ndarray], 
                      angles: Dict[str, float]) -> bool:
        """Check if pose is plank pose."""
        try:
            # Body should be relatively horizontal
            shoulder = landmarks.get('LEFT_SHOULDER', np.array([0, 0]))
            wrist = landmarks.get('LEFT_WRIST', np.array([0, 0]))
            ankle = landmarks.get('LEFT_ANKLE', np.array([0, 0]))
            
            # Shoulder-wrist-ankle should be relatively aligned
            elbow_angle = angles.get('left_elbow', 0)
            
            return elbow_angle < 120
        except:
            return False
    
    def _is_cobra_pose(self, landmarks: Dict[str, np.ndarray], 
                      angles: Dict[str, float]) -> bool:
        """Check if pose is cobra pose."""
        try:
            # Chest should be lifted, arms relatively straight
            shoulder = landmarks.get('LEFT_SHOULDER', np.array([0, 0]))
            hip = landmarks.get('LEFT_HIP', np.array([0, 0]))
            
            # Shoulder should be higher than hip (backbend)
            shoulder_higher = shoulder[1] < hip[1]
            
            elbow_angle = (angles.get('left_elbow', 0) + angles.get('right_elbow', 0)) / 2
            
            return shoulder_higher and elbow_angle < 140
        except:
            return False
    
    def draw_landmarks(self, image: np.ndarray, 
                      landmarks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Draw detected landmarks on the image.
        
        Args:
            image: Input image
            landmarks: Dictionary of landmark positions
            
        Returns:
            Image with drawn landmarks
        """
        output = image.copy()
        h, w, c = image.shape
        
        # Draw circles at each landmark
        for landmark_name, coords in landmarks.items():
            x, y = int(coords[0]), int(coords[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections (skeleton)
        connections = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        ]
        
        for connection in connections:
            if connection[0] in landmarks and connection[1] in landmarks:
                point1 = landmarks[connection[0]]
                point2 = landmarks[connection[1]]
                x1, y1 = int(point1[0]), int(point1[1])
                x2, y2 = int(point2[0]), int(point2[1])
                
                if (0 <= x1 < w and 0 <= y1 < h and 
                    0 <= x2 < w and 0 <= y2 < h):
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return output
    
    def close(self):
        """Clean up resources."""
        self.pose.close()


def process_video(video_path: str, output_path: Optional[str] = None) -> List[PoseConfidence]:
    """
    Process a video file and detect yoga poses.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save output video with pose annotations
        
    Returns:
        List of detected poses for each frame
    """
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)
    
    poses = []
    frame_count = 0
    
    # Video writer setup
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        pose_data = detector.detect_pose(frame)
        
        if pose_data:
            pose_confidence = detector.classify_pose(pose_data['landmarks'])
            poses.append(pose_confidence)
            
            # Draw landmarks
            frame = detector.draw_landmarks(frame, pose_data['landmarks'])
            
            # Add pose text
            cv2.putText(frame, 
                       f"{pose_confidence.pose.value} ({pose_confidence.confidence:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if out:
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    if out:
        out.release()
    detector.close()
    
    return poses


if __name__ == "__main__":
    # Example usage
    detector = PoseDetector()
    
    # Process video or image
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            poses = process_video(input_path, output_path)
            print(f"Detected {len(poses)} frames")
        else:
            image = cv2.imread(input_path)
            pose_data = detector.detect_pose(image)
            if pose_data:
                pose = detector.classify_pose(pose_data['landmarks'])
                print(f"Detected pose: {pose.pose.value} (Confidence: {pose.confidence})")
                
                # Draw and save
                annotated = detector.draw_landmarks(image, pose_data['landmarks'])
                cv2.imwrite('output.jpg', annotated)
    
    detector.close()
