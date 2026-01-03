"""
Pose Analyzer Module for Yoga Posture Tracking

This module provides functionality to analyze yoga poses using computer vision
and pose estimation techniques. It detects key body joints and evaluates the
correctness of yoga postures against expected pose configurations.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PoseConfidence(Enum):
    """Confidence levels for pose detection."""
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class JointPoint:
    """Represents a detected joint point in the body."""
    x: float
    y: float
    confidence: float
    name: str

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y])


@dataclass
class PoseAnalysisResult:
    """Result of pose analysis."""
    pose_name: str
    accuracy: float  # 0-100%
    is_correct: bool
    feedback: List[str]
    joint_angles: Dict[str, float]
    detected_joints: List[JointPoint]
    timestamp: str


class PoseAnalyzer:
    """
    Main class for analyzing yoga poses and providing feedback.
    """

    # Standard yoga pose configurations
    YOGA_POSES = {
        'mountain_pose': {
            'name': 'Mountain Pose (Tadasana)',
            'key_angles': {
                'left_elbow': 180.0,
                'right_elbow': 180.0,
                'left_knee': 180.0,
                'right_knee': 180.0,
                'spine': 180.0,
            },
            'tolerance': 15.0
        },
        'downward_dog': {
            'name': 'Downward Facing Dog (Adho Mukha Svanasana)',
            'key_angles': {
                'left_elbow': 180.0,
                'right_elbow': 180.0,
                'left_hip': 90.0,
                'right_hip': 90.0,
                'spine': 90.0,
            },
            'tolerance': 20.0
        },
        'warrior_i': {
            'name': 'Warrior I (Virabhadrasana I)',
            'key_angles': {
                'left_hip': 90.0,
                'right_knee': 90.0,
                'spine': 0.0,
            },
            'tolerance': 25.0
        },
        'tree_pose': {
            'name': 'Tree Pose (Vrksasana)',
            'key_angles': {
                'right_hip': 45.0,
                'spine': 0.0,
                'left_knee': 180.0,
            },
            'tolerance': 20.0
        },
        'child_pose': {
            'name': 'Child Pose (Balasana)',
            'key_angles': {
                'left_knee': 90.0,
                'right_knee': 90.0,
                'spine': 45.0,
            },
            'tolerance': 15.0
        },
    }

    def __init__(self, pose_detection_model=None):
        """
        Initialize the Pose Analyzer.

        Args:
            pose_detection_model: Optional pre-trained pose detection model
        """
        self.model = pose_detection_model
        self.detected_joints: List[JointPoint] = []
        self.frame_count = 0

    def detect_joints(self, frame: np.ndarray) -> List[JointPoint]:
        """
        Detect body joints in the given frame.

        Args:
            frame: Input image frame (BGR format from OpenCV)

        Returns:
            List of detected JointPoint objects
        """
        self.frame_count += 1

        if self.model is None:
            # Placeholder for when actual model is not available
            self.detected_joints = self._simulate_joint_detection(frame)
        else:
            self.detected_joints = self._detect_with_model(frame)

        return self.detected_joints

    def _detect_with_model(self, frame: np.ndarray) -> List[JointPoint]:
        """
        Detect joints using the pose detection model.

        Args:
            frame: Input image frame

        Returns:
            List of detected JointPoint objects
        """
        # This would integrate with actual pose detection models like:
        # - MediaPipe Pose
        # - OpenPose
        # - PoseNet
        # Implementation depends on the specific model chosen

        joints = []
        # Model inference would happen here
        return joints

    def _simulate_joint_detection(self, frame: np.ndarray) -> List[JointPoint]:
        """
        Simulate joint detection for testing purposes.

        Args:
            frame: Input image frame

        Returns:
            List of simulated JointPoint objects
        """
        height, width = frame.shape[:2]
        joints = [
            JointPoint(width // 2, height // 4, 0.95, "nose"),
            JointPoint(width // 3, height // 3, 0.92, "left_shoulder"),
            JointPoint(2 * width // 3, height // 3, 0.90, "right_shoulder"),
            JointPoint(width // 4, height // 2, 0.88, "left_elbow"),
            JointPoint(3 * width // 4, height // 2, 0.87, "right_elbow"),
            JointPoint(width // 6, height // 2, 0.85, "left_wrist"),
            JointPoint(5 * width // 6, height // 2, 0.84, "right_wrist"),
            JointPoint(width // 3, 2 * height // 3, 0.90, "left_hip"),
            JointPoint(2 * width // 3, 2 * height // 3, 0.89, "right_hip"),
            JointPoint(width // 4, height - 100, 0.86, "left_knee"),
            JointPoint(3 * width // 4, height - 100, 0.85, "right_knee"),
            JointPoint(width // 6, height - 20, 0.80, "left_ankle"),
            JointPoint(5 * width // 6, height - 20, 0.79, "right_ankle"),
        ]
        return joints

    def calculate_angle(
        self,
        joint_a: JointPoint,
        joint_b: JointPoint,
        joint_c: JointPoint
    ) -> float:
        """
        Calculate the angle formed by three joints.

        Args:
            joint_a: First joint point
            joint_b: Center/vertex joint point
            joint_c: Third joint point

        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        a = joint_a.to_array()
        b = joint_b.to_array()
        c = joint_c.to_array()

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        # Clamp to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def analyze_pose(
        self,
        frame: np.ndarray,
        target_pose: str = 'mountain_pose'
    ) -> PoseAnalysisResult:
        """
        Analyze the pose in the given frame.

        Args:
            frame: Input image frame
            target_pose: Name of the target yoga pose

        Returns:
            PoseAnalysisResult with analysis details
        """
        # Detect joints
        self.detect_joints(frame)

        if not self.detected_joints:
            return PoseAnalysisResult(
                pose_name=target_pose,
                accuracy=0.0,
                is_correct=False,
                feedback=["No joints detected. Please ensure proper lighting and pose visibility."],
                joint_angles={},
                detected_joints=[],
                timestamp=str(self.frame_count)
            )

        # Get target pose configuration
        if target_pose not in self.YOGA_POSES:
            return PoseAnalysisResult(
                pose_name=target_pose,
                accuracy=0.0,
                is_correct=False,
                feedback=[f"Unknown pose: {target_pose}"],
                joint_angles={},
                detected_joints=self.detected_joints,
                timestamp=str(self.frame_count)
            )

        pose_config = self.YOGA_POSES[target_pose]

        # Calculate joint angles
        joint_angles = self._calculate_pose_angles()

        # Compare with target configuration
        accuracy, feedback = self._evaluate_pose(
            joint_angles,
            pose_config,
            target_pose
        )

        return PoseAnalysisResult(
            pose_name=pose_config['name'],
            accuracy=accuracy,
            is_correct=accuracy >= 70.0,
            feedback=feedback,
            joint_angles=joint_angles,
            detected_joints=self.detected_joints,
            timestamp=str(self.frame_count)
        )

    def _calculate_pose_angles(self) -> Dict[str, float]:
        """
        Calculate angles for key body joints.

        Returns:
            Dictionary of joint angles
        """
        angles = {}
        joint_dict = {joint.name: joint for joint in self.detected_joints}

        # Define angle calculations based on available joints
        angle_definitions = [
            ('left_elbow', 'left_shoulder', 'left_elbow', 'left_wrist'),
            ('right_elbow', 'right_shoulder', 'right_elbow', 'right_wrist'),
            ('left_knee', 'left_hip', 'left_knee', 'left_ankle'),
            ('right_knee', 'right_hip', 'right_knee', 'right_ankle'),
        ]

        for angle_name, j1_name, j2_name, j3_name in angle_definitions:
            if j1_name in joint_dict and j2_name in joint_dict and j3_name in joint_dict:
                angle = self.calculate_angle(
                    joint_dict[j1_name],
                    joint_dict[j2_name],
                    joint_dict[j3_name]
                )
                angles[angle_name] = angle

        return angles

    def _evaluate_pose(
        self,
        joint_angles: Dict[str, float],
        pose_config: Dict,
        pose_name: str
    ) -> Tuple[float, List[str]]:
        """
        Evaluate how well the current pose matches the target.

        Args:
            joint_angles: Calculated joint angles
            pose_config: Target pose configuration
            pose_name: Name of the pose

        Returns:
            Tuple of (accuracy percentage, feedback list)
        """
        feedback = []
        angle_errors = []

        for joint_name, target_angle in pose_config['key_angles'].items():
            if joint_name in joint_angles:
                actual_angle = joint_angles[joint_name]
                error = abs(actual_angle - target_angle)
                angle_errors.append(error)

                if error > pose_config['tolerance']:
                    feedback.append(
                        f"⚠️  {joint_name}: Expected ~{target_angle}°, got {actual_angle:.1f}° (error: {error:.1f}°)"
                    )
                else:
                    feedback.append(
                        f"✓ {joint_name}: {actual_angle:.1f}° (correct)"
                    )
            else:
                feedback.append(f"⚠️  Could not detect {joint_name}")

        # Calculate accuracy
        if angle_errors:
            max_error = pose_config['tolerance'] * 2
            accuracy = max(0, 100 - (np.mean(angle_errors) / max_error * 100))
        else:
            accuracy = 0.0

        if accuracy >= 80:
            feedback.insert(0, "🎉 Excellent pose alignment!")
        elif accuracy >= 60:
            feedback.insert(0, "👍 Good pose, but adjust the highlighted joints.")
        else:
            feedback.insert(0, "📍 Continue refining your posture alignment.")

        return accuracy, feedback

    def get_available_poses(self) -> Dict[str, str]:
        """
        Get list of available yoga poses.

        Returns:
            Dictionary mapping pose keys to pose names
        """
        return {key: config['name'] for key, config in self.YOGA_POSES.items()}

    def visualize_pose(
        self,
        frame: np.ndarray,
        joints: Optional[List[JointPoint]] = None,
        draw_skeleton: bool = True
    ) -> np.ndarray:
        """
        Draw detected joints and skeleton on the frame.

        Args:
            frame: Input image frame
            joints: List of JointPoint objects to visualize
            draw_skeleton: Whether to draw connecting lines

        Returns:
            Frame with visualized pose
        """
        output_frame = frame.copy()
        joints = joints or self.detected_joints

        if not joints:
            return output_frame

        # Draw joints as circles
        for joint in joints:
            x, y = int(joint.x), int(joint.y)
            color = (0, 255, 0) if joint.confidence > 0.7 else (0, 255, 255)
            cv2.circle(output_frame, (x, y), 5, color, -1)
            cv2.putText(
                output_frame,
                joint.name,
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # Draw skeleton connections if requested
        if draw_skeleton:
            joint_dict = {j.name: j for j in joints}
            skeleton_connections = [
                ('nose', 'left_shoulder'),
                ('nose', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'),
            ]

            for joint1_name, joint2_name in skeleton_connections:
                if joint1_name in joint_dict and joint2_name in joint_dict:
                    j1 = joint_dict[joint1_name]
                    j2 = joint_dict[joint2_name]
                    cv2.line(
                        output_frame,
                        (int(j1.x), int(j1.y)),
                        (int(j2.x), int(j2.y)),
                        (255, 0, 0),
                        2
                    )

        return output_frame


def create_analyzer() -> PoseAnalyzer:
    """
    Factory function to create a PoseAnalyzer instance.

    Returns:
        Initialized PoseAnalyzer instance
    """
    return PoseAnalyzer()
