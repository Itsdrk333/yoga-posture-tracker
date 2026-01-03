import math
import numpy as np
from utils import calculate_distance, calculate_angle

class PostureDetector: 
    """Detects various yoga poses using MediaPipe landmarks"""
    
    def __init__(self):
        self.pose_name = None
        self.confidence = 0.0
    
    def detect_mountain_pose(self, landmarks):
        """Detect Tadasana (Mountain Pose)"""
        # Check if feet are together and body is straight
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Check hip alignment
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        ankle_distance = calculate_distance(left_ankle, right_ankle)
        hip_distance = calculate_distance(left_hip, right_hip)
        
        # Feet should be close together
        if ankle_distance < 0.1: 
            return True, 0.9
        return False, 0.5
    
    def detect_tree_pose(self, landmarks):
        """Detect Vrksasana (Tree Pose)"""
        # Check if one leg is standing and other is bent
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate knee-hip distances
        left_knee_height = abs(left_knee.y - left_hip.y)
        right_knee_height = abs(right_knee.y - right_hip.y)
        
        # One knee should be significantly higher
        height_diff = abs(left_knee_height - right_knee_height)
        if height_diff > 0.15:
            return True, 0.85
        return False, 0.4
    
    def detect_warrior_pose(self, landmarks):
        """Detect Virabhadrasana (Warrior Pose)"""
        # Check for proper warrior stance
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check if hips and shoulders are aligned
        hip_distance = calculate_distance(left_hip, right_hip)
        shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
        
        if hip_distance > 0.1 and shoulder_distance > 0.1:
            return True, 0.8
        return False, 0.5
    
    def detect_downward_dog(self, landmarks):
        """Detect Adho Mukha Svanasana (Downward Dog)"""
        # Check if body forms an inverted V shape
        nose = landmarks[0]
        left_hand = landmarks[15]
        right_hand = landmarks[16]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Head should be below hip level
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_y = (left_hip.y + right_hip.y) / 2
        
        if nose.y > hip_y:  # Head is down
            # Hands and feet should be spread out
            hand_distance = calculate_distance(left_hand, right_hand)
            ankle_distance = calculate_distance(left_ankle, right_ankle)
            
            if hand_distance > 0.2 and ankle_distance > 0.1:
                return True, 0.9
        
        return False, 0.4