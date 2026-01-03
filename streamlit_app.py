import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import math
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from collections import deque
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Yoga Posture Tracker",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feedback-good {
        padding: 15px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
    .feedback-warning {
        padding: 15px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }
    .feedback-error {
        padding: 15px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Define yoga poses with expected joint angles (in degrees)
YOGA_POSES = {
    "Downward Dog": {
        "description": "Head down, hips up in inverted V shape",
        "key_angles": {
            "left_shoulder": (160, 180),
            "right_shoulder": (160, 180),
            "left_hip": (45, 75),
            "right_hip": (45, 75),
            "left_knee": (140, 180),
            "right_knee": (140, 180)
        }
    },
    "Warrior I": {
        "description": "Front leg bent, back leg straight",
        "key_angles": {
            "left_knee": (80, 110),
            "right_knee": (160, 180),
            "left_hip": (80, 120),
            "right_hip": (30, 60),
            "left_shoulder": (150, 180),
            "right_shoulder": (150, 180)
        }
    },
    "Tree Pose": {
        "description": "One leg lifted, standing on the other",
        "key_angles": {
            "standing_knee": (170, 180),
            "lifted_hip": (45, 90),
            "lifted_knee": (80, 120)
        }
    },
    "Child's Pose": {
        "description": "Kneeling with forehead on ground",
        "key_angles": {
            "left_knee": (90, 120),
            "right_knee": (90, 120),
            "left_hip": (40, 70),
            "right_hip": (40, 70)
        }
    },
    "Mountain Pose": {
        "description": "Standing straight, feet together",
        "key_angles": {
            "left_knee": (170, 180),
            "right_knee": (170, 180),
            "left_hip": (170, 180),
            "right_hip": (170, 180)
        }
    }
}

# Landmark indices for key body parts
LANDMARKS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
}

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle(a, b, c):
    """
    Calculate angle between three points (in degrees).
    b is the vertex of the angle.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def extract_landmarks(results):
    """Extract landmarks from MediaPipe results."""
    landmarks = {}
    if results.pose_landmarks:
        for name, idx in LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = (lm.x, lm.y, lm.z, lm.visibility)
    return landmarks

def draw_skeleton(frame, landmarks, confidence_threshold=0.5):
    """Draw skeleton on frame with joint positions."""
    h, w, c = frame.shape
    
    if not landmarks:
        return frame
    
    # Draw landmarks
    for name, (x, y, z, visibility) in landmarks.items():
        if visibility > confidence_threshold:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, name, (cx + 5, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw connections
    connections = [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("nose", "left_eye"),
        ("nose", "right_eye")
    ]
    
    for start, end in connections:
        if start in landmarks and end in landmarks:
            x1, y1, z1, v1 = landmarks[start]
            x2, y2, z2, v2 = landmarks[end]
            if v1 > confidence_threshold and v2 > confidence_threshold:
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    
    return frame

def calculate_joint_angles(landmarks):
    """Calculate angles for all major joints."""
    angles = {}
    
    if len(landmarks) < len(LANDMARKS):
        return angles
    
    # Shoulder angles
    if all(k in landmarks for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        angles["left_shoulder"] = calculate_angle(
            landmarks["left_elbow"][:2],
            landmarks["left_shoulder"][:2],
            landmarks["left_hip"][:2]
        )
    
    if all(k in landmarks for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["right_shoulder"] = calculate_angle(
            landmarks["right_elbow"][:2],
            landmarks["right_shoulder"][:2],
            landmarks["right_hip"][:2]
        )
    
    # Elbow angles
    if all(k in landmarks for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        angles["left_elbow"] = calculate_angle(
            landmarks["left_shoulder"][:2],
            landmarks["left_elbow"][:2],
            landmarks["left_wrist"][:2]
        )
    
    if all(k in landmarks for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["right_elbow"] = calculate_angle(
            landmarks["right_shoulder"][:2],
            landmarks["right_elbow"][:2],
            landmarks["right_wrist"][:2]
        )
    
    # Hip angles
    if all(k in landmarks for k in ["left_shoulder", "left_hip", "left_knee"]):
        angles["left_hip"] = calculate_angle(
            landmarks["left_shoulder"][:2],
            landmarks["left_hip"][:2],
            landmarks["left_knee"][:2]
        )
    
    if all(k in landmarks for k in ["right_shoulder", "right_hip", "right_knee"]):
        angles["right_hip"] = calculate_angle(
            landmarks["right_shoulder"][:2],
            landmarks["right_hip"][:2],
            landmarks["right_knee"][:2]
        )
    
    # Knee angles
    if all(k in landmarks for k in ["left_hip", "left_knee", "left_ankle"]):
        angles["left_knee"] = calculate_angle(
            landmarks["left_hip"][:2],
            landmarks["left_knee"][:2],
            landmarks["left_ankle"][:2]
        )
    
    if all(k in landmarks for k in ["right_hip", "right_knee", "right_ankle"]):
        angles["right_knee"] = calculate_angle(
            landmarks["right_hip"][:2],
            landmarks["right_knee"][:2],
            landmarks["right_ankle"][:2]
        )
    
    return angles

def evaluate_pose_accuracy(selected_pose, angles):
    """Evaluate accuracy of performed pose against reference angles."""
    if selected_pose not in YOGA_POSES:
        return None, []
    
    pose_config = YOGA_POSES[selected_pose]
    key_angles = pose_config["key_angles"]
    
    accuracy_scores = []
    feedback_items = []
    
    for joint_name, (min_angle, max_angle) in key_angles.items():
        if joint_name in angles:
            actual_angle = angles[joint_name]
            
            if min_angle <= actual_angle <= max_angle:
                accuracy = 100
                feedback = f"✓ {joint_name}: {actual_angle:.1f}° (Perfect)"
            else:
                if actual_angle < min_angle:
                    diff = min_angle - actual_angle
                    accuracy = max(0, 100 - diff * 2)
                    feedback = f"✗ {joint_name}: {actual_angle:.1f}° (Too small, target: {min_angle}-{max_angle}°)"
                else:
                    diff = actual_angle - max_angle
                    accuracy = max(0, 100 - diff * 2)
                    feedback = f"✗ {joint_name}: {actual_angle:.1f}° (Too large, target: {min_angle}-{max_angle}°)"
            
            accuracy_scores.append(accuracy)
            feedback_items.append((feedback, accuracy))
    
    overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
    
    return overall_accuracy, feedback_items

def generate_real_time_feedback(overall_accuracy):
    """Generate real-time feedback based on accuracy."""
    feedback = []
    
    if overall_accuracy >= 90:
        feedback.append(("🌟 Excellent form! Keep it up!", "good"))
    elif overall_accuracy >= 75:
        feedback.append(("👍 Good form! Minor adjustments needed.", "warning"))
    elif overall_accuracy >= 60:
        feedback.append(("⚠️ Moderate form. Focus on proper alignment.", "warning"))
    else:
        feedback.append(("❌ Needs improvement. Check your body alignment.", "error"))
    
    return feedback

def process_frame(frame, selected_pose):
    """Process frame for pose detection and analysis."""
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks is None:
        return frame, None, None, "No pose detected"
    
    # Extract landmarks and calculate angles
    landmarks = extract_landmarks(results)
    angles = calculate_joint_angles(landmarks)
    
    # Draw skeleton
    frame = draw_skeleton(frame, landmarks)
    
    # Evaluate pose accuracy if a pose is selected
    overall_accuracy = None
    feedback_items = []
    
    if selected_pose and selected_pose != "Free Mode":
        overall_accuracy, feedback_items = evaluate_pose_accuracy(selected_pose, angles)
    
    return frame, angles, overall_accuracy, feedback_items

def main():
    st.markdown("<h1 class='main-header'>🧘 Yoga Posture Tracker</h1>", unsafe_allow_html=True)
    st.markdown("Real-time yoga pose detection and accuracy analysis with MediaPipe")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Live Webcam", "Video Upload", "Pose Information"]
    )
    
    selected_pose = st.sidebar.selectbox(
        "Select Yoga Pose:",
        ["Free Mode"] + list(YOGA_POSES.keys())
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        0.3, 1.0, 0.7, 0.05
    )
    
    # Main content area
    if mode == "Live Webcam":
        st.subheader("📹 Live Webcam Feed")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if selected_pose != "Free Mode":
                st.info(f"📋 {selected_pose}: {YOGA_POSES[selected_pose]['description']}")
            
            # Create placeholder for video
            video_placeholder = st.empty()
            metrics_placeholder = st.empty()
            feedback_placeholder = st.empty()
            angles_placeholder = st.empty()
        
        with col2:
            st.markdown("### Metrics")
            accuracy_metric = st.empty()
            detection_metric = st.empty()
            fps_metric = st.empty()
        
        # Start webcam capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check permissions.")
            return
        
        frame_count = 0
        accuracy_history = deque(maxlen=30)
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Resize frame for better performance
                frame = cv2.resize(frame, (640, 480))
                
                # Process frame
                processed_frame, angles, overall_accuracy, feedback_items = process_frame(
                    frame, selected_pose
                )
                
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update metrics
                frame_count += 1
                
                if overall_accuracy is not None:
                    accuracy_history.append(overall_accuracy)
                
                # Display frame
                video_placeholder.image(display_frame, use_column_width=True)
                
                # Update accuracy metric
                if accuracy_history:
                    avg_accuracy = np.mean(accuracy_history)
                    accuracy_metric.metric("Accuracy", f"{avg_accuracy:.1f}%")
                
                detection_metric.metric("Frames Processed", frame_count)
                fps_metric.metric("Status", "🟢 Detecting" if angles else "🔴 No Detection")
                
                # Display angles
                if angles:
                    angle_df = pd.DataFrame(
                        list(angles.items()),
                        columns=["Joint", "Angle (°)"]
                    )
                    angles_placeholder.dataframe(angle_df, use_container_width=True)
                
                # Display feedback
                if feedback_items:
                    with feedback_placeholder.container():
                        st.markdown("### Pose Feedback")
                        for feedback_text, accuracy in feedback_items:
                            if accuracy >= 90:
                                st.markdown(f"<div class='feedback-good'>{feedback_text}</div>", 
                                          unsafe_allow_html=True)
                            elif accuracy >= 60:
                                st.markdown(f"<div class='feedback-warning'>{feedback_text}</div>", 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='feedback-error'>{feedback_text}</div>", 
                                          unsafe_allow_html=True)
                
                # Real-time feedback
                if overall_accuracy is not None:
                    rt_feedback = generate_real_time_feedback(overall_accuracy)
                    for feedback_text, feedback_type in rt_feedback:
                        if feedback_type == "good":
                            metrics_placeholder.success(feedback_text)
                        elif feedback_type == "warning":
                            metrics_placeholder.warning(feedback_text)
                        else:
                            metrics_placeholder.error(feedback_text)
                
                # Exit condition (press 'q' in the window)
                if st.button("Stop Webcam"):
                    break
        
        finally:
            cap.release()
    
    elif mode == "Video Upload":
        st.subheader("📤 Video Upload and Analysis")
        
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                cap = cv2.VideoCapture(tmp_path)
                
                if not cap.isOpened():
                    st.error("Cannot read video file")
                    return
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                st.info(f"📊 Video Info: {total_frames} frames @ {fps:.1f} FPS")
                
                if selected_pose != "Free Mode":
                    st.info(f"📋 Analyzing for: {YOGA_POSES[selected_pose]['description']}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                video_output = st.empty()
                angles_data = []
                accuracy_scores = []
                
                frame_idx = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, (640, 480))
                    processed_frame, angles, overall_accuracy, feedback_items = process_frame(
                        frame, selected_pose
                    )
                    
                    if angles:
                        angles["frame"] = frame_idx
                        angles_data.append(angles)
                    
                    if overall_accuracy is not None:
                        accuracy_scores.append(overall_accuracy)
                    
                    frame_idx += 1
                    progress = frame_idx / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {frame_idx}/{total_frames} frames")
                    
                    if frame_idx % 10 == 0:  # Display every 10th frame
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        video_output.image(display_frame, use_column_width=True)
                
                cap.release()
                
                st.success("✅ Video analysis complete!")
                
                # Display analysis results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Frames", len(angles_data))
                
                with col2:
                    if accuracy_scores:
                        st.metric("Avg Accuracy", f"{np.mean(accuracy_scores):.1f}%")
                
                with col3:
                    if accuracy_scores:
                        st.metric("Max Accuracy", f"{np.max(accuracy_scores):.1f}%")
                
                # Plot accuracy over time
                if accuracy_scores:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=accuracy_scores,
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='rgb(31, 119, 180)', width=2),
                        marker=dict(size=4)
                    ))
                    fig.update_layout(
                        title="Pose Accuracy Over Time",
                        xaxis_title="Frame",
                        yaxis_title="Accuracy (%)",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display angle trends
                if angles_data:
                    st.subheader("Joint Angle Trends")
                    
                    angle_names = set()
                    for angle_dict in angles_data:
                        angle_names.update(angle_dict.keys())
                    angle_names.discard("frame")
                    
                    selected_angles = st.multiselect(
                        "Select angles to visualize:",
                        sorted(angle_names),
                        default=list(sorted(angle_names))[:3]
                    )
                    
                    if selected_angles:
                        fig = go.Figure()
                        
                        for angle_name in selected_angles:
                            angles_list = [d.get(angle_name) for d in angles_data]
                            fig.add_trace(go.Scatter(
                                y=angles_list,
                                mode='lines+markers',
                                name=angle_name,
                                line=dict(width=2),
                                marker=dict(size=4)
                            ))
                        
                        fig.update_layout(
                            title="Joint Angles Over Time",
                            xaxis_title="Frame",
                            yaxis_title="Angle (°)",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    elif mode == "Pose Information":
        st.subheader("📚 Yoga Pose Information")
        
        for pose_name, pose_info in YOGA_POSES.items():
            with st.expander(f"🧘 {pose_name}"):
                st.write(f"**Description:** {pose_info['description']}")
                
                st.write("**Key Angle Ranges:**")
                angle_info = pd.DataFrame([
                    {"Joint": joint, "Min (°)": angles[0], "Max (°)": angles[1]}
                    for joint, angles in pose_info['key_angles'].items()
                ])
                st.dataframe(angle_info, use_container_width=True)
                
                st.write("**Tips:**")
                tips = {
                    "Downward Dog": [
                        "Keep your hands shoulder-width apart",
                        "Press your palms firmly into the ground",
                        "Keep your head between your arms",
                        "Engage your core and straighten your back"
                    ],
                    "Warrior I": [
                        "Front knee should be over ankle",
                        "Back heel can be slightly elevated",
                        "Reach your arms up and back",
                        "Keep your torso upright"
                    ],
                    "Tree Pose": [
                        "Plant the standing foot firmly",
                        "Press the lifted foot against the inner thigh",
                        "Bring hands to heart center or above head",
                        "Gaze at a fixed point for balance"
                    ],
                    "Child's Pose": [
                        "Knees should be hip-width apart",
                        "Big toes touch at the back",
                        "Forehead rests on the ground",
                        "Arms can be extended or beside the body"
                    ],
                    "Mountain Pose": [
                        "Feet together or hip-width apart",
                        "Weight distributed evenly",
                        "Arms at sides with palms facing forward",
                        "Engage your thighs and straighten your spine"
                    ]
                }
                
                for tip in tips.get(pose_name, []):
                    st.write(f"• {tip}")

if __name__ == "__main__":
    main()
