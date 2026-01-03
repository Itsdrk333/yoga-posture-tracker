"""
Configuration settings for Yoga Posture Tracker
Created: 2026-01-03 12:13:20 UTC
"""

# Application Settings
APP_NAME = "Yoga Posture Tracker"
APP_VERSION = "1.0.0"
DEBUG = False

# Database Configuration
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "yoga_posture_db",
    "user": "yoga_user",
    "password": "secure_password",
}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TIMEOUT = 30

# Posture Detection Settings
POSTURE_DETECTION = {
    "confidence_threshold": 0.75,
    "min_detection_frames": 5,
    "max_poses_per_image": 1,
}

# Logging Configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/yoga_tracker.log",
}

# Model Configuration
MODEL_CONFIG = {
    "model_type": "mediapipe",
    "confidence_threshold": 0.5,
    "use_gpu": False,
}

# File Upload Settings
UPLOAD_SETTINGS = {
    "max_file_size": 50 * 1024 * 1024,  # 50 MB
    "allowed_extensions": ["jpg", "jpeg", "png", "mp4", "avi"],
    "upload_directory": "uploads/",
}

# Session Configuration
SESSION_CONFIG = {
    "session_timeout": 3600,  # 1 hour in seconds
    "max_sessions": 100,
}

# Email Configuration (optional)
EMAIL_CONFIG = {
    "enabled": False,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
}

# Feature Flags
FEATURES = {
    "enable_real_time_detection": True,
    "enable_pose_correction": True,
    "enable_progress_tracking": True,
    "enable_notifications": False,
}
