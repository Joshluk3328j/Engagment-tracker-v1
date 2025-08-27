# config.py

import os

# ========== CONFIGURATION CONSTANTS ==========
# Output Directory
OUTPUT_DIR = "logs"
# Ensure the output directory exists when the config is imported
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_BLOCK_DURATION = 5  # seconds for transcription chunks

# Video Processing & Detection Thresholds
VIDEO_FPS_TARGET = 30 # Target FPS for video processing and display
EAR_THRESHOLD = 0.25 # Eye Aspect Ratio for blink detection
EAR_CONSEC_FRAMES_CLOSED = 3 # Number of consecutive frames EAR must be below threshold for a blink
EAR_CONSEC_FRAMES_OPEN = 2 # Number of consecutive frames EAR must be above threshold to confirm blink end

MAR_THRESHOLD = 0.7 # Mouth Aspect Ratio for yawn detection
MAR_CONSEC_FRAMES_OPEN = 5 # Number of consecutive frames MAR must be above threshold for a yawn

# --- FOCUS DETECTION ADJUSTMENT ---
# CRITICAL FIX: Angles from cv2.RQDecomp3x3 are already in degrees.
# Adjusting pitch logic to account for "focused" pitch being near 180 or -180 degrees.
ATTENTION_YAW_THRESHOLD = 25 # Degrees head can turn left/right before considered distracted
# New threshold for pitch: if abs(pitch) is BELOW this, it's considered distracted vertically.
# If abs(pitch) is ABOVE or EQUAL to this, it's considered focused vertically.
PITCH_FOCUSED_MIN_ABS_THRESHOLD = 90 # Minimum absolute pitch value for "focused" vertical attention
ATTENTION_CONSISTENCY_SECONDS = 3 # Seconds attention state must be consistent before logging a change

# Adjusted for more specific "Hand Raised" vs "Too Frequent"
# Decreased this factor to make "Hand Raised" detection more specific to higher hand positions (e.g., above head).
HAND_RAISE_Y_THRESHOLD_FACTOR = 1 # Hand wrist Y relative to eye Y (lower means higher, adjusted for 'above head' detection)
HAND_RAISE_COOLDOWN_SECONDS = 5 # Cooldown for individual "Hand Raised" alerts
# Increased HAND_MOVEMENT_WINDOW_FRAMES to 3 seconds for "Too Frequent" detection
HAND_MOVEMENT_WINDOW_FRAMES = 90 # Number of frames to consider for hand movement std dev (3 seconds at 30 FPS)
HAND_MOVEMENT_STD_THRESHOLD = 0.04 # Standard deviation of hand positions for "frequent movement"
HAND_MOVEMENT_COOLDOWN_SECONDS = 15 # Cooldown for "Frequent Hand Movement" alerts

FATIGUE_YAWN_COUNT = 2 # Number of yawns to trigger fatigue alert
FATIGUE_YAWN_WINDOW_SECONDS = 60 # Time window for yawn count
FATIGUE_YAWN_COOLDOWN_SECONDS = 60 # Cooldown for "Fatigue: Yawning" alert

FATIGUE_BLINK_COUNT = 5 # Number of blinks to trigger fatigue alert
FATIGUE_BLINK_WINDOW_SECONDS = 10 # Time window for blink count
FATIGUE_BLINK_COOLDOWN_SECONDS = 10 # Cooldown for "Fatigue: Blinking" alert

# MediaPipe Face Mesh Indices for EAR/MAR/Pose
# Left eye: 362, 385, 387, 263, 373, 380
# Right eye: 33, 160, 158, 133, 153, 144
# Mouth: 61, 81, 13, 311, 402, 14
# Pose estimation points: Nose (1), Left Eye (33), Right Eye (263), Mouth Left (61), Mouth Right (291), Chin (152)
