import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import time
import os
import csv
from datetime import datetime
import mediapipe as mp
from collections import deque
from faster_whisper import WhisperModel
from fpdf import FPDF
import tempfile
import shutil
import math

# ========== CONFIGURATION CONSTANTS ==========
# Output Directory
OUTPUT_DIR = "logs"
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
HAND_RAISE_Y_THRESHOLD_FACTOR = 0.4 # Hand wrist Y relative to eye Y (lower means higher, adjusted for 'above head' detection)
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

# ========== UTILITY FUNCTIONS ==========
def timestamp():
    """Returns current time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

def save_csv(filename, rows, headers):
    """Saves data to a CSV file with given headers."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def generate_pdf_transcript(transcriptions):
    """Generates a PDF of all transcriptions."""
    pdf = FPDF()
    pdf.add_page()
    # Use a standard font like 'helvetica' to avoid deprecation warnings and rendering issues
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 10, "Lecture Transcriptions\n\n")
    # Explicitly set margins for consistent available width
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    # Calculate available page width
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    for ts, text in transcriptions:
        # Ensure text is not empty before attempting to render
        if text.strip():
            # Changed h from 10 to 12 and added pdf.ln(5) for better readability
            pdf.multi_cell(page_width, 12, f"[{ts}] {text}")
            pdf.ln(5)
        else:
            pdf.multi_cell(page_width, 12, f"[{ts}] (No speech detected)")
            pdf.ln(5)
    pdf.output(os.path.join(OUTPUT_DIR, "transcription_summary.pdf"))

def generate_pdf_logs(events):
    """Generates a PDF summary of all logged events."""
    pdf = FPDF()
    pdf.add_page()
    # Use a standard font like 'helvetica' to avoid deprecation warnings and rendering issues
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 10, "Engagement Log Summary\n\n")
    # Explicitly set margins for consistent available width
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    # Calculate available page width
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    for ts, event_type, description, speech_context in events:
        log_entry = f"[{ts}] {event_type}: {description}"
        if speech_context:
            log_entry += f"\n    Speech Context: \"{speech_context}\""
        # Ensure text is not empty before attempting to render
        if log_entry.strip():
            # Changed h from 10 to 12 and added pdf.ln(5) for better readability
            pdf.multi_cell(page_width, 12, log_entry)
            pdf.ln(5)
        else:
            pdf.multi_cell(page_width, 12, f"[{ts}] (Empty Log Entry)")
            pdf.ln(5)
    pdf.output(os.path.join(OUTPUT_DIR, "engagement_log_summary.pdf"))

# ========== FACIAL LANDMARK UTILITIES ==========
def euclidean_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_eye_aspect_ratio(eye_landmarks):
    """Calculates EAR for a single eye given 6 landmarks."""
    # Vertical distances
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def get_mouth_aspect_ratio(mouth_landmarks):
    """Calculates MAR for the mouth given 6 landmarks."""
    # Vertical distances
    A = euclidean_distance(mouth_landmarks[1], mouth_landmarks[5])
    B = euclidean_distance(mouth_landmarks[2], mouth_landmarks[4])
    # Horizontal distance
    C = euclidean_distance(mouth_landmarks[0], mouth_landmarks[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def get_head_pose(face_landmarks, image_shape):
    """
    Estimates head pose (pitch, yaw, roll) from face landmarks.
    Returns angles in degrees.
    """
    img_h, img_w, _ = image_shape
    # 2D image points from MediaPipe (normalized to image size)
    # Using specific landmarks for pose estimation
    # Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
    image_points = np.array([
        (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),    # Nose tip
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h), # Chin
        (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),   # Left eye corner
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h), # Right eye corner
        (face_landmarks[61].x * img_w, face_landmarks[61].y * img_h),   # Left mouth corner
        (face_landmarks[291].x * img_w, face_landmarks[291].y * img_h)  # Right mouth corner
    ], dtype="double")

    # 3D model points (arbitrary but consistent model of a human head)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals (assuming a generic webcam)
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1)) # No distortion assumed

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rmat, jacobian = cv2.Rodrigues(rotation_vector)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # --- CRITICAL FIX: Use angles directly as they are already in degrees from OpenCV ---
    x = angles[0] # Pitch
    y = angles[1] # Yaw
    z = angles[2] # Roll
    # --- END CRITICAL FIX ---

    return x, y, z # Pitch, Yaw, Roll in degrees

# ========== ENGAGEMENT LOGIC CLASS ==========
class EngagementLogic:
    def __init__(self, logger_callback):
        self.logger = logger_callback # This will be EngageTrackApp.log_event

        # Attention tracking
        # States: "Focused", "Distracted", "Logged_Distraction"
        self._current_attention_state = "Focused" # Initial state
        self._distraction_start_time = 0 # Timestamp when distraction began
        self.last_logged_attention_state = "Focused" # The last state that was actually logged

        # Hand tracking
        self.hand_events_deque = deque() # Stores timestamps of hand events
        self.hand_cooldown_end_time = 0 # End time of "Too Frequent" cooldown
        self.last_hand_raised_log_time = 0 # Cooldown for individual "Hand Raised" logs

        # Fatigue tracking
        self.yawn_events_deque = deque() # Stores timestamps of yawn events
        self.yawn_cooldown_end_time = 0 # End time of "Fatigue: Yawning" cooldown

        self.blink_events_deque = deque() # Stores timestamps of blink events
        self.blink_cooldown_end_time = 0 # End time of "Fatigue: Blinking" cooldown

        # Raw blink/yawn detection state for confirmed events
        self._is_eye_closed = False
        self._frames_eye_closed = 0
        self._is_mouth_open = False
        self._frames_mouth_open = 0

    def _now(self):
        """Helper to get current time."""
        return time.time()

    def update_attention(self, is_currently_focused: bool, current_yaw: float, current_pitch: float):
        """
        Updates the attention state based on raw input and logs changes using a state machine.
        """
        now = self._now()

        # State machine logic for attention
        if self._current_attention_state == "Focused":
            if not is_currently_focused:
                # Transition from Focused to Distracted
                self._distraction_start_time = now
                self._current_attention_state = "Distracted"
        
        elif self._current_attention_state == "Distracted":
            if is_currently_focused:
                # Transition back to Focused before logging threshold
                self._current_attention_state = "Focused"
            else:
                # Still distracted, check if logging threshold is met
                distraction_duration = now - self._distraction_start_time
                if distraction_duration >= ATTENTION_CONSISTENCY_SECONDS:
                    # Log "Distracted" event
                    direction = ""
                    if abs(current_yaw) > ATTENTION_YAW_THRESHOLD:
                        direction = "sideways"
                    # Check if pitch is outside the "focused" range (i.e., abs(current_pitch) is too low)
                    if abs(current_pitch) < PITCH_FOCUSED_MIN_ABS_THRESHOLD:
                        # Determine if looking up or down based on the sign of pitch
                        if current_pitch > 0: 
                            direction = "down" if not direction else direction + " and down"
                        else:
                            direction = "up" if not direction else direction + " and up"
                    
                    log_description = f"Distracted (looking {direction})" if direction else "Distracted"
                    self.logger(event_type="Attention", description=log_description, timestamp=now)
                    self.last_logged_attention_state = log_description # Update last logged state
                    self._current_attention_state = "Logged_Distraction" # Move to logged state

        elif self._current_attention_state == "Logged_Distraction":
            if is_currently_focused:
                # Transition from Logged_Distraction back to Focused
                self.logger(event_type="Attention", description="Focused", timestamp=now)
                self.last_logged_attention_state = "Focused" # Update last logged state
                self._current_attention_state = "Focused"
            # If still distracted, do nothing (already logged)

    def detect_and_register_blink(self, ear):
        """Detects a complete blink event and registers it with the logic."""
        now = self._now()
        if now < self.blink_cooldown_end_time: # Still in cooldown for fatigue blinking
            return

        if ear < EAR_THRESHOLD:
            self._frames_eye_closed += 1
            if not self._is_eye_closed and self._frames_eye_closed >= EAR_CONSEC_FRAMES_CLOSED:
                self._is_eye_closed = True # Confirmed eye is closed
        else:
            if self._is_eye_closed: # Eye was closed and now opened
                if self._frames_eye_closed >= EAR_CONSEC_FRAMES_CLOSED: # Was a valid closure
                    # This is a complete blink event
                    self.blink_events_deque.append(now)
                    # Clean up old events
                    while self.blink_events_deque and now - self.blink_events_deque[0] > FATIGUE_BLINK_WINDOW_SECONDS:
                        self.blink_events_deque.popleft()

                    # Check if fatigue threshold is met
                    if len(self.blink_events_deque) >= FATIGUE_BLINK_COUNT:
                        # Changed log message as per user request
                        self.logger(event_type="Fatigue", description="Blink: tired fatigue", timestamp=now)
                        self.blink_cooldown_end_time = now + FATIGUE_BLINK_COOLDOWN_SECONDS
                self._is_eye_closed = False
            self._frames_eye_closed = 0 # Reset frame counter

    def detect_and_register_yawn(self, mar):
        """Detects a complete yawn event and registers it with the logic."""
        now = self._now()
        if now < self.yawn_cooldown_end_time: # Still in cooldown for fatigue yawning
            return

        if mar > MAR_THRESHOLD:
            self._frames_mouth_open += 1
            if not self._is_mouth_open and self._frames_mouth_open >= MAR_CONSEC_FRAMES_OPEN:
                self._is_mouth_open = True # Confirmed mouth is open for a yawn
        else:
            if self._is_mouth_open: # Mouth was open and now closed
                if self._frames_mouth_open >= MAR_CONSEC_FRAMES_OPEN: # Was a valid open
                    # This is a complete yawn event
                    self.yawn_events_deque.append(now)
                    # Clean up old events
                    while self.yawn_events_deque and now - self.yawn_events_deque[0] > FATIGUE_YAWN_WINDOW_SECONDS:
                        self.yawn_events_deque.popleft()

                    # Check if fatigue threshold is met
                    if len(self.yawn_events_deque) >= FATIGUE_YAWN_COUNT:
                        self.logger(event_type="Fatigue", description="Yawning", timestamp=now)
                        self.yawn_cooldown_end_time = now + FATIGUE_YAWN_COOLDOWN_SECONDS
                self._is_mouth_open = False
            self._frames_mouth_open = 0 # Reset frame counter

    def register_hand_event(self, is_hand_raised_now: bool, hand_positions_std: float):
        """
        Registers a hand event. Logs "Hand Raised" for single events
        and "Hand Detected" for sustained high activity.
        Logic updated to prioritize "Hand Raised" detection.
        """
        now = self._now()

        # 1. First, check for "Hand Raised" (more specific gesture)
        if is_hand_raised_now and now - self.last_hand_raised_log_time > HAND_RAISE_COOLDOWN_SECONDS:
            self.logger(event_type="Hand Motion", description="Hand Raised", timestamp=now)
            self.last_hand_raised_log_time = now
            # When a hand is raised, also put "Hand Detected" on cooldown
            # to prevent it from immediately triggering due to the raising motion itself.
            self.hand_cooldown_end_time = now + HAND_MOVEMENT_COOLDOWN_SECONDS
            return 

        # 2. Then, check for "Hand Detected" (general fidgeting)
        # This check respects its own cooldown, which might have been set by a "Hand Raised" event.
        if now < self.hand_cooldown_end_time:
            return # Skip processing if still in cooldown

        # Add event timestamp for "Hand Detected" tracking
        self.hand_events_deque.append(now)
        while self.hand_events_deque and now - self.hand_events_deque[0] > HAND_MOVEMENT_WINDOW_FRAMES / VIDEO_FPS_TARGET:
            self.hand_events_deque.popleft()

        if len(self.hand_events_deque) >= (HAND_MOVEMENT_WINDOW_FRAMES / VIDEO_FPS_TARGET) * 1 and \
           hand_positions_std > HAND_MOVEMENT_STD_THRESHOLD:
            self.logger(event_type="Hand Motion", description="Hand Detected", timestamp=now) # Changed from "Too Frequent"
            self.hand_cooldown_end_time = now + HAND_MOVEMENT_COOLDOWN_SECONDS


# ========== MAIN APPLICATION CLASS ==========
class EngageTrackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EngageTrack: Intelligent Multimodal Engagement System")
        self.root.configure(bg="#1e1e1e")
        self.running = False

        # Data storage for reports
        self.transcriptions = [] # (timestamp, text)
        self.events = []         # (timestamp, event_type, description, speech_context)
        self.audio_queue = queue.Queue()
        self.last_transcription = "" # To provide speech context for events

        # MediaPipe setup (initialized once)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1) # Focus on one hand for simplicity

        # Engagement Logic instance
        self.logic = EngagementLogic(self._log_event_and_update_ui)

        # Video capture (initialized in video_loop)
        self.cap = None

        self.build_welcome_ui()

    def build_welcome_ui(self):
        """Builds the initial welcome screen UI."""
        self.clear_root()
        tk.Label(self.root, text="EngageTrack", fg="#ffffff", bg="#1e1e1e", font=("Helvetica", 32, "bold")).pack(pady=40)
        tk.Button(self.root, text="Start Session", command=self.start_session, font=("Helvetica", 16),
                  bg="#4CAF50", fg="#fff", activebackground="#66BB6A", relief="raised", bd=3).pack(pady=20)
        tk.Label(self.root, text="Press 'End Session' to generate reports.", fg="#aaa", bg="#1e1e1e", font=("Helvetica", 10)).pack(pady=10)


    def build_main_ui(self):
        """Builds the main session UI with video feed and logs."""
        self.clear_root()

        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3) # Video takes more space
        self.root.grid_columnconfigure(1, weight=1) # Sidebar

        # Video feed label
        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        # Sidebar frame
        sidebar = tk.Frame(self.root, bg="#121212", width=300)
        sidebar.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        sidebar.grid_rowconfigure(2, weight=1) # Make log area expandable

        tk.Label(sidebar, text="Live Status", fg="#00BFFF", bg="#121212", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.status_attention = tk.Label(sidebar, text="Attention: N/A", fg="#eee", bg="#121212", font=("Helvetica", 12))
        self.status_attention.pack(anchor="w", padx=10, pady=2)
        self.status_fatigue = tk.Label(sidebar, text="Fatigue: N/A", fg="#eee", bg="#121212", font=("Helvetica", 12))
        self.status_fatigue.pack(anchor="w", padx=10, pady=2)
        self.status_hand = tk.Label(sidebar, text="Hand: N/A", fg="#eee", bg="#121212", font=("Helvetica", 12))
        self.status_hand.pack(anchor="w", padx=10, pady=2)
        self.status_transcription = tk.Label(sidebar, text="Speech: ...", fg="#eee", bg="#121212", font=("Helvetica", 12), wraplength=280, justify="left")
        self.status_transcription.pack(anchor="w", padx=10, pady=5)


        tk.Label(sidebar, text="Live Log", fg="white", bg="#121212", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.log_area = scrolledtext.ScrolledText(sidebar, bg="#1e1e1e", fg="white", height=15, width=40, font=("Consolas", 10))
        self.log_area.pack(padx=5, pady=5, fill="both", expand=True)

        # End Session Button
        tk.Button(self.root, text="End Session", command=self.end_session,
                  bg="#DC3545", fg="white", font=("Helvetica", 14, "bold"),
                  activebackground="#C82333", relief="raised", bd=3).grid(row=1, column=1, pady=10, sticky="s")

    def clear_root(self):
        """Destroys all widgets in the root window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def _log_event_and_update_ui(self, event_type: str, description: str, timestamp: float):
        """
        Logs an event to the internal list and updates the UI log area.
        This is called by EngagementLogic when a *logged* event occurs (after consistency checks).
        """
        ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        speech_context = self.last_transcription # Get the most recent transcription

        log_message = f"[{ts_str}] {event_type}: {description}"
        if speech_context:
            log_message += f" (Context: \"{speech_context}\")"
        log_message += "\n"

        self.log_area.insert(tk.END, log_message)
        self.log_area.see(tk.END) # Scroll to bottom

        # Store for final report generation
        self.events.append((ts_str, event_type, description, speech_context))

        # Update live status labels based on the *logged* event
        # These are for confirmed, consistent states
        # The instant updates are handled by _update_live_status_labels_instant
        # We can still ensure the color is accurate if a logged event happens
        if event_type == "Attention":
            self.status_attention.config(fg="#4CAF50" if description == "Focused" else "#FFC107")
        elif event_type == "Fatigue":
            self.status_fatigue.config(fg="#DC3545") # Red for fatigue
        elif event_type == "Hand Motion":
            # The description could be "Hand Raised" or "Hand Detected"
            self.status_hand.config(fg="#17A2B8") # Blue for hand motion


    def _update_live_status_labels_instant(self, attention_state: str, fatigue_state: str, hand_state: str):
        """
        Updates the live status labels on the UI instantly based on raw detection,
        without waiting for consistency/logging.
        """
        self.status_attention.config(text=f"Attention: {attention_state}")
        self.status_attention.config(fg="#4CAF50" if attention_state == "Focused" else "#FFC107" if attention_state == "Distracted" else "#eee")

        self.status_fatigue.config(text=f"Fatigue: {fatigue_state}")
        self.status_fatigue.config(fg="#DC3545" if fatigue_state != "N/A" and fatigue_state != "Normal" else "#eee" if fatigue_state == "N/A" else "#FFC107" if fatigue_state == "Potential Fatigue" else "#eee")

        self.status_hand.config(text=f"Hand: {hand_state}")
        # Updated to reflect "Hand Detected" as the instant status
        self.status_hand.config(fg="#17A2B8" if hand_state != "N/A" and hand_state != "No Hand Detected" else "#eee")


    def start_session(self):
        """Initializes and starts all processing threads."""
        self.running = True
        self.build_main_ui()

        self.transcriptions.clear()
        self.events.clear()
        self.last_transcription = ""
        self.logic = EngagementLogic(self._log_event_and_update_ui) # Reset logic state for new session

        # Start threads
        threading.Thread(target=self.video_loop, daemon=True).start()
        threading.Thread(target=self.audio_loop, daemon=True).start()
        threading.Thread(target=self.transcription_loop, daemon=True).start()

    def end_session(self):
        """Stops all processing, saves logs, and generates reports."""
        self.running = False
        # Give threads a moment to finish
        time.sleep(1.5)

        # Save all collected data to CSVs
        save_csv(os.path.join(OUTPUT_DIR, "transcription.csv"),
                 self.transcriptions, ["Timestamp", "Transcription"])
        save_csv(os.path.join(OUTPUT_DIR, "engagement_log.csv"),
                 self.events, ["Timestamp", "EventType", "Description", "SpeechContext"])

        # Generate PDFs
        generate_pdf_transcript(self.transcriptions)
        generate_pdf_logs(self.events)

        messagebox.showinfo("Session Ended", f"Logs and PDFs saved in '{OUTPUT_DIR}' folder.")
        self.build_welcome_ui()

    def video_loop(self):
        """
        Captures video, performs facial and hand landmark detection,
        and updates engagement logic.
        """
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            self.running = False
            return

        # Buffers for raw detection for event confirmation
        ear_history = deque(maxlen=EAR_CONSEC_FRAMES_CLOSED + EAR_CONSEC_FRAMES_OPEN + 5) # Enough history for blink detection
        mar_history = deque(maxlen=MAR_CONSEC_FRAMES_OPEN + 5) # Enough history for yawn detection
        hand_y_positions = deque(maxlen=HAND_MOVEMENT_WINDOW_FRAMES) # For frequent movement std dev

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Flip frame for selfie-view and better processing
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width, _ = frame.shape

            # Process face landmarks
            face_results = self.mp_face_mesh.process(rgb_frame)
            hand_results = self.mp_hands.process(rgb_frame)

            # Initialize current states for instant UI update
            current_attention_state_instant = "N/A"
            current_fatigue_state_instant = "N/A"
            current_hand_state_instant = "N/A"

            # --- Process Face Data ---
            if face_results.multi_face_landmarks:
                lm = face_results.multi_face_landmarks[0].landmark
                # Convert normalized landmarks to pixel coordinates
                coords = lambda idxs: [(int(lm[i].x * frame_width), int(lm[i].y * frame_height)) for i in idxs]

                # Eye and Mouth Landmarks for EAR/MAR
                left_eye_indices = [362, 385, 387, 263, 373, 380] # P1, P2, P3, P4, P5, P6
                right_eye_indices = [33, 160, 158, 133, 153, 144] # P1, P2, P3, P4, P5, P6
                mouth_indices = [61, 81, 13, 311, 402, 14] # P1, P2, P3, P4, P5, P6

                left_eye_coords = coords(left_eye_indices)
                right_eye_coords = coords(right_eye_indices)
                mouth_coords = coords(mouth_indices)

                ear = (get_eye_aspect_ratio(left_eye_coords) + get_eye_aspect_ratio(right_eye_coords)) / 2
                mar = get_mouth_aspect_ratio(mouth_coords)

                ear_history.append(ear)
                mar_history.append(mar)

                self.logic.detect_and_register_blink(ear)
                self.logic.detect_and_register_yawn(mar)

                # Head Pose for Attention
                pitch, yaw, roll = get_head_pose(lm, frame.shape)
                
                # --- DIAGNOSTIC PRINT ---
                # Removed the diagnostic print for Yaw and Pitch as requested by the user.
                # print(f"Current Yaw: {yaw:.2f}°, Current Pitch: {pitch:.2f}°")
                # --- END DIAGNOSTIC PRINT ---

                # Determine raw attention state for instant UI update
                # Focused if yaw is within threshold AND absolute pitch is above minimum threshold
                is_currently_focused = (abs(yaw) <= ATTENTION_YAW_THRESHOLD) and \
                                       (abs(pitch) >= PITCH_FOCUSED_MIN_ABS_THRESHOLD)
                current_attention_state_instant = "Focused" if is_currently_focused else "Distracted"

                # Pass to logic for consistent logging using the state machine
                self.logic.update_attention(is_currently_focused, pitch, yaw)

                # Set instant fatigue state for UI
                if self.logic._is_eye_closed or self.logic._is_mouth_open:
                    current_fatigue_state_instant = "Potential Fatigue"
                elif self.logic.blink_cooldown_end_time > self.logic._now() or \
                     self.logic.yawn_cooldown_end_time > self.logic._now():
                    current_fatigue_state_instant = "Fatigue Detected"
                else:
                    current_fatigue_state_instant = "Normal"


                # Optionally draw landmarks for visualization (can be removed for performance)
                mp.solutions.drawing_utils.draw_landmarks(
                    rgb_frame, face_results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

            else:
                # No face detected: update attention state and reset fatigue/hand buffers
                current_attention_state_instant = "No Face Detected"
                current_fatigue_state_instant = "N/A" # No fatigue detection without face
                self.logic.update_attention(False, 0, 0) # Treat as not focused if no face
                ear_history.clear()
                mar_history.clear()
                hand_y_positions.clear() # Clear hand positions if no face to reference

            # --- Process Hand Data ---
            is_hand_raised_now = False
            current_hand_std = 0

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Wrist landmark (index 0)
                    wrist_y_norm = hand_landmarks.landmark[0].y
                    
                    # Reference Y-coordinate from face (e.g., average eye level)
                    # Only calculate if face is detected, otherwise use a default or skip
                    if face_results.multi_face_landmarks:
                        eye_y_norm = (lm[33].y + lm[263].y) / 2 # Average Y of left and right eye
                    else:
                        eye_y_norm = 0.5 # Default to middle of screen if no face

                    # Check if hand is above a certain level relative to eyes
                    # Lower Y-coordinate means higher on screen
                    if wrist_y_norm < eye_y_norm * HAND_RAISE_Y_THRESHOLD_FACTOR:
                        is_hand_raised_now = True
                        current_hand_state_instant = "Hand Raised" # Instant update for UI

                    # Collect wrist Y positions for frequent movement analysis
                    hand_y_positions.append(wrist_y_norm)
                    if len(hand_y_positions) == HAND_MOVEMENT_WINDOW_FRAMES:
                        current_hand_std = np.std(list(hand_y_positions)) # Convert deque to list for std dev
                        # If std dev is high, indicate potential frequent movement for instant UI
                        if current_hand_std > HAND_MOVEMENT_STD_THRESHOLD:
                            current_hand_state_instant = "Hand Detected" # Changed from "Frequent Movement"

                    # Optionally draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        rgb_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))
            else:
                current_hand_state_instant = "No Hand Detected"

            # Register hand event with logic (for consistent logging)
            self.logic.register_hand_event(is_hand_raised_now, current_hand_std)


            # --- Update UI ---
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk # Keep a reference!
            self.video_label.configure(image=imgtk)

            # Update live status labels instantly
            self._update_live_status_labels_instant(
                current_attention_state_instant,
                current_fatigue_state_instant,
                current_hand_state_instant
            )

            # Control frame rate for video display
            time.sleep(1 / VIDEO_FPS_TARGET)

        # Release camera resources when loop ends
        if self.cap:
            self.cap.release()
        self.mp_face_mesh.close()
        self.mp_hands.close()


    def audio_loop(self):
        """Captures audio blocks and puts them into a queue for transcription."""
        def callback(indata, frames, time_info, status):
            if self.running:
                self.audio_queue.put(indata.copy())

        try:
            with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, callback=callback):
                while self.running:
                    sd.sleep(100) # Sleep briefly to yield CPU
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not start audio stream: {e}")
            self.running = False


    def transcription_loop(self):
        """Transcribes audio blocks from the queue using Faster Whisper."""
        try:
            # Load Whisper model once. Changed to "base" as per user's request.
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        except Exception as e:
            messagebox.showerror("Whisper Model Error", f"Could not load Whisper model: {e}\nEnsure 'base.en' model files are downloaded or specify a smaller model like 'tiny' or 'small'.")
            self.running = False
            return

        audio_buffer = []
        last_transcription_time = time.time()

        while self.running:
            try:
                block = self.audio_queue.get(timeout=1) # Get audio block, with timeout
                audio_buffer.append(block)

                # Process audio if enough duration has accumulated
                if time.time() - last_transcription_time >= AUDIO_BLOCK_DURATION:
                    combined_audio = np.concatenate(audio_buffer).flatten()

                    # Save to a temporary WAV file for Faster Whisper
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        tmp_path = tmpfile.name
                        sf.write(tmp_path, combined_audio, AUDIO_SAMPLE_RATE)

                    segments, _ = self.whisper_model.transcribe(tmp_path)
                    os.remove(tmp_path) # Clean up temp file

                    for seg in segments:
                        msg = seg.text.strip()
                        if msg:
                            ts_str = timestamp()
                            self.transcriptions.append((ts_str, msg))
                            self.last_transcription = msg # Update for event context
                            # Only update UI if the app is still running
                            if self.running:
                                self.log_area.insert(tk.END, f"[{ts_str}] Speech: {msg}\n")
                                self.log_area.see(tk.END)
                                self.status_transcription.config(text=f"Speech: {msg}")

                    audio_buffer.clear()
                    last_transcription_time = time.time()

            except queue.Empty:
                # No audio in queue, continue loop
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
                # You might want to log this error to the UI or a separate error log
                audio_buffer.clear() # Clear buffer to prevent continuous errors on bad data


if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    root = tk.Tk()
    app = EngageTrackApp(root)
    root.mainloop()
