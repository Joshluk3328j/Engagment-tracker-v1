import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from config import (
    ATTENTION_YAW_THRESHOLD, PITCH_FOCUSED_MIN_ABS_THRESHOLD,
    HAND_RAISE_Y_THRESHOLD_FACTOR, HAND_MOVEMENT_WINDOW_FRAMES,
    HAND_MOVEMENT_STD_THRESHOLD, OUTPUT_DIR, AUDIO_SAMPLE_RATE, AUDIO_BLOCK_DURATION, VIDEO_FPS_TARGET
)
from utils import get_eye_aspect_ratio, get_mouth_aspect_ratio, get_head_pose, save_csv, generate_engagement_graphs, generate_full_report
from engagement_logic import EngagementLogic
import pyaudio
import websocket
import json
import threading
from urllib.parse import urlencode
from datetime import datetime
import os

class EngageTrackVideoProcessor:
    def __init__(self, api_key):
        # Flag to control the main loop and shutdown event for thread coordination
        self.running = True
        self.shutdown_event = threading.Event()
        # Engagement logic for detecting blinks, yawns, and attention states
        self.logic = EngagementLogic(self.log_event_to_console)

        # Initialize MediaPipe FaceMesh for face landmark detection (e.g., eyes, mouth)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,  # Enable detailed eye landmarks
            max_num_faces=1,  # Process only one face
            min_detection_confidence=0.5,  # Confidence threshold for face detection
            min_tracking_confidence=0.5  # Confidence for tracking face landmarks
        )
        # Initialize MediaPipe Hands for hand landmark detection (e.g., wrist)
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2,  # Detect up to two hands for robustness
            min_detection_confidence=0.5,  # Confidence for initial hand detection
            min_tracking_confidence=0.5  # Confidence for tracking hand landmarks
        )

        # Queues to store recent eye aspect ratio (EAR), mouth aspect ratio (MAR), and hand positions
        self.ear_history = deque(maxlen=50)  # For blink detection
        self.mar_history = deque(maxlen=50)  # For yawn detection
        self.hand_y_positions = deque(maxlen=HAND_MOVEMENT_WINDOW_FRAMES)  # For hand movement

        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Could not open webcam.")
            self.running = False
            self.shutdown_event.set()

        # Transcription and event storage
        self.last_transcription = ""  # Latest speech transcript from AssemblyAI
        self.transcriptions = []  # List of all transcriptions with timestamps
        self.events = deque(maxlen=100)  # Engagement events (e.g., blink, hand raise)
        self.event_queue = deque()  # Temporary queue for UI updates
        self.current_frame = None  # Current video frame
        self.frame_lock = threading.Lock()  # Lock for thread-safe frame access
        self.activities = ["N/A", "N/A", "N/A", ""]  # Attention, Fatigue, Hand, Transcription states

        # Audio configuration for AssemblyAI real-time transcription
        self.api_key = api_key
        self.audio_params = {
            "sample_rate": AUDIO_SAMPLE_RATE,  # 16000 Hz
            "format_turns": True,  # Enable turn-based transcription
        }
        self.api_endpoint = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(self.audio_params)}"
        self.frames_per_buffer = 1600  # 100 ms = 1600 samples at 16000 Hz
        self.audio = None
        self.audio_stream = None
        self.ws_app = None
        self.audio_thread = None
        self.recorded_frames = []  # Store audio chunks for WAV output
        self.recording_lock = threading.Lock()  # Lock for thread-safe audio recording

        # Start video, audio, and transcription threads
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.transcription_thread = threading.Thread(target=self.transcription_loop, daemon=True)
        self.video_thread.start()
        self.audio_thread.start()
        self.transcription_thread.start()

    def log_event_to_console(self, event_type, description, timestamp):
        # Format and log engagement events (e.g., blink, yawn, hand raise) with timestamp and speech context
        ts_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
        speech_context = self.last_transcription
        log_message = f"[{ts_str}] {event_type}: {description}"
        if speech_context:
            log_message += f" (Context: \"{speech_context}\")"
        print(log_message)
        event = (ts_str, event_type, description, speech_context)
        self.events.append(event)
        with self.frame_lock:
            self.event_queue.append(event)

    def video_loop(self):
        # Continuously process video frames at target FPS
        while self.running and not self.shutdown_event.is_set():
            frame = self.process_frame()
            if frame is None:
                continue
            with self.frame_lock:
                self.current_frame = frame
            time.sleep(1 / VIDEO_FPS_TARGET)  # Control frame rate (e.g., 15 FPS)

    def audio_loop(self):
        # Capture and stream audio to AssemblyAI in 100 ms chunks
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                input=True,
                frames_per_buffer=self.frames_per_buffer,  # 1600 samples = 100 ms
                channels=1,  # Mono audio
                format=pyaudio.paInt16,  # 16-bit audio format
                rate=AUDIO_SAMPLE_RATE,  # 16000 Hz
                input_device_index=None,  # Default microphone
            )
            print(f"Microphone stream opened. Buffer size: {self.frames_per_buffer} samples ({self.frames_per_buffer / AUDIO_SAMPLE_RATE * 1000:.1f} ms)")
            while self.running and not self.shutdown_event.is_set():
                try:
                    start_time = time.time()
                    audio_data = self.audio_stream.read(self.frames_per_buffer, exception_on_overflow=False)
                    duration_ms = len(audio_data) / AUDIO_SAMPLE_RATE * 1000
                    # print(f"Read audio chunk: {len(audio_data)} bytes, {duration_ms:.1f} ms")
                    if len(audio_data) < self.frames_per_buffer * 2:  # Ensure chunk is at least 50 ms
                        print(f"Warning: Short audio chunk ({duration_ms:.1f} ms < 50 ms)")
                        continue
                    with self.recording_lock:
                        self.recorded_frames.append(audio_data)
                    if self.ws_app and self.ws_app.sock and self.ws_app.sock.connected:
                        self.ws_app.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                        # print(f"Sent audio chunk: {duration_ms:.1f} ms")
                    elapsed = time.time() - start_time
                    sleep_time = max(0, 0.1 - elapsed)  # Maintain 100 ms cycle
                    time.sleep(sleep_time)
                except Exception as e:
                    print(f"Audio streaming error: {e}")
                    break
        except Exception as e:
            print(f"Audio Error: Could not start audio stream: {e}")
            self.running = False
            self.shutdown_event.set()

    def transcription_loop(self):
        # Handle WebSocket connection for AssemblyAI transcription
        def on_open(ws):
            print("WebSocket connection opened for AssemblyAI.")

        def on_message(ws, message):
            # Process transcription messages from AssemblyAI
            try:
                data = json.loads(message)
                msg_type = data.get('type')
                if msg_type == "Turn":
                    transcript = data.get('transcript', '').strip()
                    formatted = data.get('turn_is_formatted', False)
                    if transcript and formatted:
                        ts_str = datetime.now().strftime("%H:%M:%S")
                        self.transcriptions.append((ts_str, transcript))
                        self.last_transcription = transcript
                        print(f"[{ts_str}] Speech: {transcript}")
                        with self.frame_lock:
                            self.activities[3] = transcript
                elif msg_type == "Termination":
                    print(f"Session terminated by server: {data}")
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
            except Exception as e:
                print(f"Error handling message: {e}")

        def on_error(ws, error):
            print(f"WebSocket Error: {error}")
            self.shutdown_event.set()

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
            self.shutdown_event.set()

        try:
            self.ws_app = websocket.WebSocketApp(
                self.api_endpoint,
                header={"Authorization": self.api_key},
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            self.ws_app.run_forever()
        except Exception as e:
            print(f"WebSocket initialization error: {e}")
            self.running = False
            self.shutdown_event.set()

    def process_frame(self):
        # Process a single video frame for face and hand detection
        if not self.running or self.shutdown_event.is_set() or self.cap is None:
            return None

        # Capture and preprocess frame
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            return None

        frame = cv2.flip(frame, 1)  # Flip horizontally for natural webcam view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        h, w, _ = frame.shape  # Get frame dimensions (e.g., 1280x720)
        # print(f"Frame dimensions: {w}x{h}")

        # Process frame with MediaPipe FaceMesh and Hands
        face_results = self.mp_face_mesh.process(rgb_frame)  # Detect face landmarks
        hand_results = self.mp_hands.process(rgb_frame)  # Detect hand landmarks

        # Initialize states for attention, fatigue, and hand activity
        self.attention_state = "N/A"
        self.fatigue_state = "N/A"
        self.hand_state = "No Hand Detected"

        if face_results.multi_face_landmarks:
            # Process face landmarks for attention and fatigue detection
            lm = face_results.multi_face_landmarks[0].landmark
            # Convert normalized landmarks (0.0-1.0) to pixel coordinates
            coords = lambda idxs: [(int(lm[i].x * w), int(lm[i].y * h)) for i in idxs]

            # Define landmark indices for eyes and mouth
            left_eye_indices = [362, 385, 387, 263, 373, 380]  # Left eye landmarks
            right_eye_indices = [33, 160, 158, 133, 153, 144]  # Right eye landmarks
            mouth_indices = [61, 81, 13, 311, 402, 14]  # Mouth landmarks

            # Calculate eye and mouth coordinates for EAR and MAR
            left_eye_coords = coords(left_eye_indices)
            right_eye_coords = coords(right_eye_indices)
            mouth_coords = coords(mouth_indices)

            # Compute Eye Aspect Ratio (EAR) for blink detection
            ear = (get_eye_aspect_ratio(left_eye_coords) + get_eye_aspect_ratio(right_eye_coords)) / 2
            # Compute Mouth Aspect Ratio (MAR) for yawn detection
            mar = get_mouth_aspect_ratio(mouth_coords)

            self.ear_history.append(ear)
            self.mar_history.append(mar)

            # Detect blinks and yawns using engagement logic
            self.logic.detect_and_register_blink(ear)
            self.logic.detect_and_register_yawn(mar)

            # Calculate head pose for attention detection
            pitch, yaw, roll = get_head_pose(lm, frame.shape)
            # Determine if user is focused based on yaw and pitch thresholds
            focused = (abs(yaw) <= ATTENTION_YAW_THRESHOLD) and (abs(pitch) >= PITCH_FOCUSED_MIN_ABS_THRESHOLD)

            self.attention_state = "Focused" if focused else "Distracted"
            self.logic.update_attention(focused, pitch, yaw)

            # Update fatigue state based on blink/yawn detection
            if self.logic._is_eye_closed or self.logic._is_mouth_open:
                self.fatigue_state = "Potential Fatigue"
            elif self.logic.blink_cooldown_end_time > self.logic._now() or \
                 self.logic.yawn_cooldown_end_time > self.logic._now():
                self.fatigue_state = "Fatigue Detected"
            else:
                self.fatigue_state = "Normal"
        else:
            # Reset states and histories if no face is detected
            self.attention_state = "No Face Detected"
            self.fatigue_state = "N/A"
            self.logic.update_attention(False, 0, 0)
            self.ear_history.clear()
            self.mar_history.clear()
            self.hand_y_positions.clear()

        is_hand_raised = False
        current_hand_std = 0

        if hand_results.multi_hand_landmarks:
            # Process each detected hand for hand raise or presence
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Log number of hands detected for debugging
                # print(f"Hands detected: {len(hand_results.multi_hand_landmarks)}")

                # Get wrist y-coordinate (landmark 0) in normalized (0.0-1.0) and pixel units
                wrist_y = hand_landmarks.landmark[0].y
                wrist_y_pixel = wrist_y * h
                if face_results.multi_face_landmarks:
                    # Average y-coordinates of left (33) and right (263) eye landmarks
                    eye_y = (lm[33].y + lm[263].y) / 2
                    # eye_y_pixel = eye_y * h
                else:
                    # Fallback if no face detected: assume eye level at mid-frame
                    eye_y = 0.5
                    # eye_y_pixel = 0.5 * h

                # # Log wrist and eye positions for debugging
                # print(f"Wrist Y: {wrist_y:.3f} ({wrist_y_pixel:.1f}px), Eye Y: {eye_y:.3f} ({eye_y_pixel:.1f}px), Threshold: {eye_y * HAND_RAISE_Y_THRESHOLD_FACTOR:.3f}")

                # Check if wrist is above threshold (e.g., at or above eye level with factor=1.0)
                if wrist_y < eye_y * HAND_RAISE_Y_THRESHOLD_FACTOR:
                    is_hand_raised = True
                    self.hand_state = "Hand Raised"
                    print("Hand Raised Detected")
                else:
                    # Set "Hand Detected" if hand is present but not raised
                    self.hand_state = "Hand Detected"
                    print("Hand Detected (not raised)")

                # Track wrist y-positions for optional movement analysis
                self.hand_y_positions.append(wrist_y)
                if len(self.hand_y_positions) == HAND_MOVEMENT_WINDOW_FRAMES:
                    # Calculate standard deviation of wrist positions for movement logging
                    current_hand_std = np.std(list(self.hand_y_positions))
                    print(f"Hand movement STD: {current_hand_std:.4f} (threshold: {HAND_MOVEMENT_STD_THRESHOLD})")

            self.logic.register_hand_event(is_hand_raised, current_hand_std)
        else:
            # print("No hand landmarks detected")
            pass

        # Update activities for UI display
        with self.frame_lock:
            self.activities[0] = self.attention_state
            self.activities[1] = self.fatigue_state
            self.activities[2] = self.hand_state

        # print(f"Attention: {self.attention_state} | Fatigue: {self.fatigue_state} | Hand: {self.hand_state} | Transcription: {self.last_transcription}")

        return frame


    def get_current_frame(self):
        # Return the current frame for Flask video feed
        with self.frame_lock:
            return self.current_frame

    def get_activities(self):
        # Return current activity states for UI updates
        with self.frame_lock:
            # print(f"Activities sent to UI: {self.activities}")
            return self.activities[:]

    def get_new_events(self):
        # Return new engagement events for UI logging
        with self.frame_lock:
            events = list(self.event_queue)
            self.event_queue.clear()
            return events

    def end_session(self):
        # End the session and save outputs
        self.running = False
        self.shutdown_event.set()

        # Send termination message to AssemblyAI
        if self.ws_app and self.ws_app.sock and self.ws_app.sock.connected:
            try:
                self.ws_app.send(json.dumps({"type": "Terminate"}))
                time.sleep(0.5)
            except Exception as e:
                print(f"Error sending termination message: {e}")

        # Close WebSocket
        if self.ws_app:
            self.ws_app.close()

        # Wait for threads to terminate
        for thread in [self.video_thread, self.audio_thread, self.transcription_thread]:
            if thread.is_alive():
                print(f"Waiting for {thread.name} to terminate")
                thread.join(timeout=2.0)


        # Save transcription and engagement logs to CSV
        save_csv(os.path.join(OUTPUT_DIR, "transcription.csv"),
                 self.transcriptions, ["Timestamp", "Transcription"])
        save_csv(os.path.join(OUTPUT_DIR, "engagement_log.csv"),
                 self.events, ["Timestamp", "EventType", "Description", "SpeechContext"])

        # Generate engagement graphs and PDF report
        generate_engagement_graphs(self.events)
        generate_full_report(self.transcriptions, self.events, OUTPUT_DIR)

    def release(self):
        # Clean up resources (webcam, audio, MediaPipe, WebSocket)
        self.running = False
        self.shutdown_event.set()

        # Close WebSocket
        if self.ws_app and self.ws_app.sock and self.ws_app.sock.connected:
            try:
                self.ws_app.send(json.dumps({"type": "Terminate"}))
                time.sleep(0.5)
            except Exception as e:
                print(f"Error sending termination message: {e}")
            self.ws_app.close()

        # Wait for threads to terminate
        for thread in [self.video_thread, self.audio_thread, self.transcription_thread]:
            if thread.is_alive():
                print(f"Waiting for {thread.name} to terminate")
                thread.join(timeout=2.0)

        # Close audio resources
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        if self.audio_stream:
            self.audio_stream.close()
        if self.audio:
            self.audio.terminate()

        # Release webcam and MediaPipe
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if hasattr(self, 'mp_face_mesh'):
            self.mp_face_mesh.close()
        if hasattr(self, 'mp_hands'):
            self.mp_hands.close()
