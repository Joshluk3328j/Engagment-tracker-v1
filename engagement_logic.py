# engagement_logic.py

import time
from collections import deque

# Import constants from config module
from config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES_CLOSED, FATIGUE_BLINK_COUNT, FATIGUE_BLINK_WINDOW_SECONDS, FATIGUE_BLINK_COOLDOWN_SECONDS,
    MAR_THRESHOLD, MAR_CONSEC_FRAMES_OPEN, FATIGUE_YAWN_COUNT, FATIGUE_YAWN_WINDOW_SECONDS, FATIGUE_YAWN_COOLDOWN_SECONDS,
    ATTENTION_YAW_THRESHOLD, PITCH_FOCUSED_MIN_ABS_THRESHOLD, ATTENTION_CONSISTENCY_SECONDS,
    HAND_RAISE_Y_THRESHOLD_FACTOR, HAND_RAISE_COOLDOWN_SECONDS, HAND_MOVEMENT_WINDOW_FRAMES, HAND_MOVEMENT_STD_THRESHOLD, HAND_MOVEMENT_COOLDOWN_SECONDS,
    VIDEO_FPS_TARGET # Needed for HAND_MOVEMENT_WINDOW_FRAMES calculation
)

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
