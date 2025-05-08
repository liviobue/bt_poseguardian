import mediapipe as mp
import numpy as np
import cv2
import math
import time
import threading


class HandGesture:
    """Base class for hand gesture recognition"""
    
    def __init__(self):
        self.name = "Base Gesture"
    
    def recognize(self, hand_landmarks):
        """
        Check if the hand landmarks match this gesture
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            bool: True if gesture is recognized, False otherwise
        """
        return False
    
    def get_name(self):
        """Return the name of this gesture"""
        return self.name


class OpenHandGesture(HandGesture):
    """Gesture for fully open hand with all fingers extended and properly spaced"""
    
    def __init__(self):
        self.name = "Open Hand"
        self.math = math
        # Add a very small angle tolerance (in degrees)
        self.angle_tolerance = 2.0  # Very small tolerance of just 2 degrees
    
    def recognize(self, hand_landmarks):
        """Checks if all fingers are fully extended with appropriate separation between them."""
        
        # Landmark indices
        FINGERTIPS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        MIDDLE_JOINTS = [3, 7, 11, 15, 19] # PIP joints (middle of fingers)
        BASE_JOINTS = [2, 6, 10, 14, 18]   # MCP joints (base of fingers)
        WRIST = 0  # Wrist landmark

        # Determine if it's a left or right hand
        thumb_tip_x = hand_landmarks.landmark[4].x
        pinky_tip_x = hand_landmarks.landmark[20].x
        is_right_hand = thumb_tip_x < pinky_tip_x
        
        # Calculate hand scale using distance from wrist to middle finger MCP
        wrist_point = [
            hand_landmarks.landmark[WRIST].x,
            hand_landmarks.landmark[WRIST].y,
            hand_landmarks.landmark[WRIST].z
        ]
        middle_mcp_point = [
            hand_landmarks.landmark[9].x,
            hand_landmarks.landmark[9].y,
            hand_landmarks.landmark[9].z
        ]
        hand_scale = self.calculate_3d_distance(wrist_point, middle_mcp_point)
        
        # Calculate segment lengths for all fingers (excluding thumb)
        finger_segment_lengths = []
        for i in range(1, 5):  # Skip thumb, index=1, middle=2, ring=3, pinky=4
            tip = FINGERTIPS[i]
            middle = MIDDLE_JOINTS[i]
            base = BASE_JOINTS[i]
            
            # Get coordinates
            tip_coords = [
                hand_landmarks.landmark[tip].x,
                hand_landmarks.landmark[tip].y,
                hand_landmarks.landmark[tip].z
            ]
            middle_coords = [
                hand_landmarks.landmark[middle].x,
                hand_landmarks.landmark[middle].y,
                hand_landmarks.landmark[middle].z
            ]
            base_coords = [
                hand_landmarks.landmark[base].x,
                hand_landmarks.landmark[base].y,
                hand_landmarks.landmark[base].z
            ]
            
            # Calculate segment lengths
            tip_to_middle = self.calculate_3d_distance(tip_coords, middle_coords)
            middle_to_base = self.calculate_3d_distance(middle_coords, base_coords)
            
            finger_segment_lengths.append((tip_to_middle, middle_to_base))
        
        # Calculate average segment lengths across all fingers
        avg_tip_to_middle = sum([lengths[0] for lengths in finger_segment_lengths]) / len(finger_segment_lengths)
        avg_middle_to_base = sum([lengths[1] for lengths in finger_segment_lengths]) / len(finger_segment_lengths)
        
        # Calculate average finger length (excluding thumb) for reference
        finger_lengths = []
        for i in range(1, 5):  # Skip thumb (index 0)
            tip = FINGERTIPS[i]
            base = BASE_JOINTS[i]
            tip_coords = [
                hand_landmarks.landmark[tip].x,
                hand_landmarks.landmark[tip].y,
                hand_landmarks.landmark[tip].z
            ]
            base_coords = [
                hand_landmarks.landmark[base].x,
                hand_landmarks.landmark[base].y,
                hand_landmarks.landmark[base].z
            ]
            finger_lengths.append(self.calculate_3d_distance(tip_coords, base_coords))
        avg_finger_length = sum(finger_lengths) / len(finger_lengths) if finger_lengths else hand_scale
        
        # Dynamic thresholds based on hand size
        min_segment_distance = 0.15 * avg_finger_length  # Minimum distance between finger segments
        min_straightness_ratio = 0.85  # Minimum ratio of direct distance to sum of segment distances
        
        # NEW: Maximum allowed deviation in segment length compared to average (as percentage)
        max_segment_length_deviation = 0.25  # 25% deviation allowed
        
        # Check each finger is extended
        for i, (tip, middle, base) in enumerate(zip(FINGERTIPS, MIDDLE_JOINTS, BASE_JOINTS)):
            # Get 3D coordinates for each joint
            tip_coords = [
                hand_landmarks.landmark[tip].x, 
                hand_landmarks.landmark[tip].y, 
                hand_landmarks.landmark[tip].z
            ]
            middle_coords = [
                hand_landmarks.landmark[middle].x, 
                hand_landmarks.landmark[middle].y, 
                hand_landmarks.landmark[middle].z
            ]
            base_coords = [
                hand_landmarks.landmark[base].x, 
                hand_landmarks.landmark[base].y, 
                hand_landmarks.landmark[base].z
            ]
            
            # Calculate distances between joints
            tip_to_middle = self.calculate_3d_distance(tip_coords, middle_coords)
            middle_to_base = self.calculate_3d_distance(middle_coords, base_coords)
            tip_to_base = self.calculate_3d_distance(tip_coords, base_coords)
            
            # Special handling for thumb
            if i == 0:  # Thumb
                # For thumb, check extension based on hand orientation
                if is_right_hand:
                    if not (hand_landmarks.landmark[tip].x < hand_landmarks.landmark[middle].x < hand_landmarks.landmark[base].x):
                        return False
                else:
                    if not (hand_landmarks.landmark[tip].x > hand_landmarks.landmark[middle].x > hand_landmarks.landmark[base].x):
                        return False
                
                # Check distances for thumb extension
                if tip_to_middle < 0.7 * min_segment_distance or middle_to_base < 0.7 * min_segment_distance:
                    return False
            else:
                # For other fingers
                # 1. Check if finger is extended upward (y coordinate decreasing)
                if not (hand_landmarks.landmark[tip].y < hand_landmarks.landmark[middle].y < hand_landmarks.landmark[base].y):
                    return False
                
                # 2. Check distances to ensure finger is straight and extended
                if tip_to_middle < min_segment_distance or middle_to_base < min_segment_distance:
                    return False
                
                # 3. Check the ratio of direct distance to segment distances
                # This detects when finger is bent toward camera (segments get shorter)
                straightness_ratio = tip_to_base / (tip_to_middle + middle_to_base)
                if straightness_ratio < min_straightness_ratio:
                    return False
                
                # 4. Check if segments are too short compared to average finger length
                if (tip_to_middle + middle_to_base) < 0.6 * avg_finger_length:
                    return False
                
                # 5. Check if the finger is bending toward camera by comparing z-coordinates
                # When finger bends toward camera, the tip z-coordinate decreases significantly
                z_diff = abs(tip_coords[2] - base_coords[2])
                if z_diff > 0.2 * avg_finger_length:  # Large z-difference indicates bending
                    return False
                
                # NEW: Check if segment lengths deviate too much from average across all fingers
                # Skip thumb (i=0) since it has different proportions
                finger_idx = i - 1  # Adjust index for finger_segment_lengths array
                
                # Compare this finger's segments to the average of all fingers
                tip_middle_deviation = abs(tip_to_middle - avg_tip_to_middle) / avg_tip_to_middle
                middle_base_deviation = abs(middle_to_base - avg_middle_to_base) / avg_middle_to_base
                
                if tip_middle_deviation > max_segment_length_deviation or middle_base_deviation > max_segment_length_deviation:
                    # This finger's segments are too different from others - likely bent toward camera
                    return False
                
                # NEW: Check for absolute shortening compared to expected length
                expected_segment_ratio = 0.48  # Approx ratio of segment to full length in extended finger
                if tip_to_middle < expected_segment_ratio * avg_finger_length * 0.8:  # 20% tolerance
                    # Segment is too short - likely bent toward camera
                    return False
        
        # Now check finger spacing as in the original code
        # Check the separation between each pair of fingertips
        min_thumb_index_distance = 0.12 * hand_scale
        min_index_middle_distance = 0.09 * hand_scale  # Slightly more permissive
        min_middle_ring_distance = 0.09 * hand_scale
        min_ring_pinky_distance = 0.10 * hand_scale
        
        # Get all fingertips for easier access
        fingertips = [hand_landmarks.landmark[i] for i in FINGERTIPS]
        finger_bases = [hand_landmarks.landmark[i] for i in BASE_JOINTS]
        
        # Check distance between thumb and index
        thumb_tip = fingertips[0]
        index_tip = fingertips[1]
        distance = self.calculate_3d_distance(
            [thumb_tip.x, thumb_tip.y, thumb_tip.z],
            [index_tip.x, index_tip.y, index_tip.z]
        )
        
        if distance < min_thumb_index_distance:
            return False
            
        # Calculate and check angles between fingers
        # Thumb and index angle check
        thumb_base = finger_bases[0]
        index_base = finger_bases[1]
        
        # Create vectors from base to tip
        thumb_vector = [
            thumb_tip.x - thumb_base.x,
            thumb_tip.y - thumb_base.y,
            thumb_tip.z - thumb_base.z
        ]
        
        index_vector = [
            index_tip.x - index_base.x,
            index_tip.y - index_base.y,
            index_tip.z - index_base.z
        ]
        
        angle = self.calculate_angle(thumb_vector, index_vector)
        min_thumb_index_angle = 30 - self.angle_tolerance  # Very slightly more permissive with tolerance
        if angle < min_thumb_index_angle:
            return False
        
        # Check distances and angles between other finger pairs
        min_finger_angle = 8 - self.angle_tolerance  # Very slightly more permissive with tolerance
        
        # Check distance and angle between index and middle
        middle_tip = fingertips[2]
        middle_base = finger_bases[2]
        
        # Distance check
        distance = self.calculate_3d_distance(
            [index_tip.x, index_tip.y, index_tip.z],
            [middle_tip.x, middle_tip.y, middle_tip.z]
        )
        
        if distance < min_index_middle_distance:
            return False
        
        # Angle check
        middle_vector = [
            middle_tip.x - middle_base.x,
            middle_tip.y - middle_base.y,
            middle_tip.z - middle_base.z
        ]
        
        angle = self.calculate_angle(index_vector, middle_vector)
        if angle < min_finger_angle:
            return False
        
        # Check distance and angle between middle and ring
        ring_tip = fingertips[3]
        ring_base = finger_bases[3]
        
        # Distance check
        distance = self.calculate_3d_distance(
            [middle_tip.x, middle_tip.y, middle_tip.z],
            [ring_tip.x, ring_tip.y, ring_tip.z]
        )
        
        if distance < min_middle_ring_distance:
            return False
        
        # Angle check
        ring_vector = [
            ring_tip.x - ring_base.x,
            ring_tip.y - ring_base.y,
            ring_tip.z - ring_base.z
        ]
        
        angle = self.calculate_angle(middle_vector, ring_vector)
        if angle < min_finger_angle:
            return False
        
        # Check distance and angle between ring and pinky
        pinky_tip = fingertips[4]
        pinky_base = finger_bases[4]
        
        # Distance check
        distance = self.calculate_3d_distance(
            [ring_tip.x, ring_tip.y, ring_tip.z],
            [pinky_tip.x, pinky_tip.y, pinky_tip.z]
        )
        
        if distance < min_ring_pinky_distance:
            return False
        
        # Angle check
        pinky_vector = [
            pinky_tip.x - pinky_base.x,
            pinky_tip.y - pinky_base.y,
            pinky_tip.z - pinky_base.z
        ]
        
        angle = self.calculate_angle(ring_vector, pinky_vector)
        if angle < min_finger_angle:
            return False
        
        # If all checks pass, it's a proper open hand
        return True
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return ((point1[0] - point2[0])**2 + 
                (point1[1] - point2[1])**2 + 
                (point1[2] - point2[2])**2)**0.5
    
    def calculate_angle(self, vector1, vector2):
        """Calculate angle between two vectors in degrees"""
        # Calculate the dot product
        dot_product = (vector1[0] * vector2[0] + 
                       vector1[1] * vector2[1] + 
                       vector1[2] * vector2[2])
        
        # Calculate the magnitudes
        mag1 = (vector1[0]**2 + vector1[1]**2 + vector1[2]**2)**0.5
        mag2 = (vector2[0]**2 + vector2[1]**2 + vector2[2]**2)**0.5
        
        # Avoid division by zero
        if mag1 * mag2 == 0:
            return 0
            
        cos_angle = dot_product / (mag1 * mag2)
        
        # Ensure the cosine is within valid range (-1 to 1)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        # Convert to angle in degrees
        return self.math.degrees(self.math.acos(cos_angle))


class CylindricalGraspGesture(HandGesture):
    """Gesture for cylindrical grasp (all fingers curled as if holding a cylinder)"""
    
    def __init__(self):
        self.name = "Cylindrical Grasp"
        self.debug = True  # Set to False to disable debug prints
        
    def recognize(self, hand_landmarks):
        """Checks if all fingers are curled in a cylindrical grasp pattern."""
        
        # Landmark indices
        FINGERTIPS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        MIDDLE_JOINTS = [3, 7, 11, 15, 19] # PIP joints
        BASE_JOINTS = [2, 6, 10, 14, 18]   # MCP joints
        WRIST = 0  # Wrist landmark
        
        # Calculate hand scale
        wrist_point = [
            hand_landmarks.landmark[WRIST].x,
            hand_landmarks.landmark[WRIST].y,
            hand_landmarks.landmark[WRIST].z
        ]
        middle_mcp_point = [
            hand_landmarks.landmark[9].x,
            hand_landmarks.landmark[9].y,
            hand_landmarks.landmark[9].z
        ]
        hand_scale = self.calculate_3d_distance(wrist_point, middle_mcp_point)
        
        # Determine hand orientation
        thumb_tip_x = hand_landmarks.landmark[4].x
        pinky_tip_x = hand_landmarks.landmark[20].x
        is_right_hand = thumb_tip_x < pinky_tip_x
        
        # Debug: Show hand scale
        if self.debug:
            print(f"\nHand scale: {hand_scale:.3f}")
        
        # Check finger curl - now more lenient
        for i, (tip, middle, base) in enumerate(zip(FINGERTIPS[1:], MIDDLE_JOINTS[1:], BASE_JOINTS[1:])):
            # For index, middle, ring, and pinky fingers
            tip_y = hand_landmarks.landmark[tip].y
            middle_y = hand_landmarks.landmark[middle].y
            
            # More lenient: fingertip should be at least slightly below middle joint
            if tip_y < middle_y - 0.02:  # Was just tip_y < middle_y
                if self.debug:
                    print(f"Finger {i+1} not curled enough (tip_y: {tip_y:.3f} < middle_y: {middle_y:.3f})")
                return False
            
            # More lenient palm proximity check
            tip_to_wrist = self.calculate_3d_distance(
                [hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z],
                wrist_point
            )
            base_to_wrist = self.calculate_3d_distance(
                [hand_landmarks.landmark[base].x, hand_landmarks.landmark[base].y, hand_landmarks.landmark[base].z],
                wrist_point
            )
            
            if tip_to_wrist > base_to_wrist * 1.1:  # Was just tip_to_wrist > base_to_wrist
                if self.debug:
                    print(f"Finger {i+1} not close enough to palm (tip_to_wrist: {tip_to_wrist:.3f} > base_to_wrist: {base_to_wrist:.3f})")
                return False
        
        # Thumb position check - more intelligent calculation
        thumb_tip = [
            hand_landmarks.landmark[4].x,
            hand_landmarks.landmark[4].y,
            hand_landmarks.landmark[4].z
        ]

        # Use index finger MCP (base) as reference point
        index_base = [
            hand_landmarks.landmark[5].x,
            hand_landmarks.landmark[5].y,
            hand_landmarks.landmark[5].z
        ]

        thumb_to_index_dist = self.calculate_3d_distance(thumb_tip, index_base)

        # Dynamic threshold based on hand size and finger length
        middle_tip = [
            hand_landmarks.landmark[12].x,
            hand_landmarks.landmark[12].y,
            hand_landmarks.landmark[12].z
        ]
        middle_finger_length = self.calculate_3d_distance(index_base, middle_tip)

        # New formula: base distance + percentage of middle finger length
        max_thumb_dist = (0.2 * hand_scale) + (0.4 * middle_finger_length)

        if thumb_to_index_dist > max_thumb_dist:
            if self.debug:
                print(f"Thumb distance: {thumb_to_index_dist:.3f} (allowed: {max_thumb_dist:.3f})")
                print(f"Hand scale: {hand_scale:.3f}, Middle finger: {middle_finger_length:.3f}")
            return False
        
        # Thumb opposition check
        thumb_x = hand_landmarks.landmark[4].x
        index_base_x = hand_landmarks.landmark[5].x
        
        if is_right_hand:
            if thumb_x > index_base_x + 0.05:  # Added small tolerance
                if self.debug:
                    print(f"Right hand thumb not opposed (thumb_x: {thumb_x:.3f} > index_base_x: {index_base_x:.3f})")
                return False
        else:
            if thumb_x < index_base_x - 0.05:  # Added small tolerance
                if self.debug:
                    print(f"Left hand thumb not opposed (thumb_x: {thumb_x:.3f} < index_base_x: {index_base_x:.3f})")
                return False
        
        if self.debug:
            print("Cylindrical grasp detected!")
        return True
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return ((point1[0] - point2[0])**2 + 
                (point1[1] - point2[1])**2 + 
                (point1[2] - point2[2])**2)**0.5


class ThumbTouchAllFingersGesture(HandGesture):
    """Gesture for thumb touching all other fingers in sequence with adjustable timing"""
    
    def __init__(self):
        self.name = "Thumb Touch All"
        self.finger_touched = [False, False, False, False]  # Index, Middle, Ring, Pinky
        self.current_finger_idx = 0  # Track which finger should be touched next
        self.last_touch_time = 0
        self.display_duration = 3.0  # seconds to display the recognition message
        self.sequence_timeout = 8.0  # Increased total sequence timeout (was 5.0)
        self.sequence_start_time = None
        self.completed = False
        self.completion_time = 0
        self.in_progress = False
        self.tolerance = 0.07  # Distance tolerance for "touching" detection
        
        # Configurable timing parameters (in seconds)
        self.min_time_between_touches = 0.15  # Minimum time between touches (150ms)
        self.max_time_between_touches = 1.5   # Increased maximum time between touches (was 0.5)
        self.min_touch_duration = 0.1         # Minimum time thumb must stay on finger
        self.cooldown_between_sequences = 1.5 # Increased cooldown (was 1.0)
        
        # Touch state tracking
        self.current_touch_start_time = 0
        self.is_currently_touching = False
        self.last_touch_end_time = 0
        self.last_finger_touched_time = 0  # Time when last finger was properly touched

    def is_thumb_touching_finger(self, hand_landmarks, finger_tip_idx):
        """Check if thumb tip is touching another fingertip with visual tolerance"""
        thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
        finger_tip = hand_landmarks.landmark[finger_tip_idx]
        
        # Calculate 3D distance between thumb tip and fingertip
        distance = ((thumb_tip.x - finger_tip.x)**2 + 
                   (thumb_tip.y - finger_tip.y)**2 + 
                   (thumb_tip.z - finger_tip.z)**2)**0.5
        
        # Check against the tolerance threshold
        return distance < self.tolerance
    
    def recognize(self, hand_landmarks, current_time=None):
        """
        Check if the thumb has touched all fingers in sequence with flexible timing
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            current_time: Current timestamp for timeout tracking
            
        Returns:
            bool: True if gesture is recognized or in progress, False otherwise
        """
        if current_time is None:
            current_time = time.time()
            
        # If gesture was already completed and we're in display period
        if self.completed:
            if current_time - self.completion_time < self.display_duration:
                return True
            else:
                self.reset()
                return False
        
        # Finger tip landmark indices
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        
        # Check current touch state
        currently_touching = False
        current_touch_index = None
        
        # Determine which finger is being touched (if any)
        for i, tip_idx in enumerate(finger_tips):
            if self.is_thumb_touching_finger(hand_landmarks, tip_idx):
                currently_touching = True
                current_touch_index = i
                break
        
        # Handle new touch detection
        if currently_touching and not self.is_currently_touching:
            self.current_touch_start_time = current_time
            self.is_currently_touching = True
            
            # If this is the first touch, start the sequence
            if not self.in_progress and current_touch_index == 0:
                # Only start new sequence if enough time has passed since last sequence
                if current_time - self.last_touch_end_time > self.cooldown_between_sequences:
                    self.sequence_start_time = current_time
                    self.finger_touched[0] = True
                    self.current_finger_idx = 1
                    self.in_progress = True
                    self.last_finger_touched_time = current_time
                    return True
        
        # Handle touch release
        if self.is_currently_touching and not currently_touching:
            touch_duration = current_time - self.current_touch_start_time
            self.last_touch_end_time = current_time
            self.is_currently_touching = False
            
            # Validate touch duration was sufficient
            if touch_duration < self.min_touch_duration:
                if self.in_progress:
                    self.reset()
                return False
        
        # Check for timeout if sequence has started
        if self.sequence_start_time is not None:
            if current_time - self.sequence_start_time > self.sequence_timeout:
                self.reset()
                return False
        
        # Process in-progress sequence
        if self.in_progress and self.current_finger_idx < 4:
            # Check if current finger is being touched
            if currently_touching and current_touch_index == self.current_finger_idx:
                time_since_last_touch = current_time - self.last_finger_touched_time
                
                # Enforce minimum time between touches
                if time_since_last_touch < self.min_time_between_touches:
                    return True  # Waiting for proper timing
                
                # Record the valid touch
                self.finger_touched[self.current_finger_idx] = True
                self.current_finger_idx += 1
                self.last_finger_touched_time = current_time
                
                # Check for sequence completion
                if self.current_finger_idx >= 4:
                    self.completed = True
                    self.completion_time = current_time
                    self.in_progress = False
                    return True
            
            # Check for out-of-sequence touches
            if currently_touching and current_touch_index > self.current_finger_idx:
                self.reset()
                return False
            
            # Check maximum time between touches
            if current_time - self.last_finger_touched_time > self.max_time_between_touches:
                self.reset()
                return False
            
            # Still in progress
            return True
                        
        return False
    
    def reset(self):
        """Reset the gesture tracking state"""
        self.finger_touched = [False, False, False, False]
        self.current_finger_idx = 0
        self.sequence_start_time = None
        self.completed = False
        self.in_progress = False
        self.is_currently_touching = False
        self.last_touch_end_time = 0
        self.current_touch_start_time = 0
        self.last_finger_touched_time = 0

    def get_progress(self):
        """Return the progress of the gesture sequence"""
        touched_count = sum(1 for touched in self.finger_touched if touched)
        time_left = max(0, self.sequence_timeout - (time.time() - (self.sequence_start_time or time.time())))
        return f"{touched_count}/4 fingers (time left: {time_left:.1f}s)"
        
    def is_in_progress(self):
        """Return whether the gesture is currently in progress"""
        return self.in_progress or self.completed
    
    def set_timing_parameters(self, min_between=0.2, max_between=2.5, min_duration=0.1, sequence_timeout=8.0, cooldown=2.0):
        """Adjust the timing parameters for more/less strict recognition"""
        self.min_time_between_touches = min_between
        self.max_time_between_touches = max_between
        self.min_touch_duration = min_duration
        self.sequence_timeout = sequence_timeout
        self.cooldown_between_sequences = cooldown


class GestureRecognizer:
    """Main class for recognizing multiple hand gestures"""
    
    def __init__(self):
        self.gestures = []
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        
        # For tracking recognition messages
        self.recognized_gestures = []
        self.gesture_display_times = {}
        self.last_completed_gestures = set()  # Track which gestures were completed
        self.active_gesture = None  # Track the currently active gesture (if any)
        self.holding_object = False  # Track if the user appears to be holding an object
        
        # Register default gestures in order of priority
        self.register_gesture(CylindricalGraspGesture())  # Highest priority
        self.register_gesture(OpenHandGesture())
        self.register_gesture(ThumbTouchAllFingersGesture())  # Lowest priority
    
    def register_gesture(self, gesture):
        """Add a new gesture to the recognizer"""
        self.gestures.append(gesture)
        
    def get_gesture_by_name(self, name):
        """Get a gesture object by its name"""
        for gesture in self.gestures:
            if gesture.get_name() == name:
                return gesture
        return None
    
    def recognize_gesture(self, hand_landmarks, current_time=None):
        """
        Check which registered gesture matches the given hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            current_time: Current timestamp
            
        Returns:
            tuple: (recognized: bool, gesture_name: str, progress: str, is_running: bool)
        """

        if current_time is None:
            current_time = time.time()
        
        # First, check if we have an active gesture in progress
        if self.active_gesture is not None:
            gesture = self.get_gesture_by_name(self.active_gesture)
            
            if gesture is not None and hasattr(gesture, 'recognize') and callable(getattr(gesture, 'recognize')):
                # Pass current time to gesture recognizer for timing-based gestures
                import inspect
                params = inspect.signature(gesture.recognize).parameters
                
                if len(params) > 1:
                    recognized = gesture.recognize(hand_landmarks, current_time)
                else:
                    recognized = gesture.recognize(hand_landmarks)
                
                # If still recognized or in progress
                if recognized:
                    # Get progress information if available
                    progress = ""
                    if hasattr(gesture, 'get_progress') and callable(getattr(gesture, 'get_progress')):
                        progress = gesture.get_progress()
                    
                    # Check if it's still in progress or completed
                    is_running = True
                    if hasattr(gesture, 'is_in_progress') and callable(getattr(gesture, 'is_in_progress')):
                        is_running = gesture.is_in_progress()
                    
                    # If completed or no longer in progress, clear active gesture
                    if not is_running and hasattr(gesture, 'completed') and not gesture.completed:
                        self.active_gesture = None
                    
                    return True, gesture.get_name(), progress, is_running
                else:
                    # No longer recognized, clear active gesture
                    self.active_gesture = None
        
        # Check for cylindrical grasp first (higher priority)
        # This is intentionally kept separate from the loop below to give it highest priority
        cylindrical_gesture = self.get_gesture_by_name("Cylindrical Grasp")
        if cylindrical_gesture:
            cylindrical_gesture.debug = True  # Enable debug prints
            recognized = cylindrical_gesture.recognize(hand_landmarks)
            if recognized:
                self.holding_object = True
                return True, cylindrical_gesture.get_name(), "", False

        # Reset holding_object if cylindrical grasp isn't detected
        self.holding_object = False
            
        # Sort gestures by priority if they have that attribute
        sorted_gestures = sorted(
            self.gestures, 
            key=lambda g: getattr(g, 'priority', 5), 
            reverse=True  # Higher priority first
        )
        
        # If no active gesture or previous one ended, check all gestures in priority order
        for gesture in sorted_gestures:
            # Skip ThumbTouchAll if user is holding an object
            if self.holding_object and gesture.get_name() == "Thumb Touch All":
                continue
                
            # Skip if this gesture requires sequencing and not ready to start new sequence
            if gesture.get_name() == "Thumb Touch All":
                thumb_gesture = gesture
                if hasattr(thumb_gesture, 'completed') and thumb_gesture.completed:
                    # Skip if still in completion display phase
                    continue
            
            # Skip if we already checked cylindrical grasp
            if gesture.get_name() == "Cylindrical Grasp":
                continue
                
            # Pass current time to gesture recognizer for timing-based gestures
            recognized = False
            if hasattr(gesture, 'recognize') and callable(getattr(gesture, 'recognize')):
                # Check if the recognize method accepts a time parameter
                import inspect
                params = inspect.signature(gesture.recognize).parameters
                if len(params) > 1:
                    recognized = gesture.recognize(hand_landmarks, current_time)
                else:
                    recognized = gesture.recognize(hand_landmarks)
                    
            if recognized:
                # Get progress information if available
                progress = ""
                if hasattr(gesture, 'get_progress') and callable(getattr(gesture, 'get_progress')):
                    progress = gesture.get_progress()
                
                # Check if it's an in-progress gesture
                is_running = False
                if hasattr(gesture, 'is_in_progress') and callable(getattr(gesture, 'is_in_progress')):
                    is_running = gesture.is_in_progress()
                    if is_running:
                        # Set as active gesture if it's in progress
                        self.active_gesture = gesture.get_name()
                
                return True, gesture.get_name(), progress, is_running
        
        return False, "Unknown", "", False
    
    def extract_hand_keypoints(self, frame):
        """
        Extract hand keypoints and recognize gestures from a single frame
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            dict: Keypoints and recognition results
        """
        import time
        current_time = time.time()
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        keypoints = []
        recognized_gestures = []
        completed_gestures = []
        in_progress_gestures = []
        progress_info = {}

        # First, check if we need to keep displaying previously recognized gestures
        for gesture_name, display_until in list(self.gesture_display_times.items()):
            if current_time < display_until:
                completed_gestures.append(gesture_name)
            else:
                self.gesture_display_times.pop(gesture_name, None)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hand_keypoints = []
                for i, landmark in enumerate(hand_landmarks.landmark):
                    hand_keypoints.append({
                        "id": i,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

                keypoints.append({"keypoints": hand_keypoints})

                # Recognize the gesture
                recognized, gesture_name, progress, is_running = self.recognize_gesture(hand_landmarks, current_time)
                if recognized:
                    recognized_gestures.append(gesture_name)
                    
                    if is_running:
                        in_progress_gestures.append(gesture_name)
                    
                    # Check if this is the "Thumb Touch All" gesture and it wasn't previously in the completed list
                    if gesture_name == "Thumb Touch All":
                        thumb_gesture = self.get_gesture_by_name("Thumb Touch All")
                        if thumb_gesture and hasattr(thumb_gesture, 'completed') and thumb_gesture.completed:
                            if gesture_name not in self.last_completed_gestures:
                                # Add to completed list with display time
                                self.last_completed_gestures.add(gesture_name)
                                self.gesture_display_times[gesture_name] = current_time + 3.0  # Display for 3 seconds
                                completed_gestures.append(gesture_name)
                    
                    if progress:
                        progress_info[gesture_name] = progress

        # Update the set of completed gestures
        self.last_completed_gestures = set(completed_gestures)

        return {
            "recognized": len(recognized_gestures) > 0, 
            "gestures": recognized_gestures,
            "in_progress": in_progress_gestures,
            "completed_gestures": completed_gestures,
            "hands": keypoints,
            "progress": progress_info
        }
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on the frame"""
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def process_video(self, cap):
        """Generator function to process video frames"""
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            height, width = frame.shape[:2]
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            
            # Lists to track recognition status
            regular_gestures = []
            in_progress_gestures = []
            completed_gestures = []
            progress_info = {}

            # Initialize recognition status
            cylindrical_status = "Cylindrical: Not Recognized"
            status_color = (0, 0, 255)  # Red for not recognized
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Check for cylindrical grasp
                    cylindrical_gesture = self.get_gesture_by_name("Cylindrical Grasp")
                    if cylindrical_gesture and cylindrical_gesture.recognize(hand_landmarks):
                        cylindrical_status = "Cylindrical: Recognized"
                        status_color = (0, 255, 0)  # Green for recognized
                        
                        # Optional: Draw a cylinder icon
                        palm_x = int(hand_landmarks.landmark[9].x * width)
                        palm_y = int(hand_landmarks.landmark[9].y * height)
                        cv2.ellipse(frame, (palm_x, palm_y), (30, 15), 0, 0, 360, (0, 255, 0), 2)
                    
                    # Draw the status text (top-left corner)
                    cv2.putText(frame, cylindrical_status, (700, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Check if we need to keep displaying previously completed gestures
            for gesture_name, display_until in list(self.gesture_display_times.items()):
                if current_time < display_until:
                    completed_gestures.append(gesture_name)
                else:
                    self.gesture_display_times.pop(gesture_name, None)
                    # Remove from last completed set when display time expires
                    if gesture_name in self.last_completed_gestures:
                        self.last_completed_gestures.remove(gesture_name)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.draw_landmarks(frame, hand_landmarks)
                    
                    # Draw thumb-to-index distance line and measurement
                    thumb_x = int(hand_landmarks.landmark[4].x * width)
                    thumb_y = int(hand_landmarks.landmark[4].y * height)
                    index_base_x = int(hand_landmarks.landmark[5].x * width)
                    index_base_y = int(hand_landmarks.landmark[5].y * height)
                    
                    # Draw the distance line
                    cv2.line(frame, (thumb_x, thumb_y), (index_base_x, index_base_y), (0, 255, 255), 2)
                    
                    # Display the distance (normalized to 0-1 range)
                    distance = ((thumb_x-index_base_x)**2 + (thumb_y-index_base_y)**2)**0.5 / width
                    cv2.putText(frame, f"{distance:.2f}", 
                            ((thumb_x+index_base_x)//2, (thumb_y+index_base_y)//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                    # Recognize the gesture
                    recognized, gesture_name, progress, is_running = self.recognize_gesture(hand_landmarks, current_time)
                    if recognized:
                        # Add to appropriate list based on status
                        regular_gestures.append(gesture_name)
                        
                        if is_running:
                            in_progress_gestures.append(gesture_name)
                        
                        # Special handling for Thumb Touch All gesture
                        if gesture_name == "Thumb Touch All":
                            thumb_gesture = self.get_gesture_by_name("Thumb Touch All")
                            if thumb_gesture and hasattr(thumb_gesture, 'completed') and thumb_gesture.completed:
                                if gesture_name not in self.last_completed_gestures:
                                    # Add to completed list with display time
                                    self.last_completed_gestures.add(gesture_name)
                                    self.gesture_display_times[gesture_name] = current_time + 3.0  # Display for 3 seconds
                                    completed_gestures.append(gesture_name)
                                    
                                    # Reset the gesture after completion to allow for repeated recognition
                                    def reset_later():
                                        time.sleep(3.0)  # Wait for display duration
                                        thumb_gesture.reset()
                                    
                                    # Start a thread to reset the gesture after display time
                                    import threading
                                    threading.Thread(target=reset_later).start()
                        
                        if progress:
                            progress_info[gesture_name] = progress

            # Draw the regular recognition status on frame
            recognized_status = "Recognized" if regular_gestures else "Not Recognized"
            status_color = (0, 255, 0) if regular_gestures else (0, 0, 255)
            cv2.putText(frame, recognized_status, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
            
            # Display regular gestures below status
            if regular_gestures:
                gesture_text = ", ".join(regular_gestures)
                cv2.putText(frame, gesture_text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Add this with other visual feedback code
            if "Cylindrical Grasp" in regular_gestures:
                # Draw a green cylinder in the hand area
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Get palm center (approximate)
                        palm_x = int(hand_landmarks.landmark[9].x * width)
                        palm_y = int(hand_landmarks.landmark[9].y * height)
                        
                        # Draw a cylinder representation
                        cv2.ellipse(frame, (palm_x, palm_y-30), (30, 15), 0, 0, 360, (0, 255, 0), 2)
                        cv2.line(frame, (palm_x-30, palm_y-30), (palm_x-30, palm_y+30), (0, 255, 0), 2)
                        cv2.line(frame, (palm_x+30, palm_y-30), (palm_x+30, palm_y+30), (0, 255, 0), 2)
                        cv2.ellipse(frame, (palm_x, palm_y+30), (30, 15), 0, 0, 360, (0, 255, 0), 2)
                        
                        # Add text
                        cv2.putText(frame, "Holding", (palm_x-40, palm_y-50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display in-progress gestures with progress info
            if in_progress_gestures:
                for i, gesture in enumerate(in_progress_gestures):
                    in_progress_text = f"{gesture} IN PROGRESS"
                    if gesture in progress_info:
                        in_progress_text += f" - {progress_info[gesture]}"
                    
                    # Create a semi-transparent banner for in-progress gesture
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 150+i*50), (width, 200+i*50), (255, 165, 0), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    
                    cv2.putText(frame, in_progress_text, (50, 180+i*50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display completed gestures notification in a different area (bottom of frame)
            if completed_gestures:
                # Create a semi-transparent banner at the bottom
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, height-100), (width, height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                for i, gesture in enumerate(completed_gestures):
                    # Show completion message in a larger, more prominent font
                    cv2.putText(frame, f"{gesture} DONE", (width//2-230, height-60+i*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            yield frame, regular_gestures
            
    def close(self):
        """Release resources"""
        self.hands.close()