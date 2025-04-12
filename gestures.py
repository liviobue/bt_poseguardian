import mediapipe as mp
import numpy as np
import cv2
import math

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
        
        # Import math module if needed for angular calculations
        import math
        self.math = math
    
    def recognize(self, hand_landmarks):
        """Checks if all fingers are fully extended with appropriate separation between them."""
        
        # Landmark indices
        FINGERTIPS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        MIDDLE_JOINTS = [3, 7, 11, 15, 19] # PIP joints (middle of fingers)
        BASE_JOINTS = [2, 6, 10, 14, 18]   # MCP joints (base of fingers)
        WRIST = 0  # Wrist landmark

        tolerance = 0.05  # Tolerance for natural variations
        
        # Determine if it's a left or right hand
        thumb_tip_x = hand_landmarks.landmark[4].x
        pinky_tip_x = hand_landmarks.landmark[20].x
        
        is_right_hand = thumb_tip_x < pinky_tip_x
        
        # Check each finger is extended
        for i, (tip, middle, base) in enumerate(zip(FINGERTIPS, MIDDLE_JOINTS, BASE_JOINTS)):
            tip_y = hand_landmarks.landmark[tip].y
            middle_y = hand_landmarks.landmark[middle].y
            base_y = hand_landmarks.landmark[base].y
            
            # For thumb (i=0), handle differently based on hand orientation
            if i == 0:  # Thumb
                tip_x = hand_landmarks.landmark[tip].x
                middle_x = hand_landmarks.landmark[middle].x
                base_x = hand_landmarks.landmark[base].x
                
                # Check thumb extension based on hand orientation
                if is_right_hand:
                    if not (tip_x < middle_x < base_x):
                        return False
                else:
                    if not (tip_x > middle_x > base_x):
                        return False
                
                # Check if thumb is relatively straight
                expected_middle_x = (tip_x + base_x) / 2
                if abs(middle_x - expected_middle_x) > tolerance:
                    return False
            else:
                # For other fingers, they should be extended upward (decreasing y)
                if not (tip_y < middle_y < base_y):  
                    return False
                
                # Ensure fingers are nearly straight
                expected_middle_y = (tip_y + base_y) / 2
                if abs(middle_y - expected_middle_y) > tolerance:
                    return False
        
        # Now check the separation between each pair of fingertips
        # We'll directly calculate the distance between fingertips
        # and ensure they're not too close
        
        # UPDATED: Increased minimum distances between all fingers
        min_thumb_index_distance = 0.12   # Increased from 0.11
        min_index_middle_distance = 0.10  # Increased from 0.09
        min_middle_ring_distance = 0.10   # Increased from general min_distance
        min_ring_pinky_distance = 0.12    # Increased from 0.10
        
        # Get all fingertips for easier access
        fingertips = [hand_landmarks.landmark[i] for i in FINGERTIPS]
        finger_bases = [hand_landmarks.landmark[i] for i in BASE_JOINTS]
        
        # Check distance between thumb and index
        thumb_tip = fingertips[0]
        index_tip = fingertips[1]
        distance = ((thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2 + 
                    (thumb_tip.z - index_tip.z)**2)**0.5
        if distance < min_thumb_index_distance:
            return False
        
        # Calculate angle between thumb and index finger
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
        
        # Calculate the dot product
        dot_product = (thumb_vector[0] * index_vector[0] + 
                      thumb_vector[1] * index_vector[1] + 
                      thumb_vector[2] * index_vector[2])
        
        # Calculate the magnitudes
        thumb_mag = (thumb_vector[0]**2 + thumb_vector[1]**2 + thumb_vector[2]**2)**0.5
        index_mag = (index_vector[0]**2 + index_vector[1]**2 + index_vector[2]**2)**0.5
        
        # Calculate the cosine of the angle
        if thumb_mag * index_mag == 0:
            return False  # Avoid division by zero
            
        cos_angle = dot_product / (thumb_mag * index_mag)
        
        # Ensure the angle is within appropriate range (-1 to 1)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        # Convert to angle in degrees
        angle = self.math.degrees(self.math.acos(cos_angle))
        
        # UPDATED: Check if the angle between thumb and index is large enough (more spread)
        min_thumb_index_angle = 22
        if angle < min_thumb_index_angle:
            return False
        
        # UPDATED: Calculate and check angles between all finger pairs
        # Define minimum angles between fingers (in degrees)
        min_finger_angle = 10  # Minimum angle between adjacent fingers
        
        # Check distance and angle between index and middle
        middle_tip = fingertips[2]
        middle_base = finger_bases[2]
        
        # Distance check
        distance = ((index_tip.x - middle_tip.x)**2 + 
                    (index_tip.y - middle_tip.y)**2 + 
                    (index_tip.z - middle_tip.z)**2)**0.5
        if distance < min_index_middle_distance:
            return False
        
        # Angle check between index and middle
        middle_vector = [
            middle_tip.x - middle_base.x,
            middle_tip.y - middle_base.y,
            middle_tip.z - middle_base.z
        ]
        
        dot_product = (index_vector[0] * middle_vector[0] + 
                      index_vector[1] * middle_vector[1] + 
                      index_vector[2] * middle_vector[2])
        
        middle_mag = (middle_vector[0]**2 + middle_vector[1]**2 + middle_vector[2]**2)**0.5
        
        if index_mag * middle_mag == 0:
            return False
            
        cos_angle = dot_product / (index_mag * middle_mag)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle = self.math.degrees(self.math.acos(cos_angle))
        if angle < min_finger_angle:
            return False
        
        # Check distance and angle between middle and ring
        ring_tip = fingertips[3]
        ring_base = finger_bases[3]
        
        # Distance check
        distance = ((middle_tip.x - ring_tip.x)**2 + 
                    (middle_tip.y - ring_tip.y)**2 + 
                    (middle_tip.z - ring_tip.z)**2)**0.5
        if distance < min_middle_ring_distance:
            return False
        
        # Angle check between middle and ring
        ring_vector = [
            ring_tip.x - ring_base.x,
            ring_tip.y - ring_base.y,
            ring_tip.z - ring_base.z
        ]
        
        dot_product = (middle_vector[0] * ring_vector[0] + 
                      middle_vector[1] * ring_vector[1] + 
                      middle_vector[2] * ring_vector[2])
        
        ring_mag = (ring_vector[0]**2 + ring_vector[1]**2 + ring_vector[2]**2)**0.5
        
        if middle_mag * ring_mag == 0:
            return False
            
        cos_angle = dot_product / (middle_mag * ring_mag)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle = self.math.degrees(self.math.acos(cos_angle))
        if angle < min_finger_angle:
            return False
        
        # Check distance and angle between ring and pinky
        pinky_tip = fingertips[4]
        pinky_base = finger_bases[4]
        
        # Distance check
        distance = ((ring_tip.x - pinky_tip.x)**2 + 
                    (ring_tip.y - pinky_tip.y)**2 + 
                    (ring_tip.z - pinky_tip.z)**2)**0.5
        if distance < min_ring_pinky_distance:
            return False
        
        # Angle check between ring and pinky
        pinky_vector = [
            pinky_tip.x - pinky_base.x,
            pinky_tip.y - pinky_base.y,
            pinky_tip.z - pinky_base.z
        ]
        
        dot_product = (ring_vector[0] * pinky_vector[0] + 
                      ring_vector[1] * pinky_vector[1] + 
                      ring_vector[2] * pinky_vector[2])
        
        pinky_mag = (pinky_vector[0]**2 + pinky_vector[1]**2 + pinky_vector[2]**2)**0.5
        
        if ring_mag * pinky_mag == 0:
            return False
            
        cos_angle = dot_product / (ring_mag * pinky_mag)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle = self.math.degrees(self.math.acos(cos_angle))
        if angle < min_finger_angle:
            return False
        
        # If all checks pass, it's a proper open hand
        return True

class ThumbTouchAllFingersGesture(HandGesture):
    """Gesture for thumb touching all other fingers in sequence"""
    
    def __init__(self):
        self.name = "Thumb Touch All"
        self.finger_touched = [False, False, False, False]  # Index, Middle, Ring, Pinky
        self.current_finger_idx = 0  # Track which finger should be touched next
        self.last_touch_time = 0
        self.display_duration = 3.0  # seconds to display the recognition message
        self.sequence_timeout = 5.0  # seconds allowed to complete the sequence
        self.sequence_start_time = None
        self.completed = False
        self.completion_time = 0
    
    def is_thumb_touching_finger(self, hand_landmarks, finger_tip_idx):
        """Check if thumb tip is touching another fingertip"""
        thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
        finger_tip = hand_landmarks.landmark[finger_tip_idx]
        
        # Calculate 3D distance between thumb tip and fingertip
        distance = ((thumb_tip.x - finger_tip.x)**2 + 
                    (thumb_tip.y - finger_tip.y)**2 + 
                    (thumb_tip.z - finger_tip.z)**2)**0.5
        
        # Threshold distance for "touching"
        return distance < 0.05  # Adjust this threshold based on testing
    
    def recognize(self, hand_landmarks, current_time=None):
        """
        Check if the thumb has touched all fingers in sequence
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            current_time: Current timestamp for timeout tracking
            
        Returns:
            bool: True if gesture is recognized, False otherwise
        """
        if current_time is None:
            import time
            current_time = time.time()
            
        # If gesture was already completed and we're in display period
        if self.completed:
            if current_time - self.completion_time < self.display_duration:
                return True
            else:
                # Reset after display period to allow for a new sequence
                self.reset()
                return False
        
        # Finger tip landmark indices
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        
        # Start tracking sequence when first finger is touched
        if self.sequence_start_time is None and self.is_thumb_touching_finger(hand_landmarks, finger_tips[0]):
            self.sequence_start_time = current_time
            self.finger_touched[0] = True
            self.current_finger_idx = 1  # Next expect the middle finger
            return False
            
        # Check for timeout if sequence has started
        if self.sequence_start_time is not None:
            if current_time - self.sequence_start_time > self.sequence_timeout:
                self.reset()  # Reset if timeout
                return False
        
        # If sequence has started, check for next finger in sequence
        if self.sequence_start_time is not None and self.current_finger_idx < 4:
            # Check if the expected next finger is being touched
            if self.is_thumb_touching_finger(hand_landmarks, finger_tips[self.current_finger_idx]):
                self.finger_touched[self.current_finger_idx] = True
                self.current_finger_idx += 1  # Move to next finger
                
                # If all fingers have been touched in sequence
                if self.current_finger_idx >= 4:
                    self.completed = True
                    self.completion_time = current_time
                    return True
            
            # Check if a finger out of sequence is being touched
            for i in range(4):
                if i != self.current_finger_idx and not self.finger_touched[i]:
                    if self.is_thumb_touching_finger(hand_landmarks, finger_tips[i]):
                        # Out of sequence touch - reset
                        self.reset()
                        return False
                        
        return False
    
    def reset(self):
        """Reset the gesture tracking state"""
        self.finger_touched = [False, False, False, False]
        self.current_finger_idx = 0
        self.sequence_start_time = None
        self.completed = False

    def get_progress(self):
        """Return the progress of the gesture sequence"""
        touched_count = sum(1 for touched in self.finger_touched if touched)
        return f"{touched_count}/4 fingers"

class PointingGesture(HandGesture):
    """Gesture for pointing (index finger extended, others curled)"""
    
    def __init__(self):
        self.name = "Pointing"
    
    def recognize(self, hand_landmarks):
        """Checks if index finger is extended while other fingers are curled."""
        
        # Landmark indices for index finger
        INDEX_TIP = 8
        INDEX_PIP = 7
        INDEX_MCP = 6
        
        # Other fingertips
        OTHER_TIPS = [4, 12, 16, 20]  # Thumb, Middle, Ring, Pinky
        OTHER_BASES = [2, 10, 14, 18]  # Respective MCP joints
        
        tolerance = 0.05  # Tolerance for natural variations
        
        # 1. Check if index finger is extended
        index_tip_y = hand_landmarks.landmark[INDEX_TIP].y
        index_pip_y = hand_landmarks.landmark[INDEX_PIP].y
        index_mcp_y = hand_landmarks.landmark[INDEX_MCP].y
        
        if not (index_tip_y < index_pip_y < index_mcp_y):
            return False
        
        # 2. Check if other fingers are curled
        for tip, base in zip(OTHER_TIPS, OTHER_BASES):
            tip_y = hand_landmarks.landmark[tip].y
            base_y = hand_landmarks.landmark[base].y
            
            # Special case for thumb
            if tip == 4:
                # For thumb, we check that it's not fully extended
                thumb_tip = hand_landmarks.landmark[4]
                thumb_ip = hand_landmarks.landmark[3]
                thumb_mcp = hand_landmarks.landmark[2]
                thumb_cmc = hand_landmarks.landmark[1]
                
                # Calculate distance between thumb tip and index finger base to check if thumb is close
                thumb_tip_x = thumb_tip.x
                index_base_x = hand_landmarks.landmark[5].x  # Index finger base
                
                # Thumb should not be extended away from hand
                distance = ((thumb_tip_x - index_base_x)**2)**0.5
                if distance > 0.1:  # Arbitrary threshold, may need adjustment
                    return False
            else:
                # For all other fingers (middle, ring, pinky), ensure they're curled
                if tip_y <= base_y + tolerance:  # Adding tolerance for minor variations
                    return False
        
        return True


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
        
        # Register default gestures
        self.register_gesture(OpenHandGesture())
        self.register_gesture(ThumbTouchAllFingersGesture())  # Replace ClosedFistGesture
        self.register_gesture(PointingGesture())
    
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
            tuple: (recognized: bool, gesture_name: str, progress: str)
        """
        import time
        if current_time is None:
            current_time = time.time()
            
        for gesture in self.gestures:
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
                    
                return True, gesture.get_name(), progress
        
        return False, "Unknown", ""
    
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
                recognized, gesture_name, progress = self.recognize_gesture(hand_landmarks, current_time)
                if recognized:
                    recognized_gestures.append(gesture_name)
                    
                    # Check if this is the "Thumb Touch All" gesture and it wasn't previously in the completed list
                    if gesture_name == "Thumb Touch All" and gesture_name not in self.last_completed_gestures:
                        # Add to completed list with display time
                        self.last_completed_gestures.add(gesture_name)
                        self.gesture_display_times[gesture_name] = current_time + 3.0  # Display for 3 seconds
                    
                    if progress:
                        progress_info[gesture_name] = progress

        # Update the set of completed gestures
        self.last_completed_gestures = set(completed_gestures)

        return {
            "recognized": len(recognized_gestures) > 0, 
            "gestures": recognized_gestures,
            "completed_gestures": completed_gestures,
            "hands": keypoints,
            "progress": progress_info
        }
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on the frame"""
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def process_video(self, cap):
        """Generator function to process video frames"""
        import time
        
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
            completed_gestures = []
            progress_info = {}

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
                    
                    # Recognize the gesture
                    recognized, gesture_name, progress = self.recognize_gesture(hand_landmarks, current_time)
                    if recognized:
                        # Add to regular gestures list
                        regular_gestures.append(gesture_name)
                        
                        # Special handling for Thumb Touch All gesture
                        if gesture_name == "Thumb Touch All" and gesture_name not in self.last_completed_gestures:
                            # Add to completed list with display time
                            self.last_completed_gestures.add(gesture_name)
                            self.gesture_display_times[gesture_name] = current_time + 3.0  # Display for 3 seconds
                            completed_gestures.append(gesture_name)
                            
                            # Reset the gesture after completion to allow for repeated recognition
                            thumb_gesture = self.get_gesture_by_name("Thumb Touch All")
                            if thumb_gesture:
                                # Schedule a reset after the display time
                                def reset_later():
                                    import threading
                                    import time
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
            
            # Display completed gestures notification in a different area (bottom of frame)
            if completed_gestures:
                # Create a semi-transparent banner at the bottom
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, height-100), (width, height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                for i, gesture in enumerate(completed_gestures):
                    # Show completion message in a larger, more prominent font
                    cv2.putText(frame, "DONE", (width//2-230, height-60+i*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            yield frame, regular_gestures
            
    def close(self):
        """Release resources"""
        self.hands.close()