import mediapipe as mp
import numpy as np
import cv2

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
    """Gesture for fully open hand with all fingers extended"""
    
    def __init__(self):
        self.name = "Open Hand"
    
    def recognize(self, hand_landmarks):
        """Checks if all fingers (including thumb) are fully extended (no bending)."""
        
        # Landmark indices
        FINGERTIPS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        MIDDLE_JOINTS = [3, 7, 11, 15, 19] # PIP joints (middle of fingers)
        BASE_JOINTS = [2, 6, 10, 14, 18]   # MCP joints (base of fingers)

        tolerance = 0.01  # Small allowed deviation for natural variations

        for tip, middle, base in zip(FINGERTIPS, MIDDLE_JOINTS, BASE_JOINTS):
            tip_y = hand_landmarks.landmark[tip].y
            middle_y = hand_landmarks.landmark[middle].y
            base_y = hand_landmarks.landmark[base].y

            # For thumb, we need to check x coordinate instead of y (thumb moves differently)
            if tip == 4:  # Thumb tip
                tip_x = hand_landmarks.landmark[tip].x
                middle_x = hand_landmarks.landmark[middle].x
                base_x = hand_landmarks.landmark[base].x
                
                # Thumb is extended if tip is further out than middle joint (x coordinate)
                if not (tip_x > middle_x > base_x):
                    return False
                
                # Check if thumb is straight
                expected_middle_x = (tip_x + base_x) / 2
                if abs(middle_x - expected_middle_x) > tolerance:
                    return False
            else:
                # For other fingers, check y coordinate as before
                if not (tip_y < middle_y < base_y):  
                    return False

                # Ensure fingers are nearly straight
                expected_middle_y = (tip_y + base_y) / 2
                if abs(middle_y - expected_middle_y) > tolerance:
                    return False  

        return True  # All fingers including thumb are straight


class ClosedFistGesture(HandGesture):
    """Gesture for closed fist (all fingers curled)"""
    
    def __init__(self):
        self.name = "Closed Fist"
    
    def recognize(self, hand_landmarks):
        """Checks if all fingers are curled into a fist."""
        
        # Landmark indices
        FINGERTIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        MIDDLE_JOINTS = [3, 7, 11, 15, 19]  # PIP joints
        BASE_JOINTS = [2, 6, 10, 14, 18]  # MCP joints
        WRIST = 0

        # Check if fingertips are below their respective base joints (curled)
        for i, (tip, base) in enumerate(zip(FINGERTIPS, BASE_JOINTS)):
            tip_y = hand_landmarks.landmark[tip].y
            base_y = hand_landmarks.landmark[base].y
            
            # For thumb, check position relative to palm rather than y position
            if i == 0:  # Thumb
                wrist_x = hand_landmarks.landmark[WRIST].x
                tip_x = hand_landmarks.landmark[tip].x
                
                # In a fist, thumb tip should be close to palm (not extended outward)
                # This check depends on whether it's a left or right hand
                palm_center_x = hand_landmarks.landmark[9].x  # Middle finger MCP
                
                # If thumb is far from palm center in the outward direction, it's not a fist
                if abs(tip_x - palm_center_x) > 0.1 and ((palm_center_x < wrist_x and tip_x < palm_center_x) or (palm_center_x > wrist_x and tip_x > palm_center_x)):
                    return False
            else:
                # For all other fingers, they should be curled down (higher y-value than base)
                if tip_y <= base_y:  # If tip is above or at same level as base, finger isn't curled
                    return False
                
                # Also check if fingertips are close to the palm in z-dimension
                tip_z = hand_landmarks.landmark[tip].z
                base_z = hand_landmarks.landmark[base].z
                
                # In a fist, fingertips should be closer to the camera than their bases
                if tip_z >= base_z:
                    return False

        return True


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
        
        # Register default gestures
        self.register_gesture(OpenHandGesture())
        self.register_gesture(ClosedFistGesture())
        self.register_gesture(PointingGesture())
    
    def register_gesture(self, gesture):
        """Add a new gesture to the recognizer"""
        self.gestures.append(gesture)
    
    def recognize_gesture(self, hand_landmarks):
        """
        Check which registered gesture matches the given hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (recognized: bool, gesture_name: str)
        """
        for gesture in self.gestures:
            if gesture.recognize(hand_landmarks):
                return True, gesture.get_name()
        
        return False, "Unknown"
    
    def extract_hand_keypoints(self, frame):
        """
        Extract hand keypoints and recognize gestures from a single frame
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            dict: Keypoints and recognition results
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        keypoints = []
        recognized_gestures = []

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
                recognized, gesture_name = self.recognize_gesture(hand_landmarks)
                if recognized:
                    recognized_gestures.append(gesture_name)

        return {
            "recognized": len(recognized_gestures) > 0, 
            "gestures": recognized_gestures,
            "hands": keypoints
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

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            recognized_gestures = []

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.draw_landmarks(frame, hand_landmarks)
                    
                    # Recognize the gesture
                    recognized, gesture_name = self.recognize_gesture(hand_landmarks)
                    if recognized:
                        recognized_gestures.append(gesture_name)

            yield frame, recognized_gestures
            
    def close(self):
        """Release resources"""
        self.hands.close()