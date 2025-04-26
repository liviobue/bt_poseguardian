import mediapipe as mp
import numpy as np
import cv2
import math
import time
import threading
import inspect

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
        
        # Calculate hand scale to normalize distance thresholds
        # Use distance from wrist to middle finger MCP as reference
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
        
        # Adjust thresholds based on hand scale
        min_segment_distance = 0.035 * hand_scale  # More permissive minimum distance
        min_extended_ratio = 0.9     # Lower ratio for straight finger (more permissive)
        max_z_diff_ratio = 0.95       # More permissive z-difference threshold
        
        # Check each finger is extended
        for i, (tip, middle, base) in enumerate(zip(FINGERTIPS, MIDDLE_JOINTS, BASE_JOINTS)):
            # Get 3D coordinates for each joint
            tip_coords = [hand_landmarks.landmark[tip].x, 
                         hand_landmarks.landmark[tip].y, 
                         hand_landmarks.landmark[tip].z]
            
            middle_coords = [hand_landmarks.landmark[middle].x, 
                            hand_landmarks.landmark[middle].y, 
                            hand_landmarks.landmark[middle].z]
            
            base_coords = [hand_landmarks.landmark[base].x, 
                          hand_landmarks.landmark[base].y, 
                          hand_landmarks.landmark[base].z]
            
            # Calculate 3D distances between joints
            tip_to_middle_dist = self.calculate_3d_distance(tip_coords, middle_coords)
            middle_to_base_dist = self.calculate_3d_distance(middle_coords, base_coords)
            tip_to_base_dist = self.calculate_3d_distance(tip_coords, base_coords)
            
            # Special handling for thumb
            if i == 0:  # Thumb
                # For thumb, check extension based on hand orientation
                if is_right_hand:
                    if not (hand_landmarks.landmark[tip].x < hand_landmarks.landmark[middle].x < hand_landmarks.landmark[base].x):
                        return False
                else:
                    if not (hand_landmarks.landmark[tip].x > hand_landmarks.landmark[middle].x > hand_landmarks.landmark[base].x):
                        return False
                
                # Check distances for thumb extension - less strict for thumb
                if tip_to_middle_dist < 0.8 * min_segment_distance or middle_to_base_dist < 0.8 * min_segment_distance:
                    return False
                
            else:
                # For other fingers
                # 1. Check if finger is extended upward (y coordinate decreasing)
                if not (hand_landmarks.landmark[tip].y < hand_landmarks.landmark[middle].y < hand_landmarks.landmark[base].y):
                    return False
                
                # 2. Check distances to ensure finger is straight and extended
                if tip_to_middle_dist < min_segment_distance or middle_to_base_dist < min_segment_distance:
                    return False
                
                # 3. Check the ratio of direct distance to segment distances
                # This is still useful for detecting curling toward the camera, but with more permissive threshold
                extended_ratio = tip_to_base_dist / (tip_to_middle_dist + middle_to_base_dist)
                if extended_ratio < min_extended_ratio:
                    return False
                
                # 4. Z-depth check - but more permissive
                z_tip = hand_landmarks.landmark[tip].z
                z_middle = hand_landmarks.landmark[middle].z
                z_base = hand_landmarks.landmark[base].z
                
                # Calculate the range of z values
                z_range = max(abs(z_tip - z_middle), abs(z_middle - z_base), abs(z_tip - z_base))
                
                # Calculate the total finger length in 3D
                finger_segment_length = tip_to_middle_dist + middle_to_base_dist
                
                # More permissive check for z variation
                if z_range > max_z_diff_ratio * finger_segment_length:
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
        min_thumb_index_angle = 30  # Slightly more permissive
        if angle < min_thumb_index_angle:
            return False
        
        # Check distances and angles between other finger pairs
        min_finger_angle = 8  # Slightly more permissive minimum angle
        
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
        Check if the thumb has touched all fingers in sequence, not simultaneously
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            current_time: Current timestamp for timeout tracking
            
        Returns:
            bool: True if gesture is recognized, False otherwise
        """
        if current_time is None:
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
        
        # Check how many fingers are currently touching the thumb
        currently_touching = [self.is_thumb_touching_finger(hand_landmarks, tip) for tip in finger_tips]
        touching_count = sum(currently_touching)
        
        # If more than one finger is touching, it's not a valid sequence
        if touching_count > 1:
            self.reset()
            return False
        
        # Start tracking sequence when first finger is touched
        if self.sequence_start_time is None and self.is_thumb_touching_finger(hand_landmarks, finger_tips[0]):
            # Make sure other fingers are not touching
            if touching_count == 1:
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
            # Check if the expected next finger is being touched (and ONLY that finger)
            expected_finger_touching = self.is_thumb_touching_finger(hand_landmarks, finger_tips[self.current_finger_idx])
            
            if expected_finger_touching and touching_count == 1:
                self.finger_touched[self.current_finger_idx] = True
                self.current_finger_idx += 1  # Move to next finger
                
                # If all fingers have been touched in sequence
                if self.current_finger_idx >= 4:
                    self.completed = True
                    self.completion_time = current_time
                    return True
            
            # Check if a finger out of sequence is being touched
            elif touching_count > 0:
                for i in range(4):
                    if i != self.current_finger_idx and self.is_thumb_touching_finger(hand_landmarks, finger_tips[i]):
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

class CylindricalGraspGesture(HandGesture):
    """Gesture for cylindrical grasp (all digits flexed in a C-shape as if grasping a cylinder)"""
    
    def __init__(self):
        self.name = "Cylindrical Grasp"
        self.math = math
    
    def recognize(self, hand_landmarks):
        """
        Recognizes the cylindrical grasp gesture based on clinical specifications.
        
        For a cylindrical grasp:
        - All fingers are flexed in a C-shape around an imaginary cylindrical object
        - MCP joints flex to 45-60°
        - PIP and DIP joints flex to 40-50°
        - Thumb opposes the ulnar side of the middle finger's proximal phalanx
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            bool: True if gesture is recognized, False otherwise
        """
        # Landmark indices
        WRIST = 0
        THUMB_CMC = 1  # Carpometacarpal joint
        THUMB_MCP = 2  # Metacarpophalangeal joint
        THUMB_IP = 3   # Interphalangeal joint
        THUMB_TIP = 4
        
        # Finger base (MCP), PIP, DIP and tips
        INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
        MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
        RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
        PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
        
        # Finger groups for easier iteration
        MCPs = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        PIPs = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        DIPs = [INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]
        TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        
        # Calculate hand scale to normalize distance thresholds
        wrist_point = [
            hand_landmarks.landmark[WRIST].x,
            hand_landmarks.landmark[WRIST].y,
            hand_landmarks.landmark[WRIST].z
        ]
        middle_mcp_point = [
            hand_landmarks.landmark[MIDDLE_MCP].x,
            hand_landmarks.landmark[MIDDLE_MCP].y,
            hand_landmarks.landmark[MIDDLE_MCP].z
        ]
        hand_scale = self.calculate_3d_distance(wrist_point, middle_mcp_point)
        
        # Check if the hand is in a cylindrical grasp configuration
        
        # 1. Check MCP joint flexion angles (should be 45-60°)
        for mcp in MCPs:
            # Calculate vectors for MCP angle
            mcp_point = hand_landmarks.landmark[mcp]
            pip_point = hand_landmarks.landmark[mcp + 1]  # PIP is always MCP + 1
            palm_direction = self.get_palm_direction(hand_landmarks)
            
            # Vector from MCP to PIP
            mcp_to_pip = [
                pip_point.x - mcp_point.x,
                pip_point.y - mcp_point.y,
                pip_point.z - mcp_point.z
            ]
            
            # Calculate angle between palm direction and finger direction
            angle = self.calculate_angle(palm_direction, mcp_to_pip)
            
            # Convert to joint angle (90° - angle between vectors)
            joint_angle = 90 - angle
            
            # MCP joints should flex to 45-60°
            if not (40 <= joint_angle <= 65):  # Slightly wider range for tolerance
                return False
        
        # 2. Check PIP joint flexion angles (should be 40-50°)
        for pip in PIPs:
            pip_angle = self.calculate_joint_angle(hand_landmarks, pip-1, pip, pip+1)
            if not (35 <= pip_angle <= 55):  # Slightly wider range for tolerance
                return False
        
        # 3. Check DIP joint flexion angles (should be 40-50°)
        for dip in DIPs:
            dip_angle = self.calculate_joint_angle(hand_landmarks, dip-1, dip, dip+1)
            if not (35 <= dip_angle <= 55):  # Slightly wider range for tolerance
                return False
        
        # 4. Check thumb opposition to middle finger's proximal phalanx
        thumb_tip = [
            hand_landmarks.landmark[THUMB_TIP].x,
            hand_landmarks.landmark[THUMB_TIP].y,
            hand_landmarks.landmark[THUMB_TIP].z
        ]
        
        middle_proximal = [
            hand_landmarks.landmark[MIDDLE_MCP].x,
            hand_landmarks.landmark[MIDDLE_MCP].y,
            hand_landmarks.landmark[MIDDLE_MCP].z
        ]
        
        # Distance between thumb tip and middle finger's proximal phalanx
        thumb_to_middle_dist = self.calculate_3d_distance(thumb_tip, middle_proximal)
        
        # The distance should be within a reasonable range based on hand scale
        # Optimal spacing is mentioned as 3.5-4.2 cm, but we need to normalize to hand scale
        optimal_min = 0.3 * hand_scale
        optimal_max = 0.5 * hand_scale
        
        if not (optimal_min <= thumb_to_middle_dist <= optimal_max):
            return False
        
        # 5. Check C-shape formation of fingers
        # For a proper C-shape, fingertips should be closer to the palm than PIPs
        for tip, pip, mcp in zip(TIPS, PIPs, MCPs):
            tip_point = hand_landmarks.landmark[tip]
            pip_point = hand_landmarks.landmark[pip]
            mcp_point = hand_landmarks.landmark[mcp]
            
            # Get vector from palm center to fingertip
            palm_center = self.calculate_palm_center(hand_landmarks)
            palm_to_tip = [
                tip_point.x - palm_center[0],
                tip_point.y - palm_center[1],
                tip_point.z - palm_center[2]
            ]
            
            # Get vector from palm center to PIP
            palm_to_pip = [
                pip_point.x - palm_center[0],
                pip_point.y - palm_center[1],
                pip_point.z - palm_center[2]
            ]
            
            # Distance from palm to tip should be less than distance from palm to PIP
            # This ensures the C-shape curling of fingers
            palm_tip_dist = self.calculate_vector_magnitude(palm_to_tip)
            palm_pip_dist = self.calculate_vector_magnitude(palm_to_pip)
            
            if palm_tip_dist >= palm_pip_dist:
                return False
        
        # 6. Check thumb CMC joint adduction (30-35° of palmar adduction)
        # This is approximated by checking the thumb's position relative to the index finger
        thumb_tip = hand_landmarks.landmark[THUMB_TIP]
        index_mcp = hand_landmarks.landmark[INDEX_MCP]
        
        # For a cylindrical grasp, thumb tip should be closer to the palm than extended
        if thumb_tip.z >= index_mcp.z:
            return False
        
        # If all checks pass, it's a cylindrical grasp
        return True
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return ((point1[0] - point2[0])**2 + 
                (point1[1] - point2[1])**2 + 
                (point1[2] - point2[2])**2)**0.5
    
    def calculate_vector_magnitude(self, vector):
        """Calculate the magnitude of a 3D vector"""
        return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5
    
    def calculate_angle(self, vector1, vector2):
        """Calculate angle between two vectors in degrees"""
        # Calculate the dot product
        dot_product = (vector1[0] * vector2[0] + 
                       vector1[1] * vector2[1] + 
                       vector1[2] * vector2[2])
        
        # Calculate the magnitudes
        mag1 = self.calculate_vector_magnitude(vector1)
        mag2 = self.calculate_vector_magnitude(vector2)
        
        # Avoid division by zero
        if mag1 * mag2 == 0:
            return 0
            
        cos_angle = dot_product / (mag1 * mag2)
        
        # Ensure the cosine is within valid range (-1 to 1)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        # Convert to angle in degrees
        return self.math.degrees(self.math.acos(cos_angle))
    
    def calculate_joint_angle(self, hand_landmarks, prev_idx, curr_idx, next_idx):
        """Calculate the angle at a joint in degrees"""
        prev_point = hand_landmarks.landmark[prev_idx]
        curr_point = hand_landmarks.landmark[curr_idx]
        next_point = hand_landmarks.landmark[next_idx]
        
        # Vector from current joint to previous joint
        vector1 = [
            prev_point.x - curr_point.x,
            prev_point.y - curr_point.y,
            prev_point.z - curr_point.z
        ]
        
        # Vector from current joint to next joint
        vector2 = [
            next_point.x - curr_point.x,
            next_point.y - curr_point.y,
            next_point.z - curr_point.z
        ]
        
        # Calculate angle between vectors
        angle = self.calculate_angle(vector1, vector2)
        
        # Return the supplement of the angle to get the joint angle
        return 180 - angle
    
    def get_palm_direction(self, hand_landmarks):
        """Calculate the palm normal direction vector"""
        # Use index, middle, and pinky MCP joints to define the palm plane
        index_mcp = hand_landmarks.landmark[5]
        middle_mcp = hand_landmarks.landmark[9]
        pinky_mcp = hand_landmarks.landmark[17]
        
        # Create vectors in the palm plane
        vector1 = [
            middle_mcp.x - index_mcp.x,
            middle_mcp.y - index_mcp.y,
            middle_mcp.z - index_mcp.z
        ]
        
        vector2 = [
            pinky_mcp.x - middle_mcp.x,
            pinky_mcp.y - middle_mcp.y,
            pinky_mcp.z - middle_mcp.z
        ]
        
        # Calculate cross product to get palm normal
        palm_normal = [
            vector1[1] * vector2[2] - vector1[2] * vector2[1],
            vector1[2] * vector2[0] - vector1[0] * vector2[2],
            vector1[0] * vector2[1] - vector1[1] * vector2[0]
        ]
        
        # Normalize the vector
        magnitude = self.calculate_vector_magnitude(palm_normal)
        if magnitude > 0:
            palm_normal = [
                palm_normal[0] / magnitude,
                palm_normal[1] / magnitude,
                palm_normal[2] / magnitude
            ]
            
        return palm_normal
    
    def calculate_palm_center(self, hand_landmarks):
        """Calculate the approximate center of the palm"""
        # Use the average position of MCP joints as palm center
        mcp_indices = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCPs
        
        sum_x = sum(hand_landmarks.landmark[idx].x for idx in mcp_indices)
        sum_y = sum(hand_landmarks.landmark[idx].y for idx in mcp_indices)
        sum_z = sum(hand_landmarks.landmark[idx].z for idx in mcp_indices)
        
        return [sum_x / len(mcp_indices), sum_y / len(mcp_indices), sum_z / len(mcp_indices)]

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
        self.register_gesture(ThumbTouchAllFingersGesture())
        self.register_gesture(CylindricalGraspGesture())
    
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
        if current_time is None:
            current_time = time.time()
            
        for gesture in self.gestures:
            # Pass current time to gesture recognizer for timing-based gestures
            recognized = False
            if hasattr(gesture, 'recognize') and callable(getattr(gesture, 'recognize')):
                # Check if the recognize method accepts a time parameter
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
                                    time.sleep(3.0)  # Wait for display duration
                                    thumb_gesture.reset()
                                
                                # Start a thread to reset the gesture after display time
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