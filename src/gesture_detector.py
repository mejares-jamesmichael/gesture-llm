import mediapipe as mp
import cv2
import os


class GestureDetector:
    def __init__(self):
        model_path = os.path.join(os.getcwd(), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "hand_landmarker.task not found. Please download it and place it in the project root."
            )

        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = self.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.timestamp_ms += 33

        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            gesture = self.detect_gesture(landmarks)
            return landmarks, gesture
        return None, None
    
    def detect_gesture(self, landmarks):
        fingers = self.get_finger_states(landmarks)
        
        if self.is_thumbs_up(landmarks, fingers):
            return 'thumbs_up'
        elif self.is_thumbs_down(landmarks, fingers):
            return 'thumbs_down'
        elif self.is_open_palm(fingers):
            return 'open_palm'
        elif self.is_fist(fingers):
            return 'fist'
        elif self.is_pointing(fingers):
            return 'pointing'
        
        return None
    
    def get_finger_states(self, landmarks):
        # MediaPipe Tasks landmarks have x, y, z attributes
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        finger_states = []
        for tip, pip in zip(finger_tips, finger_pips):
            tip_y = landmarks[tip].y
            pip_y = landmarks[pip].y
            finger_states.append(tip_y < pip_y)
        
        return finger_states
    
    def is_thumbs_up(self, landmarks, fingers):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_extended = thumb_tip.y < thumb_ip.y
        
        # Check if other fingers are closed (fingers 1-4 correspond to indices 1-4 in fingers list)
        other_fingers_closed = not any(fingers[1:])
        
        return thumb_extended and other_fingers_closed
    
    def is_thumbs_down(self, landmarks, fingers):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_extended = thumb_tip.y > thumb_ip.y
        
        other_fingers_closed = not any(fingers[1:])
        
        return thumb_extended and other_fingers_closed
    
    def is_open_palm(self, fingers):
        return all(fingers)
    
    def is_fist(self, fingers):
        return not any(fingers)
    
    def is_pointing(self, fingers):
        # Index (fingers[1]) is open, others (fingers[2:]) are closed
        return fingers[1] and not any(fingers[2:])
