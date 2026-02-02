import cv2
import numpy as np
import time


class Display:
    def __init__(self, font_scale=0.7, thickness=2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        
        # Animation state
        self.target_text = ""
        self.displayed_text = ""
        self.last_update_time = 0
        self.word_delay = 0.3  # Seconds between words
        self.words_to_display = []
        self.current_word_idx = 0
        
        # Hand connections (landmark indices)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (9, 10), (10, 11), (11, 12),              # Middle (0-9 is implicit via palm)
            (13, 14), (14, 15), (15, 16),             # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
            (5, 9), (9, 13), (13, 17)                 # Palm
        ]
    
    def display_frame(self, frame, current_gesture, buffered_sentence, llm_reply, timeout_remaining, landmarks=None):
        frame = frame.copy()
        
        # Update animation state
        self._update_animation(llm_reply)
        
        if landmarks:
            self._draw_landmarks(frame, landmarks)
        
        self._draw_gesture_info(frame, current_gesture, buffered_sentence)
        self._draw_timeout(frame, timeout_remaining)
        self._draw_typing_text(frame)
        
        return frame
    
    def _update_animation(self, llm_reply):
        # Reset if new text received
        if llm_reply != self.target_text:
            self.target_text = llm_reply
            self.words_to_display = llm_reply.split()
            self.displayed_text = ""
            self.current_word_idx = 0
            self.last_update_time = time.time()
        
        # Update displayed text word by word
        current_time = time.time()
        if self.current_word_idx < len(self.words_to_display):
            if current_time - self.last_update_time > self.word_delay:
                # Add next word
                next_word = self.words_to_display[self.current_word_idx]
                if self.displayed_text:
                    self.displayed_text += " " + next_word
                else:
                    self.displayed_text = next_word
                
                self.current_word_idx += 1
                self.last_update_time = current_time

    def _draw_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        points = []
        for lm in landmarks:
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))
        
        # Draw connections
        for start_idx, end_idx in self.connections:
            # Handle 0-9 connection separately if not in list, but I added palm connections
            # Wait, middle finger connects 0->9? No, usually 0->5, 0->17. 
            # 5-9-13-17 form the knuckles.
            # My connections list is a bit simplified, let's stick to standard if possible.
            # Actually, MediaPipe standard connections are:
            # 0-1, 1-2, 2-3, 3-4
            # 0-5, 5-6, 6-7, 7-8
            # 0-9, 9-10, 10-11, 11-12 (Wait, standard is usually 0->9 too? Or 5->9?)
            # Let's just draw lines between points that exist.
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        
        # Draw points
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

    def _draw_gesture_info(self, frame, gesture, sentence):
        y_pos = 40
        line_height = 35
        
        if gesture:
            text = f"Gesture: {gesture.upper()}"
            cv2.putText(frame, text, (20, y_pos), self.font, 
                       self.font_scale, (0, 255, 0), self.thickness)
            y_pos += line_height
        
        if sentence:
            text = f"Buffer: {sentence}"
            cv2.putText(frame, text, (20, y_pos), self.font,
                       self.font_scale, (0, 165, 255), self.thickness)
    
    def _draw_timeout(self, frame, timeout_remaining):
        height, width = frame.shape[:2]
        text = f"Timeout: {timeout_remaining:.1f}s"
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
        x_pos = width - text_size[0] - 20
        y_pos = height - 20
        cv2.putText(frame, text, (x_pos, y_pos), self.font,
                   self.font_scale, (0, 255, 255), self.thickness)
    
    def _draw_typing_text(self, frame):
        if not self.displayed_text:
            return
            
        text = f"LLM: {self.displayed_text}"
        height, width = frame.shape[:2]
        
        # Calculate text wrapping
        max_width = width - 40
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line_str = " ".join(current_line)
            text_size = cv2.getTextSize(line_str, self.font, self.font_scale, self.thickness)[0]
            if text_size[0] > max_width:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw text lines
        y_pos = 30
        line_height = 30
        
        # Draw background rectangle for readability
        bg_height = len(lines) * line_height + 10
        cv2.rectangle(frame, (0, 0), (width, bg_height), (0, 0, 0), -1)
        
        for line in lines:
            cv2.putText(frame, line, (20, y_pos), self.font,
                       self.font_scale, (255, 255, 0), self.thickness)
            y_pos += line_height
