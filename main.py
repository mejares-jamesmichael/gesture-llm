import cv2
import sys
import os
import time
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from camera import Camera
from gesture_detector import GestureDetector
from text_buffer import TextBuffer
from webhook_client import WebhookClient
from display import Display


def main():
    camera = Camera()
    detector = GestureDetector()
    buffer = TextBuffer(timeout=config.TIMEOUT_SECONDS)
    webhook = WebhookClient(config.WEBHOOK_URL, session_id=config.SESSION_ID)
    display = Display()
    
    # State variables
    llm_reply = ""
    current_gesture = None
    last_gesture = None
    gesture_stable_frames = 0
    last_sent_word = None
    last_sent_time = 0.0
    
    # Threading state
    is_sending = False
    
    print("Starting ASL to LLM...")
    print("Press 'q' to quit")
    print("Gestures:")
    for gesture, word in config.GESTURE_MAP.items():
        print(f"  {gesture}: {word}")
    
    def handle_webhook(sentence):
        nonlocal llm_reply, is_sending
        try:
            response = webhook.send(sentence)
            if response:
                # Handle array format: [{"output": "..."}]
                if isinstance(response, list) and len(response) > 0:
                    response = response[0]
                
                # Handle dict format: {"output": "..."}
                if isinstance(response, dict):
                    output = response.get('output')
                    if output:
                        llm_reply = str(output)
                    else:
                        text = response.get('text', response.get('message', str(response)))
                        llm_reply = str(text)
                    print(f"LLM Reply: {llm_reply}")
        except Exception as e:
            print(f"Error in webhook thread: {e}")
        finally:
            is_sending = False

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break
            
            landmarks, gesture = detector.process_frame(frame)
            
            if gesture:
                if gesture == last_gesture:
                    gesture_stable_frames += 1
                else:
                    gesture_stable_frames = 0
                    last_gesture = gesture
                
                if gesture_stable_frames >= 15:
                    current_gesture = gesture
                    
                    if gesture in config.GESTURE_MAP:
                        word = config.GESTURE_MAP[gesture]
                        now = time.time()
                        cooldown = config.GESTURE_COOLDOWN_SECONDS
                        if word != last_sent_word or (now - last_sent_time) > cooldown:
                            buffer.add_word(word)
                            last_sent_word = word
                            last_sent_time = now
                            llm_reply = ""
            else:
                gesture_stable_frames = 0
                if gesture_stable_frames < 3:
                    current_gesture = None
            
            # Check if we should send, but only if not already sending
            if buffer.should_send() and not is_sending:
                sentence = buffer.get_sentence()
                print(f"Sending: {sentence}")
                
                # Start sending in a separate thread
                is_sending = True
                sender_thread = threading.Thread(target=handle_webhook, args=(sentence,))
                sender_thread.daemon = True
                sender_thread.start()
                
                buffer.clear()
            
            timeout_remaining = buffer.get_timeout_remaining()
            
            # Add visual indicator if sending
            status_text = "Sending..." if is_sending else None
            if status_text:
                # You might want to pass this to display_frame in the future
                # For now, let's just use the existing display logic
                pass

            frame_with_overlays = display.display_frame(
                frame, 
                current_gesture,
                buffer.get_sentence() + (" [Sending...]" if is_sending else ""),
                llm_reply,
                timeout_remaining,
                landmarks
            )
            
            cv2.imshow('ASL to LLM', frame_with_overlays)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Closed!")


if __name__ == "__main__":
    main()
