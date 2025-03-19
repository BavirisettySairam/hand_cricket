import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time

class HandCricketGame:
    def __init__(self):
        self.player_score = 0
        self.computer_score = 0
        self.last_update_time = time.time()
        self.update_interval = 1.0  # Update game state every 1 second

    def update_score(self, player_fingers: int, computer_fingers: int):
        if time.time() - self.last_update_time >= self.update_interval:
            if player_fingers == computer_fingers:
                self.player_score += 1
                self.computer_score += 1
            elif player_fingers > computer_fingers:
                self.player_score += 1
            else:
                self.computer_score += 1
            self.last_update_time = time.time()

class HandDetector:
    def __init__(self):
        self.kernel = np.ones((3,3), np.uint8)  # Reduced kernel size for better performance
        self.required_stable_frames = 4  # Slightly reduced for faster response

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)

    def count_fingers(self, hand_contour: np.ndarray) -> int:
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)
        if defects is None:
            return 0
        
        finger_count = sum(
            np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / np.pi) < 90
            for s, e, f, d in defects[:, 0]
            for start, end, far in [(hand_contour[s][0], hand_contour[e][0], hand_contour[f][0])]
            for a, b, c in [(np.linalg.norm(np.array(start) - np.array(end)), 
                             np.linalg.norm(np.array(start) - np.array(far)), 
                             np.linalg.norm(np.array(end) - np.array(far)))]
        )
        return min(finger_count + 1, 6)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hand_detector = HandDetector()
        self.game = HandCricketGame()
        self.last_valid_input = None
        self.input_stable_count = 0
        self.required_stable_frames = 4  # Adjusted for real-time performance

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = self.hand_detector.preprocess_image(img)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 4000]
            if valid_contours:
                hand_contour = max(valid_contours, key=cv2.contourArea)
                fingers = self.hand_detector.count_fingers(hand_contour)

                if 0 <= fingers <= 6:
                    if fingers == self.last_valid_input:
                        self.input_stable_count += 1
                    else:
                        self.input_stable_count = 0
                        self.last_valid_input = fingers

                    if self.input_stable_count >= self.required_stable_frames:
                        computer_fingers = np.random.randint(0, 7)
                        self.game.update_score(fingers, computer_fingers)

                        cv2.putText(img, f'You: {fingers}', (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f'Comp: {computer_fingers}', (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

def main():
    st.set_page_config(page_title="Hand Cricket Game", layout="wide")

    st.title("Hand Cricket Game")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        webrtc_streamer(key="hand-cricket", video_processor_factory=VideoProcessor)

    with col2:
        st.subheader("Instructions")
        st.markdown("""
            1. Show 0-6 fingers.
            2. Hold for a moment to register.
            3. Higher number wins.
        """)
        st.button("Start New Game", on_click=lambda: st.success("Game Started!"))
        st.button("Reset Scores", on_click=lambda: st.success("Scores Reset!"))

if __name__ == "__main__":
    main()
