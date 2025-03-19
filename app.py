import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time
import random

class HandCricketGame:
    def __init__(self):
        self.player_name = ""
        self.player_score = 0
        self.computer_score = 0
        self.current_innings = 1  # 1 for first innings, 2 for second
        self.batting_player = None  # "player" or "computer"
        self.game_state = "registration"  # registration, toss, selection, playing, finished
        self.last_player_input = None
        self.last_computer_input = None
        self.round_start_time = time.time()
        self.round_duration = 3  # seconds per round
        self.toss_result = None
        self.toss_winner = None
        self.is_batting = None

    def start_toss(self):
        self.toss_result = random.choice(["heads", "tails"])
        return self.toss_result

    def set_batting(self, is_batting):
        self.is_batting = is_batting
        if is_batting:
            self.batting_player = "player"
        else:
            self.batting_player = "computer"

    def update_score(self, player_fingers: int, computer_fingers: int):
        if player_fingers == computer_fingers:
            # Batsman is out
            if self.current_innings == 1:
                self.current_innings = 2
                self.batting_player = "computer" if self.batting_player == "player" else "player"
            else:
                self.game_state = "finished"
        else:
            # Add runs
            if self.batting_player == "player":
                self.player_score += player_fingers
            else:
                self.computer_score += computer_fingers

    def is_round_complete(self):
        return time.time() - self.round_start_time >= self.round_duration

class HandDetector:
    def __init__(self):
        self.kernel = np.ones((3,3), np.uint8)
        self.required_stable_frames = 3

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (320, 240))
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
        self.required_stable_frames = 3
        self.last_frame_time = time.time()
        self.frame_interval = 1/15
        self.skip_frames = 2
        self.frame_counter = 0

    def recv(self, frame):
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return frame
        
        self.last_frame_time = current_time
        self.frame_counter += 1
        
        if self.frame_counter % self.skip_frames != 0:
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        processed = self.hand_detector.preprocess_image(img)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
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
                        cv2.drawContours(img, [hand_contour], -1, (0, 255, 0), 2)
                        
                        if self.game.game_state == "playing":
                            cv2.putText(img, f'{self.game.player_name}: {fingers}', (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            if self.game.is_round_complete():
                                computer_fingers = np.random.randint(0, 7)
                                self.game.update_score(fingers, computer_fingers)
                                cv2.putText(img, f'Computer: {computer_fingers}', (50, 100), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                
                                if self.game.game_state == "finished":
                                    cv2.putText(img, "Game Over!", (50, 150), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                else:
                                    remaining_time = int(self.game.round_duration - 
                                                        (time.time() - self.game.round_start_time))
                                    cv2.putText(img, f'Round ends in: {remaining_time}s', (50, 150), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        elif self.game.game_state == "toss":
                            cv2.putText(img, f'Your guess: {fingers}', (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.set_page_config(page_title="Hand Cricket Game", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .game-container {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Hand Cricket Game")
    st.markdown("---")

    # Initialize session state
    if 'game' not in st.session_state:
        st.session_state.game = HandCricketGame()

    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            webrtc_streamer(
                key="hand-cricket",
                video_processor_factory=VideoProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )
        except Exception as e:
            st.error(f"Error initializing video stream: {str(e)}")
            st.info("Please make sure your camera is connected and you have granted camera permissions.")

    with col2:
        if st.session_state.game.game_state == "registration":
            st.subheader("Welcome to Hand Cricket!")
            player_name = st.text_input("Enter your name:")
            if st.button("Start Game") and player_name:
                st.session_state.game.player_name = player_name
                st.session_state.game.game_state = "toss"
                st.success(f"Welcome, {player_name}! Let's start with the toss!")

        elif st.session_state.game.game_state == "toss":
            st.subheader("Toss Phase")
            st.write("Show your fingers (0-6) to guess heads or tails")
            if st.button("Start Toss"):
                toss_result = st.session_state.game.start_toss()
                st.write(f"Coin shows: {toss_result}")
                st.session_state.game.game_state = "selection"

        elif st.session_state.game.game_state == "selection":
            st.subheader("Choose Batting or Bowling")
            if st.button("Batting"):
                st.session_state.game.set_batting(True)
                st.session_state.game.game_state = "playing"
                st.success("You're batting first!")
            if st.button("Bowling"):
                st.session_state.game.set_batting(False)
                st.session_state.game.game_state = "playing"
                st.success("You're bowling first!")

        elif st.session_state.game.game_state == "playing":
            st.subheader("Game Status")
            st.markdown(f"""
                - **Current Innings**: {st.session_state.game.current_innings}
                - **{st.session_state.game.player_name}'s Score**: {st.session_state.game.player_score}
                - **Computer's Score**: {st.session_state.game.computer_score}
                - **Batting**: {st.session_state.game.batting_player}
            """)
            st.write("Show your fingers (0-6) to play!")

        elif st.session_state.game.game_state == "finished":
            st.subheader("Game Over!")
            st.markdown(f"""
                ### Final Scores
                - **{st.session_state.game.player_name}**: {st.session_state.game.player_score}
                - **Computer**: {st.session_state.game.computer_score}
            """)
            if st.session_state.game.player_score > st.session_state.game.computer_score:
                st.success(f"Congratulations {st.session_state.game.player_name}! You won!")
            elif st.session_state.game.player_score < st.session_state.game.computer_score:
                st.error("Computer won!")
            else:
                st.info("It's a tie!")
            
            if st.button("Play Again"):
                st.session_state.game = HandCricketGame()
                st.experimental_rerun()

if __name__ == "__main__":
    main()
