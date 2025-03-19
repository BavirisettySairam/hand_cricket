# Hand Cricket Game

A real-time hand cricket game using computer vision and Streamlit.

## Features
- Real-time finger detection using OpenCV
- Interactive game interface
- Score tracking
- Computer opponent

## Local Development
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Deployment
This application is deployed on Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select your forked repository
5. Deploy!

## Requirements
- Python 3.8+
- OpenCV
- Streamlit
- Streamlit WebRTC

## How to Play

1. Show your fingers (0-6) to the camera
2. Wait for the count to stabilize
3. The computer will make its move
4. Whoever shows more fingers wins the round

## Technical Details

- The application uses OpenCV for hand detection and finger counting
- Finger detection is stabilized to prevent false readings
- The UI is built with Streamlit and styled with custom CSS
- WebRTC is used for real-time video streaming 