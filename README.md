# Gesture Translator

A real-time system that translates hand gestures into spoken words or text, enabling communication for people with speech impairments.

## Features

- Real-time hand gesture detection and tracking
- Translation of gestures to text and speech
- Support for custom gesture training
- Multiple output modes (Text/Speech/Emoji)
- User-friendly web interface

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Allow camera access when prompted
2. Select your preferred output mode (Text/Speech/Emoji)
3. Perform gestures in front of the camera
4. View the translations in real-time

## Custom Gesture Training

1. Go to the "Train Custom Gestures" section
2. Record your gesture while holding the "Record" button
3. Label your gesture
4. Save the gesture to your custom collection

## Technologies Used

- Streamlit: Web interface
- MediaPipe: Hand tracking and gesture detection
- OpenCV: Image processing
- TensorFlow: Gesture classification
- gTTS: Text-to-Speech conversion 