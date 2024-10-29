# HandTracking Project

This project uses hand gestures to control various functionalities of a computer. By leveraging the power of OpenCV, MediaPipe, and TensorFlow Lite, the project tracks hand movements and interprets them into actions like moving the cursor, taking screenshots, left-clicking, and quitting the application.

## Features

The hand gestures recognized in this project trigger specific computer actions:

- **One Finger:** Moves the cursor around the screen.
- **Index Finger and Thumb Pinch:** Simulates a left-click.
- **Shaka Gesture:** Captures a screenshot of the current screen.
- **Middle Finger Gesture:** Quits the application.
- **Customizable Gestures for Two, Three, and Four fingers up!**

## Requirements

To get started with this project, ensure that the following dependencies are installed:

```
opencv-python
mediapipe
numpy
pyautogui
```

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Installation and Setup

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Dominickator/HandTrackingProject.git
   ```

2. **Install Dependencies**  
   Navigate to the project directory and install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project**  
   Execute the `main.py` file to start the hand tracking and gesture recognition:
   ```bash
   python main.py
   ```

## How it Works

The project uses a hand-tracking module based on the MediaPipe library. It tracks key landmarks of the hand, which are then processed to detect specific gestures. These gestures are mapped to actions on the computer:

- **Moving the Cursor:** By holding up a single finger, you can move the cursor in real-time.
- **Left Click:** Pinching together the index finger and thumb will perform a left-click.
- **Screenshot:** The "Shaka" hand gesture is interpreted as a screenshot command.
- **Quit Application:** Raising the middle finger will close the application.

## Contribution

Feel free to open issues or contribute to this project by submitting a pull request.
