import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os

class HandMouseController:
    def __init__(self):
        # Initialize Mediapipe Hand model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen and camera dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        # Variables for cursor movement
        self.smoothening = 5
        self.prev_x, self.prev_y = 0, 0

        # Gesture state variables
        self.dragging = False
        self.touch_start_time = None
        self.click_threshold = 0.3  # seconds
        self.drag_threshold = 0.3  # seconds

        # Screenshot state variables
        self.screenshot_cooldown = 2  # seconds
        self.last_screenshot_time = 0

    def get_landmark_positions(self, hand_landmarks):
        """Extracts landmark positions and returns key fingertip coordinates."""
        lm = hand_landmarks.landmark

        # Get coordinates of necessary landmarks
        thumb_tip = (int(lm[4].x * self.cam_width), int(lm[4].y * self.cam_height))
        index_tip = (int(lm[8].x * self.cam_width), int(lm[8].y * self.cam_height))

        # For finger detection, store all landmarks
        self.lm_list = []
        for id, lm in enumerate(hand_landmarks.landmark):
            cx = int(lm.x * self.cam_width)
            cy = int(lm.y * self.cam_height)
            self.lm_list.append((cx, cy))

        return thumb_tip, index_tip

    def is_fingers_touching(self, finger1, finger2, threshold=20):
        """Checks if two fingertips are touching based on a distance threshold."""
        return np.linalg.norm(np.array(finger1) - np.array(finger2)) < threshold

    def move_cursor(self, index_tip):
        """Moves the cursor smoothly based on the index fingertip position."""
        # Map the fingertip position to the screen size
        x = np.interp(index_tip[0], (100, self.cam_width - 100), (0, self.screen_width))
        y = np.interp(index_tip[1], (100, self.cam_height - 100), (0, self.screen_height))

        # Smooth the cursor movement
        curr_x = self.prev_x + (x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (y - self.prev_y) / self.smoothening

        # Move the cursor
        pyautogui.moveTo(self.screen_width - curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y

    def handle_gesture(self, thumb_tip, index_tip):
        """Handles click and drag events based on thumb and index fingertip positions."""
        # Check if only thumb and index finger are up
        fingers = self.fingers_up()
        if fingers == [1, 1, 0, 0, 0]:
            if self.is_fingers_touching(thumb_tip, index_tip):
                if self.touch_start_time is None:
                    self.touch_start_time = time.time()
                    self.dragging = False
                else:
                    elapsed_time = time.time() - self.touch_start_time
                    if elapsed_time > self.drag_threshold and not self.dragging:
                        # Start dragging
                        pyautogui.mouseDown()
                        self.dragging = True
            else:
                if self.touch_start_time is not None:
                    elapsed_time = time.time() - self.touch_start_time
                    if elapsed_time < self.click_threshold and not self.dragging:
                        # Perform click
                        pyautogui.click()
                    elif self.dragging:
                        # Stop dragging
                        pyautogui.mouseUp()
                        self.dragging = False
                self.touch_start_time = None
        else:
            # If other fingers are up, reset dragging state
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            self.touch_start_time = None

    def fingers_up(self):
        """Determines which fingers are up and returns a list."""
        fingers = []

        # Thumb
        if self.lm_list[4][0] > self.lm_list[3][0]:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down

        # Fingers (Index to Pinky)
        tips_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]

        for tip_id, pip_id in zip(tips_ids, pip_ids):
            if self.lm_list[tip_id][1] < self.lm_list[pip_id][1]:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down

        return fingers  # [Thumb, Index, Middle, Ring, Pinky]

    def detect_middle_finger_gesture(self):
        """Detects if the middle finger is up and other fingers are down."""
        fingers = self.fingers_up()
        # Check if only the middle finger is up
        if fingers == [0, 0, 1, 0, 0]:
            return True
        else:
            return False

    def detect_shaka_sign(self):
        """Detects if the user is making the hang loose (Shaka) sign."""
        fingers = self.fingers_up()

        # Check for Shaka sign: Thumb and Pinky up, other fingers down
        if fingers == [1, 0, 0, 0, 1]:
            # Additional check: Thumb and Pinky spread apart
            thumb_tip = np.array(self.lm_list[4])
            pinky_tip = np.array(self.lm_list[20])
            distance = np.linalg.norm(thumb_tip - pinky_tip)
            if distance > 100:  # Adjust this threshold as needed
                return True
        return False

    def take_screenshot(self):
        """Takes a screenshot and saves it to the Downloads folder."""
        current_time = time.time()
        if current_time - self.last_screenshot_time > self.screenshot_cooldown:
            # Get the Downloads folder path
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')

            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(downloads_path, filename)

            # Take screenshot and save
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)

            print(f"Screenshot saved to {filepath}")
            self.last_screenshot_time = current_time

    def run(self):
        """Main loop to process video frames and control the cursor."""
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    thumb_tip, index_tip = self.get_landmark_positions(hand_landmarks)

                    # Move cursor
                    self.move_cursor(index_tip)

                    # Detect middle finger gesture
                    if self.detect_middle_finger_gesture():
                        print("Middle finger gesture detected - Exiting")
                        break

                    # Detect Shaka sign for screenshot
                    elif self.detect_shaka_sign():
                        self.take_screenshot()

                    # Handle click and drag
                    else:
                        self.handle_gesture(thumb_tip, index_tip)

                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Display the image
                cv2.imshow("Hand Tracking Mouse", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Program interrupted by user.")
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandMouseController()
    controller.run()
