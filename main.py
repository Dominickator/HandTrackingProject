import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
import tkinter as tk
from tkinter import messagebox, ttk
from threading import Thread

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

        if not self.cap.isOpened():
            self.display_no_camera_popup()
            return

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

    def display_no_camera_popup(self):
        """Displays a popup window if no camera is detected and closes the application after a delay."""
        popup_window = tk.Tk()
        popup_window.withdraw()  # Hide the root window

        # Create a messagebox with a delay
        def close_after_delay():
            time.sleep(10)  # Wait for 10 seconds
            popup_window.quit()  # Close the window

        # Show the error message
        messagebox.showerror("Camera Not Found", "No camera detected. The application will close in 10 seconds or press OK to exit now.")
        
        # Start the delay in a separate thread so it doesn't block the UI
        delay_thread = Thread(target=close_after_delay)
        delay_thread.start()

        # Start the Tkinter event loop to keep the popup alive
        popup_window.mainloop()

        # Ensure the application exits gracefully
        os._exit(1)


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
        fingers = self.fingers_up()
        # Check if index finger is up, and middle, ring, and pinky fingers are down
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # Check if thumb and index finger are touching
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

        # Thumb (we'll ignore thumb detection for now)
        fingers.append(0)  # Set thumb as down by default or adjust as needed

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

        # Thumb detection adjusted for consistency
        thumb_up = self.lm_list[4][0] > self.lm_list[2][0]
        fingers[0] = 1 if thumb_up else 0

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

def handle_login_gui(username):
    # Destroy the login window and create the main application window
    login_window.destroy()
    app_window = tk.Tk()
    app_window.title("Hand Gesture Mouse Controller")
    app_window.geometry("640x480")  # Set the resolution of the main window

    style = ttk.Style()
    style.configure("TLabel", font=("Verdana", 15, "bold"))
    style.configure("TButton", font=("Verdana", 12), padding=10)

    # Create a label to display the username
    ttk.Label(app_window, text=f"Welcome, {username}!", padding=(20, 10)).pack()

    # Create a button to start the hand gesture controller
    start_button = ttk.Button(app_window, text="Start Hand Gesture Controller", command=lambda: start_hand_mouse_controller(app_window))
    start_button.pack(pady=20)

    # Create a button to view the hand gesture guide
    view_button = ttk.Button(app_window, text="View Hand Gesture Guide", command=view_gestures)
    view_button.pack(pady=10)

    app_window.mainloop()

def start_hand_mouse_controller(app_window):
    # Create a loading window
    loading_window = tk.Toplevel(app_window)
    loading_window.title("Loading")
    loading_window.geometry("300x100")
    
    # Create a label to indicate loading
    ttk.Label(loading_window, text="Starting camera...", padding=(20, 10)).pack()
    
    # Create a progress bar
    progress_bar = ttk.Progressbar(loading_window, mode='indeterminate')
    progress_bar.pack(pady=20, padx=20, fill=tk.X)
    progress_bar.start()
    
    # Function to start the hand mouse controller
    def start_controller():
        controller = HandMouseController()
        loading_window.destroy()
        controller.run()
        app_window.destroy()
    
    # Start the controller in a separate thread to avoid blocking the UI
    Thread(target=start_controller).start()

def view_gestures():
    # Create a new window to display the gesture guide
    gestures_window = tk.Toplevel()
    gestures_window.title("Hand Gesture Guide")
    gestures_window.geometry("640x480")  # Set the resolution of the gestures window

    # Show the gesture guide image
    gestures_image = tk.PhotoImage(file="gestures.png")
    gestures_label = ttk.Label(gestures_window, image=gestures_image)
    gestures_label.pack()

    gestures_window.mainloop()

if __name__ == "__main__":
    # Create the login screen
    login_window = tk.Tk()
    login_window.title("Login Screen")
    login_window.geometry("640x480")  # Set the resolution of the login window

    style = ttk.Style()
    style.configure("TLabel", font=("Verdana", 12))
    style.configure("TButton", font=("Verdana", 12), padding=10)
    style.configure("TEntry", font=("Verdana", 12))

    frame = ttk.Frame(login_window, padding="20")
    frame.pack(expand=True)

    ttk.Label(frame, text="Username:").pack(pady=10)
    username_entry = ttk.Entry(frame, width=40)
    username_entry.pack(pady=10, fill=tk.X)

    ttk.Label(frame, text="Password:").pack(pady=10)
    password_entry = ttk.Entry(frame, width=40, show="*")
    password_entry.pack(pady=10, fill=tk.X)

    login_button = ttk.Button(frame, text="Login", command=lambda: handle_login_gui(username_entry.get()))
    login_button.pack(pady=20)

    login_window.mainloop()
