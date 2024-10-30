import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
import webbrowser
from tkinter import simpledialog
import tkinter as tk
from tkinter import messagebox, ttk
import requests
import ttkbootstrap as ttkbs
from threading import Thread
import json  # To store configurations
import subprocess  # To run applications

# Imports for firebase integration
import firebase_admin
from firebase_admin import credentials, auth

from ttkbootstrap.scrolled import ScrolledFrame
from ttkbootstrap.toast import ToastNotification
from ttkbootstrap.tooltip import ToolTip
from ttkbootstrap.widgets import DateEntry, Floodgauge, Meter

# Initialize Firebase with the certificate
cred = credentials.Certificate("config/capstone-project-1a804-firebase-adminsdk-46w3i-a1cb2293c8.json")
firebase_admin.initialize_app(cred)

class HandMouseController:
    def __init__(self, gesture_actions):
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
        self.click_threshold = 0.1  # seconds
        self.drag_threshold = 0.4  # seconds
        self.button_threshold = 10  # Threshold for imaginary button

        # Screenshot state variables
        self.screenshot_cooldown = 2  # seconds
        self.last_screenshot_time = 0

        # Gesture actions
        self.gesture_actions = gesture_actions
        self.last_gesture_time = 0  # To prevent multiple triggers
        self.gesture_cooldown = 2  # seconds

        # Gesture hold tracking
        self.current_gesture = None
        self.gesture_start_time = None

        self.fist_detected_seconds = 0  # Counter for the seconds the fist is detected
        pass

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
        for id, lm_point in enumerate(hand_landmarks.landmark):
            cx = int(lm_point.x * self.cam_width)
            cy = int(lm_point.y * self.cam_height)
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
        # Calculate the midpoint between thumb and index finger
        midpoint = (
            (thumb_tip[0] + index_tip[0]) // 2,
            (thumb_tip[1] + index_tip[1]) // 2
        )
        # No need to define button_threshold here

        # Calculate distances from thumb tip and index tip to the midpoint
        thumb_distance = np.linalg.norm(np.array(thumb_tip) - np.array(midpoint))
        index_distance = np.linalg.norm(np.array(index_tip) - np.array(midpoint))

        print(f"Thumb distance to midpoint: {thumb_distance}")
        print(f"Index distance to midpoint: {index_distance}")

        # Check if both fingers are close to the imaginary button
        if thumb_distance < self.button_threshold and index_distance < self.button_threshold:
            print("Both fingers are touching the imaginary button")  # Debugging
            if self.touch_start_time is None:
                self.touch_start_time = time.time()
                self.dragging = False
                print("Touch start time set")  # Debugging
            else:
                elapsed_time = time.time() - self.touch_start_time
                print(f"Elapsed time: {elapsed_time}")  # Debugging
                if elapsed_time > self.drag_threshold and not self.dragging:
                    # Start dragging
                    print("Starting drag")  # Debugging
                    pyautogui.mouseDown()
                    self.dragging = True
        else:
            print("Fingers are not touching the imaginary button")  # Debugging
            if self.touch_start_time is not None:
                elapsed_time = time.time() - self.touch_start_time
                print(f"Elapsed time: {elapsed_time}")  # Debugging
                if elapsed_time < self.drag_threshold and not self.dragging:
                    # Perform click
                    print("Performing click")  # Debugging
                    pyautogui.click()
                elif self.dragging:
                    # Stop dragging
                    print("Stopping drag")  # Debugging
                    pyautogui.mouseUp()
                    self.dragging = False
            self.touch_start_time = None



    def fingers_up(self):
        """Determines which fingers are up and returns a list."""
        fingers = []
        threshold = 5  # Adjust as needed

        # Thumb detection using x-coordinates
        thumb_tip_x = self.lm_list[4][0]
        thumb_mcp_x = self.lm_list[2][0]
        if self.hand_label == 'Right':
            if thumb_tip_x > thumb_mcp_x + threshold:
                fingers.append(1)  # Thumb is up
            else:
                fingers.append(0)  # Thumb is down
        else:  # Left hand
            if thumb_tip_x < thumb_mcp_x - threshold:
                fingers.append(1)  # Thumb is up
            else:
                fingers.append(0)  # Thumb is down

        # Fingers (Index to Pinky)
        tips_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]  # PIP joints

        for tip_id, pip_id in zip(tips_ids, pip_ids):
            tip_y = self.lm_list[tip_id][1]
            pip_y = self.lm_list[pip_id][1]
            if tip_y < pip_y - threshold:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down

        print(f"Fingers Up: {fingers}")  # Debugging
        return fingers  # [Thumb, Index, Middle, Ring, Pinky]





    def detect_middle_finger_gesture(self):
        """Detects if the middle finger is up and other fingers are down."""
        fingers = self.fingers_up()
        print(f"Middle Finger Gesture Detection - Fingers: {fingers}")  # Debugging statement
        # Exclude thumb from consideration
        if fingers[1:] == [0, 1, 0, 0]:  # [Index, Middle, Ring, Pinky]
            return True
        else:
            return False
    
    def detect_curled_fist(self):
        """Detects if all fingers are down."""
        fingers = self.fingers_up()
        # Exclude thumb from consideration
        if fingers == [0, 0, 0, 0, 0]:  # [Thumb, Index, Middle, Ring, Pinky]
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

    def detect_custom_gestures(self):
        """Detects specific finger patterns and performs assigned action after holding the gesture."""
        fingers = self.fingers_up()
        # Exclude thumb for gesture detection
        fingers_pattern = tuple(fingers[1:])  # Index to Pinky

        # Define the gesture patterns
        gesture_patterns = {
            (1, 1, 0, 0): 2,  # 2 fingers up (index and middle)
            (1, 1, 1, 0): 3,  # 3 fingers up (index, middle, ring)
            (1, 1, 1, 1): 4,  # 4 fingers up (index, middle, ring, pinky)
        }

        current_time = time.time()

        if fingers_pattern in gesture_patterns:
            gesture = str(gesture_patterns[fingers_pattern])  # Convert gesture number to string
            print(f"Detected Gesture: {gesture}")
            print(f"Type of Current Gesture: {type(self.current_gesture)}, Type of Detected Gesture: {type(gesture)}")
            if gesture in self.gesture_actions:
                action = self.gesture_actions[gesture]
                if self.current_gesture != gesture:
                    # New gesture detected
                    self.current_gesture = gesture
                    self.gesture_start_time = current_time
                else:
                    # Same gesture, check if held long enough
                    elapsed_time = current_time - self.gesture_start_time
                    if elapsed_time >= 1:  # Hold for 1 second
                        if current_time - self.last_gesture_time > self.gesture_cooldown:
                            self.execute_action(action)
                            self.last_gesture_time = current_time
                            # Reset gesture tracking
                            self.current_gesture = None
                            self.gesture_start_time = None
        else:
            # Gesture not recognized or fingers not held in position
            self.current_gesture = None
            self.gesture_start_time = None




    def execute_action(self, action):
        """Executes the assigned action for a gesture."""
        print(f"Executing action: {action}")
        normalized_action = action.strip().lower()
        try:
            if normalized_action == "open snipping tool":
                try:
                    subprocess.Popen('snippingtool')
                except FileNotFoundError:
                    subprocess.Popen('SnippingTool.exe')
            elif normalized_action == "open calculator":
                subprocess.Popen('calc')
            elif normalized_action == "open notepad":
                subprocess.Popen('notepad')
            elif normalized_action == "lock screen":
                pyautogui.hotkey('win', 'l')
            elif normalized_action == "play/pause media":
                pyautogui.press('playpause')
            elif normalized_action == "next track":
                pyautogui.press('nexttrack')
            elif normalized_action == "previous track":
                pyautogui.press('prevtrack')
            elif normalized_action == "volume up":
                pyautogui.press('volumeup')
            elif normalized_action == "volume down":
                pyautogui.press('volumedown')
            elif normalized_action == "mute/unmute":
                pyautogui.press('volumemute')
            elif action.startswith('http://') or action.startswith('https://'):
                webbrowser.open(action)
            else:
                # Attempt to execute the action as a command
                subprocess.Popen(action, shell=True)
        except Exception as e:
            print(f"Error executing action '{action}': {e}")





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
                    hand_handedness = results.multi_handedness[0]
                    label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                    self.hand_label = label

                    thumb_tip, index_tip = self.get_landmark_positions(hand_landmarks)

                    # Move cursor
                    self.move_cursor(index_tip)

                    # Handle click and drag first
                    self.handle_gesture(thumb_tip, index_tip)

                    # Calculate midpoint for visualization
                    midpoint = (
                        (thumb_tip[0] + index_tip[0]) // 2,
                        (thumb_tip[1] + index_tip[1]) // 2
                    )

                    # Draw the imaginary button (for visualization)
                    cv2.circle(img, midpoint, self.button_threshold, (0, 255, 0), 2)  # Green circle

                    # Detect other gestures
                    if self.detect_middle_finger_gesture():
                        print("Middle finger gesture detected - Exiting")
                        break
                    elif self.detect_shaka_sign():
                        self.take_screenshot()
                    elif self.detect_curled_fist():
                        self.fist_detected_seconds += 0.1  # Increment counter if fist is detected
                        if self.fist_detected_seconds >= 42:  # Easter Egg
                            webbrowser.open("https://blacklivesmatter.com/")
                            self.fist_detected_seconds = 0
                    else:
                        self.detect_custom_gestures()

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




# GUI functions for configuration
def configure_gestures():
    """Opens a GUI for users to assign actions to gestures."""
    config_window = tk.Toplevel()  # Use Toplevel instead of ttkbs.Window
    config_window.title('Configure Gestures')
    config_window.geometry('640x480')

    # Title for the configuration window
    title_label = ttkbs.Label(config_window, text='Configure Gestures', font=('Arial', 24, 'bold'))
    title_label.pack(pady=20)

    gestures = [2, 3, 4]  # Number of fingers up to configure
    actions = [
        "Open Snipping Tool",
        "Open Calculator",
        "Open Notepad",
        "Lock Screen",
        "Play/Pause Media",
        "Next Track",
        "Previous Track",
        "Volume Up",
        "Volume Down",
        "Mute/Unmute",
        "Other..."
    ]

    default_gesture_actions = {
        "2": "Open Calculator",
        "3": "Open Snipping Tool",
        "4": "Open Notepad"
    }

    # Load existing configuration if available
    config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gesture_config.json')
    if os.path.exists(config_file) and os.path.getsize(config_file) > 0:
        with open(config_file, 'r') as f:
            try:
                gesture_actions = json.load(f)
            except json.JSONDecodeError:
                print("Invalid JSON in gesture_config.json. Loading default configuration.")
                messagebox.showwarning("Invalid Configuration", "The gesture configuration file is invalid. Loading default configuration.")
                gesture_actions = default_gesture_actions.copy()
                # Save default configuration back to the file
                with open(config_file, 'w') as fw:
                    json.dump(gesture_actions, fw)
    else:
        # File doesn't exist or is empty, use default configuration
        gesture_actions = default_gesture_actions.copy()
        # Save default configuration to the file
        with open(config_file, 'w') as f:
            json.dump(gesture_actions, f)

    dropdowns = {}

    def save_configuration():
        # Save the selected actions
        for g in gestures:
            selected_action = dropdowns[g].get()
            if selected_action == "Other...":
                # Prompt user for custom input
                custom_action = simpledialog.askstring("Custom Action", f"Enter custom action for gesture {g}:")
                if custom_action:
                    gesture_actions[str(g)] = custom_action
                else:
                    # If no input, default to the previous action
                    gesture_actions[str(g)] = gesture_actions.get(str(g), actions[0])
                    dropdowns[g].set(gesture_actions[str(g)])
            else:
                gesture_actions[str(g)] = selected_action
        try:
            with open(config_file, 'w') as f:
                json.dump(gesture_actions, f)
            toast = ToastNotification("Configuration Saved", "Gestures configuration saved successfully.", duration=3000, bootstyle='success', position = (50, 50, 'ne'))
        except Exception as e:
            toast = ToastNotification("Configuration Error", f"Failed to save configurations: {e}", duration=3000, bootstyle='danger', position = (50, 50, 'ne'))
            print(f"Failed to save configurations: {e}")
        toast.show_toast()
        config_window.destroy()

    for g in gestures:
        label = ttkbs.Label(config_window, text=f'Gesture: {g} Fingers Up', font=('Arial', 12))
        label.pack(pady=5)

        default_value = gesture_actions.get(str(g), actions[0])
        dropdown = ttkbs.Combobox(config_window, values=actions, state='readonly', bootstyle='primary')
        dropdown.set(default_value)
        dropdown.pack(pady=5)
        dropdowns[g] = dropdown

    save_button = ttkbs.Button(config_window, text='Save', bootstyle='primary', command=save_configuration)
    save_button.pack(pady=20)

    config_window.mainloop()

# Define the login function
def login(email, password):
    api_key = "AIzaSyCp1nDe4uciVKuJn0G-Io8JVQ5Tsz869OM"  # Replace with your actual API key
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("User authenticated successfully:", email)
        return True
    else:
        print("Failed to authenticate:", response.json().get("error", {}).get("message"))
        return False

def handle_login_gui(username, password):
    # Handle login using Firebase
    if login(username, password):
        # Login successful, open main application window
        open_main_window(username)
    else:
        messagebox.showerror("Login Failed", "Invalid username or password. Please try again.")

def open_main_window(username):
    # Hide the login window instead of destroying it
    window.withdraw()

    # Create the main application window
    app_window = tk.Toplevel()
    app_window.title('Hand Gesture Mouse Controller')
    app_window.geometry('640x480')

    label = ttkbs.Label(app_window, text=f'Welcome, {username}!', font=('Arial', 24, 'bold'))
    label.pack(pady=20)

    start_button = ttkbs.Button(app_window, text='Start Hand Gesture Controller', bootstyle='primary',
                                command=lambda: start_hand_mouse_controller(app_window))
    start_button.pack(pady=20)

    config_button = ttkbs.Button(app_window, text='Configure Gestures', bootstyle='primary', command=configure_gestures)
    config_button.pack(pady=10)

    view_button = ttkbs.Button(app_window, text='View Hand Gesture Guide', bootstyle='primary', command=view_gestures)
    view_button.pack(pady=10)

    def on_app_window_close():
        # Make sure the main window is destroyed when the app window is closed
        app_window.destroy()
        window.destroy()

    # Bind the close event to ensure the main window is properly destroyed
    app_window.protocol("WM_DELETE_WINDOW", on_app_window_close)

    app_window.mainloop()

def start_hand_mouse_controller(app_window):
    # Hide the app window temporarily while the loading window is shown
    app_window.withdraw()

    # Create a loading window
    loading_window = tk.Toplevel(app_window)
    loading_window.title('Loading')
    loading_window.geometry('400x125')

    label = ttkbs.Label(loading_window, text='Starting camera...', font=('Arial', 18, 'bold'))
    label.pack(pady=20)

    progress_bar = ttkbs.Progressbar(loading_window, mode='indeterminate', bootstyle='primary-striped', length=300)
    progress_bar.pack(pady=10)
    progress_bar.start()

    # Function to start the hand mouse controller
    def start_controller():
        # Load gesture configurations
        config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gesture_config.json')
        print(f"Configuration file path (loading): {config_file}")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                gesture_actions = json.load(f)
        else:
            gesture_actions = {}

        # Initialize the HandMouseController
        controller = HandMouseController(gesture_actions)
        loading_window.destroy()  # Close the loading window
        app_window.deiconify()  # Show the app window again after loading
        controller.run()
        app_window.destroy()  # Destroy the app window when the controller finishes

    # Start the controller in a separate thread to avoid blocking the UI
    Thread(target=start_controller).start()


def view_gestures():
    # Create a gestures guide window
    gestures_window = tk.Toplevel()
    gestures_window.title('Hand Gesture Guide')
    gestures_window.geometry('640x480')

    label = ttkbs.Label(gestures_window, text='Hand Gesture Guide', font=('Arial', 24, 'bold'))
    label.pack(pady=20)

    gesture_frame = ttkbs.Frame(gestures_window, padding=10)
    gesture_frame.pack(expand=True, fill=tk.BOTH)

    # Load the gesture configurations from file
    config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gesture_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            gesture_actions = json.load(f)
    else:
        gesture_actions = {}

    gesture_descriptions = {
        "1 Finger Up": "Move Cursor",
        "2 Fingers Up": gesture_actions.get("2", "Not Configured"),
        "3 Fingers Up": gesture_actions.get("3", "Not Configured"),
        "4 Fingers Up": gesture_actions.get("4", "Not Configured"),
        "Thumb and Index Touching": "Click/Drag",
        "Middle Finger Up": "Exit Application",
        "Shaka Sign": "Take Screenshot"
    }

    for gesture, description in gesture_descriptions.items():
        label = ttkbs.Label(gesture_frame, text=f'{gesture}: {description}', font=('Arial', 12))
        label.pack(pady=5, anchor=tk.W)

    gestures_window.mainloop()


if __name__ == "__main__":
    window = ttkbs.Window(themename='darkly')  # Main window
    window.title('Hand Gesture Mouse Controller')
    window.geometry('640x480')

    label = ttkbs.Label(window, text='Login', font=('Arial', 24, 'bold'))
    label.pack(pady=20)

    username_label = ttkbs.Label(window, text='Username', font=('Arial', 14))
    username_label.pack(pady=5)

    username_entry = ttkbs.Entry(window, font=('Arial', 12))
    username_entry.pack(pady=5)

    password_label = ttkbs.Label(window, text='Password', font=('Arial', 14))
    password_label.pack(pady=5)

    password_entry = ttkbs.Entry(window, font=('Arial', 12), show='*')
    password_entry.pack(pady=5)

    login_button = ttkbs.Button(window, text='Login', bootstyle='primary',
                                command=lambda: handle_login_gui(username_entry.get(), password_entry.get()))
    login_button.pack(padx=10, pady=10)

    window.mainloop()