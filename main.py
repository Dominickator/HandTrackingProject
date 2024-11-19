import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
import webbrowser
import requests
from threading import Thread, Event
from win11toast import toast
import json  # To store configurations
import subprocess  # To run applications
from PIL import Image

import sounddevice
import speech_to_text as stt

# Imports for firebase integration
import firebase_admin
from firebase_admin import credentials, auth

import customtkinter as ctk


#tmp = os.listdir("HandTrackingProject/config")


# Initialize CustomTkinter App
ctk.set_appearance_mode("dark")  # Modes: "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


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

        # Show the error message
        toast("Camera Not Found", "No camera detected. The application will close in 10 seconds or press OK to exit now.")

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

        # Check if both fingers are close to the imaginary button
        if thumb_distance < self.button_threshold and index_distance < self.button_threshold:
            if self.touch_start_time is None:
                self.touch_start_time = time.time()
                self.dragging = False
            else:
                elapsed_time = time.time() - self.touch_start_time
                print(f"Elapsed time: {elapsed_time}")  # Debugging
                if elapsed_time > self.drag_threshold and not self.dragging:
                    # Start dragging
                    pyautogui.mouseDown()
                    self.dragging = True
        else:
            if self.touch_start_time is not None:
                elapsed_time = time.time() - self.touch_start_time
                print(f"Elapsed time: {elapsed_time}")  # Debugging
                if elapsed_time < self.drag_threshold and not self.dragging:
                    # Perform click
                    pyautogui.click()
                elif self.dragging:
                    # Stop dragging
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

        return fingers  # [Thumb, Index, Middle, Ring, Pinky]





    def detect_middle_finger_gesture(self):
        """Detects if the middle finger is up and other fingers are down."""
        fingers = self.fingers_up()
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
            elif normalized_action == "play/pause media":
                pyautogui.press('playpause')
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

def activate_stt():
    global stop_event
    stop_event = Event()
    run_stt(stop_event)

def run_stt(stop_event:Event):
    #Loading which sound device to use from config file
    config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stt_config.json')
    with open(config_file, 'r') as f:
        try:
            stt_data = json.load(f)
            device = stt_data['device']
            toast("Configuration Loaded", f"Speech-To-Text Configuration loaded successfully. Using device: {device}")
        except json.JSONDecodeError:
            toast("Configuration Error", f"Failed to load Speech-To-Text Configuration. Is it configured correctly?")
    global stt_thread
    stt_thread = Thread(name="stt_thread",target=stt.run_stt, daemon=True, args=(device, stop_event))
    stt_thread.start()

def deactivate_stt():
    global stt_thread
    global stop_event
    stop_event.set()
    stt_thread.join()

class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.exists("first_time.txt"):
            self.open_login_window()
        else:
            self.first_time_user()
    
    def open_login_window(self, screenshot_window=None):
        login_window = ctk.CTk()
        login_window.title("Login")
        login_window.geometry("800x600")

        if screenshot_window:
            screenshot_window.destroy()

            # Create the first_time.txt file
            with open("first_time.txt", "w") as f:
                f.write("First Time")

        # Create a frame for the login form
        login_frame = ctk.CTkFrame(login_window)
        login_frame.pack(pady=100)

        # Create a title label
        title_label = ctk.CTkLabel(login_frame, text="Login", font=("Arial", 28, "bold"))
        title_label.pack(pady=20)

        # Create a username label and entry
        username_label = ctk.CTkLabel(login_frame, text="Username", font=("Arial", 16))
        username_label.pack(pady=5, padx=75)
        username_entry = ctk.CTkEntry(login_frame, font=("Arial", 16), width=200)
        username_entry.pack(pady=10, padx=75)

        # Create a password label and entry
        password_label = ctk.CTkLabel(login_frame, text="Password", font=("Arial", 16))
        password_label.pack(pady=5, padx=75)
        password_entry = ctk.CTkEntry(login_frame, show="*", font=("Arial", 16), width=200)
        password_entry.pack(pady=10, padx=75)

        # Create a login button
        def login():
            email = username_entry.get()
            password = password_entry.get()
            api_key = "AIzaSyCp1nDe4uciVKuJn0G-Io8JVQ5Tsz869OM"
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                # Create a toast on a separate thread
                Thread(target=lambda: toast("Login Successful", "You have been logged in successfully!")).start()
                return True
            else:
                return False
        
        def handle_login():
            success = login()
            if success:
                login_window.withdraw()
                self.open_main_window()

        login_button = ctk.CTkButton(login_frame, text="Login", font=("Arial", 16), command=handle_login)
        login_button.pack(pady=20)

        login_window.resizable(False, False)
        login_window.mainloop()
    
    def check_new_user(self):
        # Check for the existence of the first_time.txt file
        if not os.path.exists("first_time.txt"):
            # If the file does not exist, create it and return True
            with open("first_time.txt", "w") as f:
                f.write("First Time")
            return True
        else:
            # If the file exists, return False
            return False

    def config_stt(self):
        """Opens a new window to configure Speech-To-Text settings."""
        config_window = ctk.CTkToplevel()  # Use CTkToplevel instead of Toplevel
        config_window.title('Configure Speech-to-Text')
        config_window.geometry('640x480')

        # Title for the configuration window
        title_label = ctk.CTkLabel(config_window, text='Configure Speech-to-Text', font=('Arial', 24, 'bold'))
        title_label.pack(pady=20)

        devList = list(sounddevice.query_devices())
        devNames = []
        devDict = {}
        for dev in devList:
            devName = f"[{dev['index']}] {dev['name']}"
            devNames.append(devName)
            devDict[devName] = dev['index']

        default_sound_settings = {
            "device": ''
        }

        # Load existing configuration if available
        config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stt_config.json')
        if os.path.exists(config_file) and os.path.getsize(config_file) > 0:
            with open(config_file, 'r') as f:
                try:
                    stt_data = json.load(f)
                except json.JSONDecodeError:
                    print("Invalid JSON in stt_config.json. Loading default configuration.")
                    Thread(target=lambda: toast("Invalid Configuration", "The gesture configuration file is invalid. Loading default configuration.")).start()
                    stt_data = default_sound_settings.copy()
                    # Save default configuration back to the file
                    with open(config_file, 'w') as fw:
                        json.dump(stt_data, fw)
        else:
            # File doesn't exist or is empty, use default configuration
            stt_data = default_sound_settings.copy()
            # Save default configuration to the file
            with open(config_file, 'w') as f:
                json.dump(stt_data, f)

        def save_configuration():
            selected_device = devDict[dropdown.get()]  # devName
            default_sound_settings['device'] = selected_device
            try:
                with open(config_file, 'w') as f:
                    json.dump(default_sound_settings, f)
                Thread(target=lambda: toast("Configuration Saved", "Speech-to-Text configuration saved successfully!")).start()
            except Exception as e:
                Thread(target=lambda: toast("Configuration Error", f"Failed to save settings: {e}")).start()
                print(f"Failed to save settings: {e}")
            config_window.destroy()

        label = ctk.CTkLabel(config_window, text="Choose Input Device:", font=('Arial', 12))
        label.pack(pady=5)

        default_value = stt_data.get("device", devNames[0])
        dropdown = ctk.CTkOptionMenu(config_window, values=devNames)
        dropdown.set(default_value)
        dropdown.pack(pady=5)

        save_button = ctk.CTkButton(config_window, text='Save', command=save_configuration)
        save_button.pack(pady=20)

        config_window.mainloop()

    def config_gestures(self):
        """Opens a new window to configure gesture actions."""
        config_window = ctk.CTkToplevel()  # Use CTkToplevel instead of Toplevel
        config_window.title('Configure Gestures')
        config_window.geometry('640x480')

        # Title for the configuration window
        title_label = ctk.CTkLabel(config_window, text='Configure Gestures', font=('Arial', 24, 'bold'))
        title_label.pack(pady=20)

        gestures = [2, 3, 4]  # Number of fingers up to configure
        actions = [
            "Open Snipping Tool",
            "Open Calculator",
            "Open Notepad",
            "Play/Pause Media",
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
                    Thread(target=lambda: toast("Invalid Configuration", "The gesture configuration file is invalid. Loading default configuration.")).start()
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

        for gesture in gestures:
            frame = ctk.CTkFrame(config_window)
            frame.pack(pady=10)

            label = ctk.CTkLabel(frame, text=f"{gesture} Fingers Up", font=('Arial', 16))
            label.pack(side="left", padx=10)

            var = ctk.StringVar(value=gesture_actions.get(str(gesture), ""))
            dropdown = ctk.CTkOptionMenu(frame, variable=var, values=actions)
            dropdown.pack(side="left", padx=10)

            dropdowns[gesture] = var

        def save_configuration():
            # Save the selected actions
            for g in gestures:
                selected_action = dropdowns[g].get()
                if selected_action == "Other...":
                    # Prompt user for custom input
                    custom_action = ctk.CTkInputDialog(text="Enter action", title="Custom Action")
                    if custom_action:
                        gesture_actions[str(g)] = custom_action.get_input()
                    else:
                        # If no input, default to the previous action
                        gesture_actions[str(g)] = gesture_actions.get(str(g), actions[0])
                        dropdowns[g].set(gesture_actions[str(g)])
                else:
                    gesture_actions[str(g)] = selected_action
            try:
                with open(config_file, 'w') as f:
                    json.dump(gesture_actions, f)
                Thread(target=lambda: toast("Configuration Saved", "Gesture configuration saved successfully!")).start()
            except Exception as e:
                Thread(target=lambda: toast("Configuration Error", f"Failed to save configuration with error: {e}")).start()
                print(f"Failed to save configurations: {e}")
            config_window.destroy()

        save_button = ctk.CTkButton(config_window, text="Save Configuration", command=save_configuration)
        save_button.pack(pady=20)

        config_window.mainloop()

    def gesture_info(self):
        """Opens a new window to display information about the gestures."""
        # Create a gestures guide window
        gestures_window = ctk.CTkToplevel()
        gestures_window.title('Hand Gesture Guide')
        gestures_window.geometry('640x480')

        label = ctk.CTkLabel(gestures_window, text='Hand Gesture Guide', font=('Arial', 24, 'bold'))
        label.pack(pady=20)

        gesture_frame = ctk.CTkFrame(gestures_window)
        gesture_frame.pack(pady=20)

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
            label = ctk.CTkLabel(gesture_frame, text=f'{gesture}: {description}', font=('Arial', 12))
            label.pack(pady=5, padx=10)

        gestures_window.mainloop()

    def open_main_window(self):
        """Opens the main window of the project."""
        main_window = ctk.CTk()
        main_window.title("Hand Gesture Control")
        main_window.geometry("800x650")

        # Check if the user is new
        new_user = self.check_new_user()

        # Create a frame for the main window
        main_frame = ctk.CTkFrame(main_window)
        main_frame.pack(pady=15)

        # Create a title label
        title_label = ctk.CTkLabel(main_frame, text="Main Menu", font=("Arial", 28, "bold"))
        title_label.pack(pady=20, padx=50)

        # Create a frame for gesture controls
        gesture_frame = ctk.CTkFrame(main_frame)
        gesture_frame.pack(pady=20, padx=20, fill="both", expand=True)

        gesture_label = ctk.CTkLabel(gesture_frame, text="Gesture Controls", font=("Arial", 20, "bold"))
        gesture_label.pack(pady=10)

        # Create a button to start the Hand Gesture Control
        def start_hand_gesture_control():
            # Create a loading window
            loading_window = ctk.CTkToplevel(self)
            loading_window.title('Loading')
            loading_window.geometry('400x125')

            label = ctk.CTkLabel(loading_window, text='Starting camera...', font=('Arial', 18, 'bold'))
            label.pack(pady=20)

            progress_bar = ctk.CTkProgressBar(loading_window, mode='indeterminate')
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
                controller.run()

            # Start the controller in a separate thread to avoid blocking the UI
            controller_thread = Thread(target=start_controller)
            controller_thread.daemon = True
            controller_thread.start()

            loading_window.after(15000, lambda: loading_window.destroy())

        start_button = ctk.CTkButton(gesture_frame, text="Start Hand Gesture Control", font=("Arial", 16), command=start_hand_gesture_control)
        start_button.pack(pady=10)

        # Create a button to open the configuration window for Gesture Actions
        config_gestures_button = ctk.CTkButton(gesture_frame, text="Configure Gesture Actions", font=("Arial", 16), command=self.config_gestures)
        config_gestures_button.pack(pady=10)

        # Create a button to open the gesture information window
        gesture_info_button = ctk.CTkButton(gesture_frame, text="Gesture Information", font=("Arial", 16), command=self.gesture_info)
        gesture_info_button.pack(pady=10)

        # Create a frame for speech-to-text controls
        stt_frame = ctk.CTkFrame(main_frame)
        stt_frame.pack(pady=20, padx=20, fill="both", expand=True)

        stt_label = ctk.CTkLabel(stt_frame, text="Speech-To-Text Controls", font=("Arial", 20, "bold"))
        stt_label.pack(pady=10)

        # Create a button to start the Speech-To-Text
        def start_speech_to_text():
            # Start the Speech-To-Text in a new thread
            activate_stt()

        start_stt_button = ctk.CTkButton(stt_frame, text="Start Speech-To-Text", font=("Arial", 16), command=start_speech_to_text)
        start_stt_button.pack(pady=10)

        # Create a button to stop the Speech-To-Text
        def stop_speech_to_text():
            # Stop the Speech-To-Text
            deactivate_stt()

        stop_stt_button = ctk.CTkButton(stt_frame, text="Stop Speech-To-Text", font=("Arial", 16), command=stop_speech_to_text)
        stop_stt_button.pack(pady=10)

        # Create a button to open the configuration window for Speech-To-Text
        config_stt_button = ctk.CTkButton(stt_frame, text="Configure Speech-To-Text", font=("Arial", 16), command=self.config_stt)
        config_stt_button.pack(pady=10)

        # Create a frame for appearance mode controls
        mode_frame = ctk.CTkFrame(main_frame)
        mode_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Create a label and switch for appearance mode
        mode_label = ctk.CTkLabel(mode_frame, text="Appearance Mode", font=("Arial", 16))
        mode_label.pack(side="left", padx=10)

        def change_mode():
            value = switch_var.get()
            if value:
                ctk.set_appearance_mode("dark")
                # Write the theme configuration to a file
                with open("theme_config.txt", "w") as f:
                    f.write("dark")
            else:
                ctk.set_appearance_mode("light")
                # Write the theme configuration to a file
                with open("theme_config.txt", "w") as f:
                    f.write("light")

        switch_var = ctk.BooleanVar()
        mode_switch = ctk.CTkSwitch(mode_frame, text="Dark Mode", command=change_mode, onvalue=True, offvalue=False, variable=switch_var)
        mode_switch.pack(side="left", padx=10)
        
        # If the theme is dark, set the switch to True
        if ctk.get_appearance_mode() == "dark":
            switch_var.set(True)
            mode_switch.select()

        main_window.mainloop()

    def first_time_user(self):
        self.movement_window()

    def movement_window(self):
        """Create a window showing the user how to move the cursor."""
        movement_window = ctk.CTkToplevel()
        movement_window.title('Movement Guide')
        movement_window.geometry('640x480')

        label = ctk.CTkLabel(movement_window, text='Movement Guide', font=('Arial', 24, 'bold'))
        label.pack(pady=20)

        label = ctk.CTkLabel(movement_window, text='1. Use the pointer finger to move the cursor', font=('Arial', 14))
        label.pack(pady=5)

        img = Image.open('images/tap.png')

        click_gif = ctk.CTkImage(light_image=img, dark_image=img, size=(150, 150))
        click_label = ctk.CTkLabel(movement_window, image=click_gif, text='')
        click_label.pack(pady=10)

        next_button = ctk.CTkButton(movement_window, text='Next', command=lambda: self.click_window(movement_window))
        next_button.pack(pady=10)

        def on_app_window_close():
            # Make sure the main window is destroyed when the app window is closed
            movement_window.destroy()
            self.destroy()

        # Bind the close event to ensure the main window is properly destroyed
        movement_window.protocol("WM_DELETE_WINDOW", on_app_window_close)

        movement_window.mainloop()

    def click_window(self, movement_window):
        """Create a window showing the user how to click."""
        click_window = ctk.CTkToplevel()
        click_window.title('Click Guide')
        click_window.geometry('640x480')

        label = ctk.CTkLabel(click_window, text='Click Guide', font=('Arial', 24, 'bold'))
        label.pack(pady=20)

        label = ctk.CTkLabel(click_window, text='2. Touch the thumb and index finger to click or drag', font=('Arial', 14))
        label.pack(pady=5)

        img = Image.open('images/measure.png')

        click_gif = ctk.CTkImage(light_image=img, dark_image=img, size=(150, 150))
        click_label = ctk.CTkLabel(click_window, image=click_gif, text='')
        click_label.pack(pady=10)

        movement_window.destroy()

        next_button = ctk.CTkButton(click_window, text='Next', command=lambda: self.screenshot_window(click_window))
        next_button.pack(pady=10)

        click_window.mainloop()

    def screenshot_window(self, click_window):
        """Create a window showing the user how to take a screenshot."""
        screenshot_window = ctk.CTkToplevel()
        screenshot_window.title('Screenshot Guide')
        screenshot_window.geometry('640x480')

        label = ctk.CTkLabel(screenshot_window, text='Screenshot Guide', font=('Arial', 24, 'bold'))
        label.pack(pady=20)

        label = ctk.CTkLabel(screenshot_window, text='3. Make the shaka sign to take a screenshot', font=('Arial', 14))
        label.pack(pady=5)

        img = Image.open('images/shaka.png')

        click_gif = ctk.CTkImage(light_image=img, dark_image=img, size=(150, 150))
        click_label = ctk.CTkLabel(screenshot_window, image=click_gif, text='')
        click_label.pack(pady=10)

        click_window.destroy()

        next_button = ctk.CTkButton(screenshot_window, text='Next', command=lambda: self.open_login_window(screenshot_window))
        next_button.pack(pady=10)

        screenshot_window.mainloop()

if __name__ == "__main__":
    # Load gesture actions from configuration file
    config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gesture_config.json')
    with open(config_file, 'r') as f:
        gesture_actions = json.load(f)

    # Check for the existence of the theme_config.txt file
    if os.path.exists("theme_config.txt"):
        # If the file exists, load the theme configuration
        with open("theme_config.txt", "r") as f:
            theme = f.read().strip()
        ctk.set_appearance_mode(theme)
    else:
        # If the file does not exist, use the default theme
        ctk.set_appearance_mode("dark")

    # Start the app
    app = App()
    app.mainloop()