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
from threading import Thread
import json  # To store configurations
import subprocess  # To run applications

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
        self.click_threshold = 0.3  # seconds
        self.drag_threshold = 0.3  # seconds

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

        # Thumb detection
        thumb_tip_x = self.lm_list[4][0]
        thumb_ip_x = self.lm_list[2][0]  # Compare with the IP joint for better accuracy
        if thumb_tip_x > thumb_ip_x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down

        # Fingers (Index to Pinky)
        tips_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]  # PIP joints

        for tip_id, pip_id in zip(tips_ids, pip_ids):
            tip_y = self.lm_list[tip_id][1]
            pip_y = self.lm_list[pip_id][1]
            if tip_y < pip_y - 5:  # Adjust threshold as needed
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down

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
                    thumb_tip, index_tip = self.get_landmark_positions(hand_landmarks)

                    # Move cursor
                    self.move_cursor(index_tip)

                    # Detect gestures
                    if self.detect_middle_finger_gesture():
                        print("Middle finger gesture detected - Exiting")
                        break
                    elif self.detect_shaka_sign():
                        self.take_screenshot()
                    elif self.detect_curled_fist():
                        self.fist_detected_seconds += 0.1  # Increment counter if fist is detected
                        if self.fist_detected_seconds >= 43:  # Check if the fist has been held for 60 seconds
                            webbrowser.open("https://blacklivesmatter.com/") #Easter Egg
                            self.fist_detected_seconds = 0  # Reset the counter after opening the website
                    else:
                        self.detect_custom_gestures()
                        # Handle click and drag
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

# GUI functions for configuration
def configure_gestures():
    """Opens a GUI for users to assign actions to gestures."""
    config_window = tk.Tk()
    config_window.title("Configure Gestures")
    config_window.geometry("400x400")

    gestures = [2, 3, 4]
    actions = [  # Updated actions list with "Other..."
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

    # Load existing configuration if available
    config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gesture_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            gesture_actions = json.load(f)
    else:
        gesture_actions = {str(g): actions[0] for g in gestures}

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
                    # If no input, default to first action
                    gesture_actions[str(g)] = actions[0]
                    dropdowns[g].set(actions[0])
            else:
                gesture_actions[str(g)] = selected_action
        try:
            with open(config_file, 'w') as f:
                json.dump(gesture_actions, f)
            messagebox.showinfo("Configuration Saved", "Your gesture configurations have been saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations: {e}")
        config_window.destroy()

    for g in gestures:
        ttk.Label(config_window, text=f"Gesture: {g} Fingers Up").pack(pady=5)
        default_value = gesture_actions.get(str(g), actions[0])
        combobox = ttk.Combobox(config_window, values=actions, state='readonly')
        # Include custom action if not in actions
        if default_value not in actions:
            combobox['values'] = actions + [default_value]
        combobox.set(default_value)
        combobox.pack(pady=5)
        dropdowns[g] = combobox

    ttk.Button(config_window, text="Save", command=save_configuration).pack(pady=20)
    config_window.mainloop()



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

    # Create a button to configure gestures
    config_button = ttk.Button(app_window, text="Configure Gestures", command=configure_gestures)
    config_button.pack(pady=10)

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
        # Load gesture configurations
        config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gesture_config.json')
        print(f"Configuration file path (loading): {config_file}")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                gesture_actions = json.load(f)
            # Do not convert keys to integers
            # gesture_actions = {int(k): v for k, v in gesture_actions.items()}
        else:
            gesture_actions = {}

        controller = HandMouseController(gesture_actions)
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
    # Ensure you have an image file named 'gestures.png' in the same directory
    gestures_image = tk.PhotoImage(file="gestures.png")
    gestures_label = ttk.Label(gestures_window, image=gestures_image)
    gestures_label.image = gestures_image  # Keep a reference
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
