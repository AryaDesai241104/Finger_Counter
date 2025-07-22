````markdown
# Real-time Finger Counter with MediaPipe

This project implements a real-time finger counter using OpenCV and Google's MediaPipe library. It detects hands from a live webcam feed, identifies finger landmarks, counts the number of fingers extended, and overlays a corresponding image (e.g., an icon representing the count) onto the video stream. It supports counting fingers from **up to two hands**, allowing for counts up to 10.

## Features

- **Real-time Hand Detection:** Utilizes MediaPipe's robust hand tracking solution.
- **Multi-Hand Support:** Detects and counts fingers from up to two hands simultaneously.
- **Accurate Finger Counting:** Logic to determine extended fingers, including a refined thumb detection based on hand orientation.
- **Customizable Overlays:** Displays custom images corresponding to the total finger count (0-10).
- **FPS Display:** Shows the frames per second of the live feed.

## Requirements

- Python 3.9 - 3.12 (MediaPipe often has compatibility issues with the very latest Python versions like 3.13)
- `pip` (Python package installer)

## Installation

1.  **Install Dependencies:**
    Navigate to your project directory (e.g., where `FingerCountingProject.py` is located) in your terminal or command prompt. Then, install the required libraries using `pip`:

    ```bash
    # Navigate to your project directory (e.g., where FingerCountingProject.py is)
    cd path/to/your/Finger_Counter_Project

    # Install dependencies
    pip install -r requirements.txt
    ```

    **Troubleshooting Installation Errors (especially for MediaPipe/NumPy):**
    If you encounter `OSError: [WinError 2] The system cannot find the file specified` or similar errors during installation, try running your terminal/command prompt **as an Administrator** and then repeat the `pip install` command. Temporarily disabling of antivirus might also be necessary.

## Project Structure
````

Finger_Counter/
├── FingerCountingProject.py \# Main script to run the application
├── HandTrackingModule.py \# Contains the handDetector class
├── finger_images/ \# Folder to store overlay images
│ ├── 0.png \# Image for 0 fingers
│ ├── 1.png \# Image for 1 finger
│ ├── 2.png \# ...
│ ├── ... \# ... up to 10.png (or your max count)
│ └── 10.png
└── requirements.txt \# List of Python dependencies

````

## Usage

1.  **Prepare Overlay Images:**
    Create a folder named `finger_images` in the same directory as `FingerCountingProject.py`. Place your overlay images (e.g., icons, numbers) inside this folder.
    * Name them numerically: `0.png`, `1.png`, `2.png`, ..., `10.png` (or up to your maximum desired count). The script sorts them numerically.
    * Ensure these images are relatively small (e.g., 200x200 pixels) or the script will resize them, but large source images can impact performance.

2.  **Run the Application:**
    Open your terminal or command prompt, navigate to your project directory, and run the main script:

    ```bash
    python FingerCountingProject.py
    ```

3.  **Control:**
    * A window showing your webcam feed with hand tracking and the finger count overlay will appear.
    * **Press the 'q' key** on your keyboard to quit the application.

## Customization

* **Camera Index:** In `FingerCountingProject.py`, change `cap = cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or another index if your default webcam is not `0`.
* **Overlay Image Size:** In `FingerCountingProject.py`, adjust `target_overlay_size = (200, 200)` to change the size of the overlaid images.
* **Detection/Tracking Confidence:** In `HandTrackingModule.py`, you can adjust `detectionCon` and `trackCon` in the `handDetector` constructor (e.g., `handDetector(detectionCon=0.8, trackCon=0.7)`). Higher values mean stricter detection/tracking.
* **Suppress Console Warnings:** To hide `INFO` and `WARNING` messages from MediaPipe/TensorFlow Lite in your console, set the following environment variables in your terminal *before* running the script:
    * **For Command Prompt (cmd.exe):**
        ```cmd
        set GLOG_minloglevel=2
        set MEDIAPIPE_DISABLE_GPU=1
        set TF_CPP_MIN_LOG_LEVEL=2
        ```
    * **For PowerShell:**
        ```powershell
        $env:GLOG_minloglevel = "2"
        $env:MEDIAPIPE_DISABLE_GPU = "1"
        $env:TF_CPP_MIN_LOG_LEVEL = "2"
        ```

## Troubleshooting

* **`ModuleNotFoundError: No module named 'HandTrackingModule'`**: Ensure `HandTrackingModule.py` is saved in the same directory as `FingerCountingProject.py`.
* **`AttributeError: 'NoneType' object has no attribute 'copy'` or `cv2.error: (-215:Assertion failed) !_src.empty()`**:
    * The webcam failed to open or read a frame.
    * Check if `cv2.VideoCapture(0)` should be `(1)` or another index.
    * Ensure no other application is using your webcam.
    * Verify webcam drivers are installed and up-to-date.
* **Incorrect Finger Counting (especially thumb or off-by-one):**
    * **Hand Orientation:** Try different hand orientations (e.g., flat, slightly angled) and observe the count. The thumb logic is sensitive to the hand's pose relative to the camera.
    * **Lighting:** Ensure good, even lighting on your hand.
    * **Occlusion:** Make sure your fingers are clearly visible and not overlapping excessively.
    * **`tipIds` logic:** The `tipIds` and comparison logic (`<` or `>`) for finger extension are standard but might need minor tweaks if your hand poses are unusual.
* **Overlay Image Not Appearing or Appearing Incorrectly:**
    * Verify `finger_images` folder exists.
    * Confirm images are named `0.png`, `1.png`, etc., correctly.
    * Check if `totalFingersUp` is going beyond the number of images you have (e.g., if you only have `0.png` to `5.png` but `totalFingersUp` becomes 6). The code handles this by printing a warning.

````
