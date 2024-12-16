import cv2
import numpy as np

def extract_keyframes(video_path, threshold=30):
    """
    Extract keyframes from a video based on frame differences.

    :param video_path: Path to the video file.
    :param threshold: The threshold for detecting significant differences between frames.
    :return: List of extracted keyframes (images).
    """
    try:
        # Open the video
        video = cv2.VideoCapture(video_path)
        ret, prev_frame = video.read()
        keyframes = []

        if not ret:
            return keyframes  # Return empty list if video can't be read

        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

        while ret:
            ret, frame = video.read()
            if not ret:
                break

            # Convert the frame to HSV (Hue, Saturation, Value) color space for better comparison
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate frame difference
            diff = np.sum(np.abs(hsv - prev_hsv))  # Sum the differences between the current and previous frames

            # If the difference is above the threshold, we consider this a keyframe
            if diff > threshold:
                keyframes.append(frame)  # Append the frame as a keyframe

            prev_hsv = hsv  # Update previous frame for next comparison

        video.release()  # Release the video after processing
        return keyframes
    except Exception as e:
        raise Exception(f"Error during keyframe extraction: {str(e)}")
