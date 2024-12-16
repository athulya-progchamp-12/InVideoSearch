import cv2
import numpy as np

def extract_keyframes(video_path, threshold=30):
    try:
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
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            diff = np.sum(np.abs(hsv - prev_hsv))  # Calculate frame difference
            if diff > threshold:
                keyframes.append(frame)
            prev_hsv = hsv

        video.release()
        return keyframes
    except Exception as e:
        raise Exception(f"Error during keyframe extraction: {str(e)}")
