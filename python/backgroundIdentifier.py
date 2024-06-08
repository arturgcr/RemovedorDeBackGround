import numpy as np
import cv2
import os

class BackGroundIdentifier:
    @staticmethod
    def calculate_background(video_path="videos/video.mp4", background_img_path="images/background.png", batch_size=1, gaussian_kernel_size=(1, 1), gaussian_sigma=1, bilateral_diameter=1, bilateral_sigma_color=0, bilateral_sigma_space=0):
        # Get the output directory
        output_dir = os.path.dirname(background_img_path)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check if the video exists
        video = cv2.VideoCapture(video_path)
        # Check if the video was opened
        if not video.isOpened():
            # Raise an error if the video couldn't be opened
            raise FileNotFoundError(f"Couldn't open the video '{video_path}'. Please check the file path.")

        # Get the video's height and width
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Create a numpy array to store the accumulated background
        accumulated_background = np.zeros((height, width, 3), np.float32)
        # Initialize the number of frames
        num_frames = 0

        # Loop through the video
        while True:
            # Create a list to store the frames
            frames = []
            # Loop through the batch size
            for _ in range(batch_size):
                # Read the video frame
                success, frame = video.read()
                # Break the loop if the frame couldn't be read
                if not success:
                    break
                # Append the frame to the list
                frames.append(frame)
            # Break the loop if there are no frames
            if not frames:
                break
            # Convert the frames to a numpy array
            frames = np.array(frames, dtype=np.float32)
            # Accumulate the frames
            accumulated_background += np.sum(frames, axis=0)
            # Increment the number of frames
            num_frames += len(frames)
        #Verify if the number of frames is greater than 0 to avoid division by zero
        if num_frames > 0:
            # Calculate the average background
            accumulated_background /= num_frames
        # Convert the accumulated background to an unsigned 8-bit integer
        background = accumulated_background.astype(np.uint8)
        # Apply bilateral filtering to the background
        background = cv2.bilateralFilter(background, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
        # Apply additional Gaussian smoothing
        background = cv2.GaussianBlur(background, gaussian_kernel_size, gaussian_sigma)
        # Save the background image
        cv2.imwrite(background_img_path, background)
        print("Finished!")

BackGroundIdentifier.calculate_background()
