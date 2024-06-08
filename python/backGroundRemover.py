import cv2
import numpy as np
import os

class BackgroundRemover:
    @staticmethod
    def remove_background(background_img_path, video_path, result_path, threshold_value=40, morph_kernel_size=(5, 5), min_size=15):
        # Check if the video exists
        if not os.path.exists(background_img_path):
            # Raise an error if the video couldn't be opened
            raise FileNotFoundError(f"Background image '{background_img_path}' not found. Please verify if the path is correct and if the image was generated.")
        # Get the video
        video = cv2.VideoCapture(video_path)
        # Get the background image
        background = cv2.imread(background_img_path)
        # Check if the image was opened
        if background is None:
            # Raise an error if the image couldn't be opened
            raise ValueError(f"Background image '{background_img_path}' could not be read. Please verify if the file is intact and accessible.")
        # Get the video's height and width
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize background to match video resolution if necessary
        background_resized = cv2.resize(background, (video_width, video_height))
        # Create the output video path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(result_path, fourcc, video.get(cv2.CAP_PROP_FPS), (video_width, video_height))
        # Loop through the video
        while True:
            # Read the video frame
            ret, frame = video.read()
            # Break the loop if the frame couldn't be read
            if not ret:
                break
            # Create a mask to remove the background
            frame_no_background = cv2.absdiff(frame, background_resized)
            # Convert the frame to grayscale
            frame_no_background_gray = cv2.cvtColor(frame_no_background, cv2.COLOR_BGR2GRAY)
            # Apply a threshold to the frame
            _, thresholded = cv2.threshold(frame_no_background_gray, threshold_value, 255, cv2.THRESH_BINARY)
            # Apply morphological operations to the thresholded frame
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            # Find connected components in the thresholded frame
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
            # Create a mask to remove small connected components
            sizes = stats[1:, -1]
            mask = np.zeros(labels.shape, np.uint8)
            # Loop through the connected components
            for j in range(0, num_labels - 1):
                # Check if the connected component is large enough
                if sizes[j] >= min_size:
                    mask[labels == j + 1] = 255
            # Apply the mask to the frame
            frame_final = cv2.bitwise_and(frame, frame, mask=mask)
            # Write the frame to the output video
            output_video.write(frame_final)
        # Release the video and output video
        video.release()
        output_video.release()
        cv2.destroyAllWindows()
        print("Finished!")

BackgroundRemover.remove_background("images/background.png", "videos/video.mp4", "videos/video_novo.mp4",
    threshold_value=45, 
    morph_kernel_size=(15, 15), 
    min_size=550
)
