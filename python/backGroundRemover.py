import cv2
import numpy as np
import os

class BackgroundRemover:
    @staticmethod
    def remove_background(background_img_path, video_path, result_path, threshold_value=40, morph_kernel_size=(5, 5), min_size=15, alpha=0.5):
        """
        Remove the background from a video based on a provided background image.

        Parameters:
            background_img_path (str): The path to the background image.
            video_path (str): The path to the video to be processed.
            result_path (str): The path where the processed video will be saved.
            threshold_value (int): Threshold value for color segmentation.
            morph_kernel_size (tuple): Kernel size for morphological operations.
            min_size (int): Minimum size of regions of interest.
            alpha (float): Interpolation factor for frame smoothing.

        Returns:
            None
        """
        # Verify file paths
        if not os.path.exists(background_img_path):
            raise FileNotFoundError(f"Background image '{background_img_path}' not found.")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")

        # Load resources
        video = cv2.VideoCapture(video_path)
        background = cv2.imread(background_img_path)
        if background is None:
            raise ValueError(f"Cannot read background image '{background_img_path}'.")

        # Get video properties
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        background_resized = cv2.resize(background, (video_width, video_height))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(result_path, fourcc, video.get(cv2.CAP_PROP_FPS), (video_width, video_height))

        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)

        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        prev_frame = None

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Frame interpolation
            interpolated_frame = cv2.addWeighted(prev_frame, alpha, frame, 1 - alpha, 0) if prev_frame is not None else frame

            # Apply Kalman filter to smooth the frame
            gray_frame = cv2.cvtColor(interpolated_frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_frame)

            # Create a measurement
            measurement = np.array([[np.float32(mean_intensity)], [0]], np.float32)

            # Correct the Kalman filter
            kalman.correct(measurement)
            prediction = kalman.predict()

            smoothed_intensity = prediction[0]

            # Adjust frame intensity based on the smoothed intensity
            adjustment_factor = smoothed_intensity / mean_intensity
            smoothed_frame = (interpolated_frame * adjustment_factor).astype(np.uint8)

            # Background subtraction and processing
            frame_no_background = cv2.absdiff(smoothed_frame, background_resized)

            # Use color difference instead of just grayscale difference
            diff_channels = cv2.split(frame_no_background)
            thresholded_channels = [cv2.threshold(channel, threshold_value, 255, cv2.THRESH_BINARY)[1] for channel in diff_channels]
            combined_threshold = cv2.bitwise_or(cv2.bitwise_or(thresholded_channels[0], thresholded_channels[1]), thresholded_channels[2])

            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
            morph = cv2.morphologyEx(combined_threshold, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

            # Remove small regions
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
            sizes = stats[1:, -1]
            mask = np.zeros(labels.shape, np.uint8)
            for i in range(num_labels - 1):
                if sizes[i] >= min_size:
                    mask[labels == i + 1] = 255

            # Apply mask to the frame
            frame_final = cv2.bitwise_and(frame, frame, mask=mask)
            output_video.write(frame_final)
            prev_frame = frame

        # Release resources
        video.release()
        output_video.release()
        cv2.destroyAllWindows()
        print("Finished!")

# Execute the background remover
BackgroundRemover.remove_background(
    "images/background.png", 
    "videos/video.mp4", 
    "videos/video_novo.mp4",
    threshold_value=45,
    morph_kernel_size=(15, 15),
    min_size=850,
    alpha=0.5
)
