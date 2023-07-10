import cv2
import numpy as np


def run_pose_estimation_webcam():
    # Load your compiled model and other necessary configurations here.

    # ...


    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the frame")
            break

        # If the frame is larger than full HD, reduce size to improve performance.
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Resize the image and change dimensions to fit neural network input.
        input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

        # Measure processing time.
        start_time = time.time()
        # Get results.
        results = compiled_model([input_img])
        stop_time = time.time()

        pafs = results[pafs_output_key]
        heatmaps = results[heatmaps_output_key]
        # Get poses from network results.
        poses, scores = process_results(frame, pafs, heatmaps)

        # Draw poses on a frame.
        frame = draw_poses(frame, poses, 0.1)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_estimation_webcam()