import cv2
import numpy as np

def main():
    # Open a connection to the video stream using FFmpeg
    cap = cv2.VideoCapture("udp://127.0.0.1:5010", cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        # Draw the lines on the frame
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(0):
            break
    


    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
