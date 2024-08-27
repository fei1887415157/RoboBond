import cv2
import numpy as np
import subprocess
import math
import time





def capture_photo():
    # Capture a photo using libcamera-still
    # "--hdr"
    subprocess.run(["libcamera-still", "-o", "Photos/test.jpg", "-t", "1", "--nopreview", "--hdr"], check=True)

    # Capture a photo using rpicam-jpeg
    #subprocess.run(["rpicam-jpeg", "--output", "Photos/test.jpg", "-t", "1000", "--width", "854", "--height", "480", "--hdr", "auto"], check=True)
    #subprocess.run(["rpicam-jpeg", "--output", "Photos/test.jpg", "-t", "1000", "--hdr", "auto"], check=True)



def calculate_line_intersections(line1, line2, infinite_lines=True):
    """
    Find the intersection of two lines given in endpoints form.
    Returns (x, y) or None if there is no intersection.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel

    # Calculate numerators
    px_num = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
    py_num = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)

    px = px_num / denom
    py = py_num / denom

    if infinite_lines:
        return int(px), int(py)
    else:
        # Check if the intersection point is within both line segments
        if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
            min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
            return int(px), int(py)
        else:
            return None



def merge_intersections(intersections, threshold=100):
    unique_intersections = []

    for i, cur_i in enumerate(intersections):
        keep = True
        for cur_j in unique_intersections:
            x1, y1 = cur_i
            x2, y2 = cur_j
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance < threshold:
                keep = False
                break
        if keep:
            unique_intersections.append(cur_i)

    return unique_intersections



def draw_lines_and_intersections(image, lines):
    intersections = []
    
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            intersection = calculate_line_intersections(line1, line2, False)
            if intersection:
                intersections.append(intersection)
    
    intersections = merge_intersections(intersections)
    
    for x1, y1, x2, y2 in lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 20)

    for intersection in intersections:
        cv2.circle(image, intersection, 20, (0, 0, 255), -1)

    return image
    




# Probablistic Hough Line Transform returns line segments.
def draw_lines(lines, image):
    if lines is None:
        print("no lines detected")
        return
    
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 20)     # overlay lines color and thickness

        slope = calculate_slope(x1, y1, x2, y2)
        print(f'Line: ({x1}, {y1}), ({x2}, {y2}), Slope: {slope}')



def calculate_slope(x1, y1, x2, y2):

    # Handle the case where the line is vertical to avoid division by zero
    if x2 - x1 == 0:
        return float('inf')
    
    return (y2 - y1) / (x2 - x1)



def merge_lines(lines, threshold_angle=30, threshold_distance=1000):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        added = False
        for merged_line in merged_lines:
            mx1, my1, mx2, my2 = merged_line
            slope1 = calculate_slope(x1, y1, x2, y2)
            slope2 = calculate_slope(mx1, my1, mx2, my2)
            angle = math.degrees(math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2))))
            if angle < threshold_angle:
                if abs(x1 - mx1) < threshold_distance and abs(y1 - my1) < threshold_distance:
                    merged_line[0] = min(x1, mx1)
                    merged_line[1] = min(y1, my1)
                    merged_line[2] = max(x2, mx2)
                    merged_line[3] = max(y2, my2)
                    added = True
                    break
        if not added:
            merged_lines.append([x1, y1, x2, y2])
    return merged_lines






def main():

    while True:

        start_time = time.time()

        # Capture a photo
        capture_photo()

        # Read the photo
        image = cv2.imread("Photos/test.jpg")
        
        # Change brightness and contrast
        #image = image.astype(np.float32)
        #image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

        # Reduce image resolution
        scale_percent = 50      # Percentage compared to the original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dimension = (width, height)
        image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

        if image is None:
            print("Error: Could not read image.")
            break



        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur image to reduce noise
        blurred_gray = cv2.GaussianBlur(gray, (51, 51), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred_gray, 5, 15)

        # Use Probabilistic Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=100)
        
        if lines is None:
            print("\n no lines detected \n")
            return



        lines = lines[:, 0, :]       # Reshape for convenience

        


        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The function took {execution_time} seconds to execute.")
        

        draw_lines(lines, image)
        #lines = merge_lines(lines)
        #image = draw_lines_and_intersections(image.copy(), lines)



        #resized_canny = cv2.resize(edges, (665, 500))
        #resized_image = cv2.resize(image, (665, 500))

        resized_canny = cv2.resize(edges, (854, 480))
        resized_image = cv2.resize(image, (854, 480))
        resized_blur = cv2.resize(blurred_gray, (854, 480))


        # Display the resulting image
        cv2.imshow("Canny Edges", resized_canny)
        cv2.imshow("Frame", resized_image)
        cv2.imshow("Blurred Gray", resized_blur)
        

        
        key = cv2.waitKey(0)
        if key == ord('q'):             # Press 'q' to stop
            break
        if key == 32:
            cv2.destroyAllWindows()     # Press 'space' to take another photo



    # When everything is done, close windows
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()
