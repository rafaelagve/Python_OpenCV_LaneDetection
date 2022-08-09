import numpy as np
import cv2


def draw_the_lines(image, lines):
    # create a distinct image for lines [0, 255]
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # There are (x,y) for the starting and end points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # Merge image and lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)

    return image_with_lines


def region_of_interest(image, region_points):
    # Replacing pixels with 0 (black) for regions of non interest
    mask = np.zeros_like(image)

    # Region of interest (lower triangle) - 255 white pixels
    cv2.fillPoly(mask, region_points, 255)

    # Using the mask: maitaining the original image on white pixels
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    # Turn the image into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)

    # Region of interest: lower region

    region_of_interest_vertices = [(0, height), (width / 2, height * 0.65), (width, height)]
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # Line detection algorithm (radians: 1 degree == pi/180), rho: distance of origin (in pi),
    # threshold: accept the line if > 50 curves are intercepted, minLineLength
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=150)

    # draw lines on the image
    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines


video = cv2.VideoCapture('lane_detection_video.mp4')

while video.isOpened():
    is_grabbed, frame = video.read()

    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)

    cv2.imshow('Lane Detection Video', frame)
    cv2.waitKey(70)

video.release()
cv2.destroyAllWindows()
