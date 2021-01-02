# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:37:26 2020

@author: DHRUV
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# make the coordinates of images
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# canny effects on video
def canny_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# region_of_interest function this function is use for masking
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


# this function for displaying lines
def display_line(image, lines):
    line_img = np.zeros_like(image)
    if (lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img

# for finding a slops
def average_slope_intercept(image, lines):
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = make_coordinates(image, left_avg)
    right_line = make_coordinates(image, right_avg)
    return np.array([left_line, right_line])

img = cv2.imread("testing_image.jpg")
lane_img = np.copy(img)
#canny_img = canny_effect(lane_img)
#cropped_img = region_of_interest(canny_img)
#lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
#avg_lines = average_slope_intercept(lane_img, lines)
#line_img = display_line(lane_img, avg_lines)
#combo = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
#cv2.imshow("result", combo)
#cv2.imshow("result", gray)
#cv2.imshow("result", img)
#cv2.waitKey(0)

cap = cv2.VideoCapture("testing_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny_effect(lane_img)
    cropped_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    avg_lines = average_slope_intercept(frame, lines)
    line_img = display_line(frame, avg_lines)
    combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow("result", combo)
    #cv2.imshow("result", gray)
    #cv2.imshow("result", img)
    if (cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
