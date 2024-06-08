#!/usr/bin/env python
import cv2 as cv
import numpy as np
import random as rng
from scipy.stats import entropy

# Initialize random seed
rng.seed(12345)

def preprocess_image(src):
    """ Convert image to grayscale and apply binary inversion thresholding. """
    # Convert the image to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv.GaussianBlur(gray, (5, 5), 0)  # Using a 5x5 kernel with a standard deviation of 0
    
    # Apply binary inversion thresholding
    # Threshold set closer to 255 to capture nearly perfect white
    _, src_gray = cv.threshold(blurred, 240, 255, cv.THRESH_BINARY_INV)
    
    # Save the grayscale image (optional)
    return src_gray


def find_and_draw_contours(src_gray):
    """ Find contours, calculate and display contours and hulls with their area difference ratio. """
    canny_output = cv.Canny(src_gray, 100, 200)  # Fixed threshold values
    kernel = np.ones((3,3), np.uint8)
    canny_output = cv.morphologyEx(canny_output, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
    countour_object = None

    # Find the index of the outermost contour (parent contour with no parent in hierarchy)
    if hierarchy is not None and len(hierarchy) > 0:
        hierarchy = hierarchy[0]  # Get the first hierarchy level which contains info for each contour
        for i, h in enumerate(hierarchy):
            # h[3] == -1 indicates that the contour has no parent
            if h[3] == -1:
                # This contour has no parent, so it's an outer contour
                countour_object = contours[i]

    # close gaps in the contour
    epsilon = 0.001 * cv.arcLength(countour_object, True)
    countour_object = cv.approxPolyDP(countour_object, epsilon, True)
    ratio, hull = draw_and_print_contour_properties(drawing, countour_object)
    cv.drawContours(drawing, [countour_object], -1, (0, 255, 0), 1)  # Contour in green
    cv.drawContours(drawing, [hull], -1, (255,255,255), 1)           # Hull in red color
    return drawing, ratio

def draw_and_print_contour_properties(drawing, contour):
    """ Draw contour and its convex hull, and compute area difference ratio. """
    # print(f'Contour: {contour.shape}')
    contour_area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    area_difference_ratio = (hull_area - contour_area) / hull_area if hull_area != 0 else 0
    print(hull_area, contour_area, area_difference_ratio)

    # Display area difference ratio
    # print(f'Area Difference Ratio: {area_difference_ratio:.2f}')

    # Draw contour and hull points for visualization

    return area_difference_ratio, hull

def calculate_entropy(image):
    """ Calculate the entropy of an image. """
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    prob = hist / hist.sum()
    e = entropy(prob, base=2)
    return e.sum()

def count_edges(src_gray):
    """ Count the number of edges in the grayscale image using Canny edge detection. """
    edges = cv.Canny(src_gray, 100, 200)  # Apply Canny edge detection
    edge_count = cv.countNonZero(edges)  # Count non-zero pixels which represent edges
    return edge_count

def count_corners(src_gray):
    """ Count the number of corners in the grayscale image using Harris corner detection. """
    # Parameters for Harris corner detection
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detect corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Result is dilated for marking the corners
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image
    corners = dst > 0.01 * dst.max()
    corner_count = np.sum(corners)  # Count true values which represent corners
    return corner_count