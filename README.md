# isolated-point-detection
isolated-point-detection
# Detection and Localization of Isolated Points in Images

## Digital Image Processing Mini Project

### Project Overview

This project demonstrates the detection and localization of isolated points in an image using Digital Image Processing techniques.

An isolated point is a pixel or small region whose intensity is significantly different from the surrounding pixels. The project uses a point-detection mask to identify these points and then uses thresholding and contour detection to locate them.

## Objectives

- Detect isolated points in an image.
- Apply a point-detection mask.
- Use thresholding to separate detected points.
- Locate the detected points using contour detection.
- Mark the detected points on the original image.
- Compare detection results using different threshold values.

## Methodology

The project follows these steps:

1. Load the input image.
2. Convert the image to grayscale.
3. Apply the point-detection mask.
4. Apply thresholding.
5. Detect contours.
6. Locate the detected points.
7. Draw circles around the detected points.
8. Display the final result.

## Point-Detection Mask

The mask used in this project is:

```text
-1  -1  -1
-1   8  -1
-1  -1  -1
