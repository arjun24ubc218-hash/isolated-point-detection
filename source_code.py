import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the Input Image
# --------------------------------------------------

image = cv2.imread("input_image.jpg")

if image is None:
    print("Error: input_image.jpg not found.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Input Image")
plt.axis("off")
plt.show()


# --------------------------------------------------
# 2. Point-Detection Mask
# --------------------------------------------------

point_mask = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

print("Point Detection Mask:")
print(point_mask)


# --------------------------------------------------
# 3. Apply the Point-Detection Mask
# --------------------------------------------------

filtered = cv2.filter2D(
    gray.astype(np.float32),
    -1,
    point_mask
)

plt.figure(figsize=(7, 7))
plt.imshow(filtered, cmap="gray")
plt.title("Point Detection Response")
plt.colorbar()
plt.axis("off")
plt.show()


# --------------------------------------------------
# 4. Thresholding
# --------------------------------------------------

threshold_value = 100

binary = np.where(
    filtered > threshold_value,
    255,
    0
).astype(np.uint8)

plt.figure(figsize=(7, 7))
plt.imshow(binary, cmap="gray")
plt.title(f"Thresholded Image (Threshold = {threshold_value})")
plt.axis("off")
plt.show()


# --------------------------------------------------
# 5. Find Detected Regions
# --------------------------------------------------

contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Detected regions:", len(contours))


# --------------------------------------------------
# 6. Localize the Detected Points
# --------------------------------------------------

result = image.copy()

detected_centers = []

for contour in contours:

    M = cv2.moments(contour)

    if M["m00"] != 0:

        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        detected_centers.append(
            (center_x, center_y)
        )


# Sort the detected points
detected_centers = sorted(
    detected_centers,
    key=lambda p: (p[1], p[0])
)


# --------------------------------------------------
# 7. Draw Circles Around Detected Points
# --------------------------------------------------

for i, (x, y) in enumerate(
    detected_centers,
    start=1
):

    cv2.circle(
        result,
        (x, y),
        10,
        (0, 0, 255),
        2
    )

    cv2.putText(
        result,
        str(i),
        (x + 12, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )


detected_count = len(detected_centers)

print(
    "Detected isolated points:",
    detected_count
)


# --------------------------------------------------
# 8. Display Final Result
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"Detected Isolated Points: {detected_count}"
)

plt.axis("off")
plt.show()


# --------------------------------------------------
# 9. Save the Result
# --------------------------------------------------

cv2.imwrite(
    "isolated_points_detected.png",
    result
)

print(
    "Result saved as isolated_points_detected.png"
)


# --------------------------------------------------
# 10. Before and After Comparison
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Original Input")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Detected Isolated Points")
plt.axis("off")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Threshold Comparison
# --------------------------------------------------

thresholds = [50, 100, 200, 300]

print("\nThreshold Comparison")
print("--------------------")
print("Threshold | Detected Regions")

for t in thresholds:

    test_binary = np.where(
        filtered > t,
        255,
        0
    ).astype(np.uint8)

    test_contours, _ = cv2.findContours(
        test_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(
        f"{t:9} | {len(test_contours)}")    
