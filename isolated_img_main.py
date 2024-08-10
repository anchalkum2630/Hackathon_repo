import numpy as np
import cv2
import matplotlib.pyplot as plt


# this code is to run only isolated image in the problem


def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Example color list
    for i, XYs in enumerate(paths_XYs):
      c = colours[i % len(colours)]
      for XY in XYs:
          ax.plot(XY[:, 0], XY[:, 1], color=c, linewidth=1.5)
    ax.set_aspect('equal')
    # Hide axes
    ax.axis('off')

    # Save the figure
    plt.savefig('isolated_image.png', bbox_inches='tight', pad_inches=0)

    # Display the image
    plt.show()

    # Close the figure
    plt.close()

# read csv file and plot the picture of problem
path1=read_csv('problem/isolated.csv')
plot(path1)

# image processing for correct shapes
image = cv2.imread('isolated_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to get a binary image
_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas with the same dimensions as the original image
canvas = np.ones_like(image) * 255  # White canvas

i = 0

# Function to correct triangle shape
def correct_triangle(contour, image):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    
    if len(approx) == 3:
        # Get the points of the triangle
        pts = approx.reshape(3, 2)
        
        # Calculate the centroid of the triangle
        centroid = np.mean(pts, axis=0)
        
        # Sort the points by angle from the centroid
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        pts = pts[np.argsort(angles)]
        
        # Calculate the desired regular triangle's points based on the centroid
        side_length = np.linalg.norm(pts[0] - pts[1])
        angle_offset = np.pi / 3  # 60 degrees
        
        corrected_pts = []
        for i in range(3):
            angle = angles[i] + angle_offset * i
            corrected_pt = [
                centroid[0] + side_length * np.cos(angle),
                centroid[1] + side_length * np.sin(angle)
            ]
            corrected_pts.append(corrected_pt)
        
        corrected_pts = np.array(corrected_pts, dtype=np.float32)
        
        # Draw the corrected triangle on the canvas
        corrected_pts = corrected_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [corrected_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        return "Triangle"

# Function to correct circle shape
def correct_circle(contour, image):
    # Fit a minimum enclosing circle around the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Draw the corrected circle on the canvas
    cv2.circle(canvas, center, radius, (0, 255, 0), 2)
    
    return "Circle"

# Function to identify and correct each shape
def correct_shape(contour, image):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 3:
        return correct_triangle(contour, image)
    elif len(approx) == 4:
        # Check if the shape is square or rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)
        shape = "Square" if 0.95 <= aspectRatio <= 1.05 else "Rectangle"
        
        # Correct the shape by applying a perspective transformation
        pts = np.float32(approx.reshape(4, 2))
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)  # Convert to float32
        
        # Ensure pts is also in float32 format
        if pts.shape == (4, 2):
            M = cv2.getPerspectiveTransform(pts, box)
            corrected_shape = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            cv2.drawContours(canvas, [np.int32(box)], -1, (0, 255, 0), 2)
        
        return shape
    elif len(approx) == 10:
        cv2.drawContours(canvas, [approx], -1, (0, 255, 0), 2)
        return "Star"
    elif len(approx) > 5:
        return correct_circle(contour, image) if cv2.isContourConvex(approx) else "Ellipse"
    else:
        return "Unknown"

# Loop through each contour and correct the shape
for contour in contours:
    if i == 0:
        i += 1
        continue
    shape = correct_shape(contour, image)
    print(f"Detected and corrected shape: {shape}")

# Show the corrected image on the original and canvas
# cv2_imshow(image)
cv2.imshow('detectedcanvas',canvas)
cv2.imwrite('isolated_corrected_shapes.png',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
