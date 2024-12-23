import matplotlib.pyplot as plt
import os
import random
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

path = "Dataset"
print("Path to dataset files:", path)
image_dir_altered = "Dataset/Altered"
image_dir_real = "Dataset/Real1"

# Recursively find all .BMP files in the Altered directory
all_images_altered = []
for root, _, files in os.walk(image_dir_altered):
    for file in files:
        if file.endswith('.BMP'):
            all_images_altered.append(os.path.join(root, file))

# Find all .BMP files in the Real directory
all_images_real = [os.path.join(image_dir_real, f) for f in os.listdir(image_dir_real) if f.endswith('.BMP')]
all_images = all_images_altered + all_images_real

def process_and_plot(image_path, image_name):
    # Load the fingerprint image
    image = cv2.imread(image_path)

    # Resizing images
    resized_image = cv2.resize(image, (0, 0), fx=7, fy=7, interpolation=cv2.INTER_CUBIC)

    # Denoising
    denoised_image = cv2.fastNlMeansDenoising(resized_image, None, 7, 7, 21)

    # Convert to grayscale
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Blurring
    blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    # Adaptive thresholding
    binary_image = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Edge detection
    edges = cv2.Canny(binary_image, threshold1=60, threshold2=120)

    # Compute orientation fields for enhanced and binary images
    gradient_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=5)
    orientation_enhanced = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

    gradient_x = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=5)
    orientation_binary = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

    # Minutiae extraction
    minutiae_visualization = np.dstack((binary_image, binary_image, binary_image))  # Create an RGB visualization
    minutiae_points = []  # List to store minutiae (ridge endings and bifurcations)

    # Scan through the binary image and identify minutiae points
    rows, cols = binary_image.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if binary_image[y, x] == 255:  # If the pixel is part of a ridge
                # Extract the 3x3 neighborhood
                neighborhood = binary_image[y - 1:y + 2, x - 1:x + 2]
                ridge_count = np.sum(neighborhood == 255)

                if ridge_count == 2:  # Ridge ending
                    minutiae_points.append((x, y))
                    cv2.circle(minutiae_visualization, (x, y), 3, (0, 0, 255), -1)  # Red circle for ridge ending
                elif ridge_count == 4:  # Bifurcation
                    minutiae_points.append((x, y))
                    cv2.circle(minutiae_visualization, (x, y), 3, (0, 255, 0), -1)  # Green circle for bifurcation

    # Save minutiae visualization
    output_dir = "Minutiae_processed_images"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_minutiae_visualization.jpg"), minutiae_visualization)

    # Determine the output directory based on the source directory
    if image_dir_altered in image_path:
        output_dir = "Minutiae_processed_images/Altered"
    else:
        output_dir = "Minutiae_processed_images/Real"

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_minutiae_visualization.jpg"), minutiae_visualization)

'''
     # Plot the results
    fig, axes = plt.subplots(1, 9, figsize=(18, 5))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gray_image, cmap="gray")
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Blurred Image")
    axes[2].axis("off")

    axes[3].imshow(edges, cmap="gray")
    axes[3].set_title("Edges")
    axes[3].axis("off")

    axes[4].imshow(enhanced_image, cmap="gray")
    axes[4].set_title("Enhanced Image")
    axes[4].axis("off")

    axes[5].imshow(orientation_enhanced, cmap="gray")
    axes[5].set_title("Orientation Enhanced")
    axes[5].axis("off")

    axes[6].imshow(binary_image, cmap="gray")
    axes[6].set_title("Binary Image")
    axes[6].axis("off")

    axes[7].imshow(orientation_binary, cmap="gray")
    axes[7].set_title("Orientation Binary")
    axes[7].axis("off")

    axes[8].imshow(cv2.cvtColor(minutiae_visualization, cv2.COLOR_BGR2RGB))
    axes[8].set_title("Minutiae Points")
    axes[8].axis("off")

    fig.suptitle(image_name, fontsize=14, y=0.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
'''
for img_path in all_images:
    img_name = os.path.basename(img_path)
    process_and_plot(img_path, img_name)

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(folder.split('/')[-1])  # Use folder name as label
    return images, labels

def extract_features(images):
    features = []
    for img in images:
        # Flatten the image to create a feature vector
        features.append(img.flatten())
    return np.array(features)

# Load training data
train_images, train_labels = load_images_from_folder('Minutiae_processed_images/Real')
train_features = extract_features(train_images)

# Load testing data
test_images, test_labels = load_images_from_folder('Minutiae_processed_images/Altered')
test_features = extract_features(test_images)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(train_features, train_labels)

# Test SVM model
predictions = svm_model.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
