import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.io import imread

# Paths to each subtype folder
folders = {
    "ACA": "E:\\project_folder\\Lung_cancer\\data\\lung_aca",
    "SCC": "E:\\project_folder\\Lung_cancer\\data\\lung_scc",
    "BNT": "E:\\project_folder\\Lung_cancer\\data\\lung_bnt"
}

PATCH_SIZE = (64, 64)

def load_images_from_folder(folder_path):
    """Load all valid images from a folder."""
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and filename.endswith(('.jpg', '.png', '.jpeg')):
            img = imread(img_path)
            images.append(img)
    return images

def extract_patches(image, patch_size=PATCH_SIZE):
    """Divide an image into patches of a given size."""
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.shape == patch_size:
                patches.append(patch)
    return patches

def extract_glcm_features_from_patches(patches):
    """Extract GLCM features from a list of patches."""
    features = []
    for patch in patches:
        glcm = graycomatrix((patch * 255).astype('uint8'), distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        feature_vector = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0]
        ]
        features.append(feature_vector)
    return np.array(features)

def extract_glcm_features(images):
    """Extract GLCM features from patches in each image."""
    all_image_features = []
    for img in images:
        gray_img = rgb2gray(img)
        patches = extract_patches(gray_img)
        patch_features = extract_glcm_features_from_patches(patches)
        all_image_features.extend(patch_features)  # Collect features for all patches in all images
    return np.array(all_image_features)

if __name__ == "__main__":
    print("Running 01_patch_processing.py")
    all_features = {}

    # Process each folder (ACA, SCC, BNT)
    for label, folder_path in folders.items():
        if not os.path.exists(folder_path):
            print(f"Folder path does not exist: {folder_path}")
            continue

        print(f"Loading images from {label} folder...")
        images = load_images_from_folder(folder_path)

        print(f"Extracting GLCM features from patches for {label} images...")
        features = extract_glcm_features(images)
        all_features[label] = features

        # Save features to a .npy file
        np.save(f"glcm_features_{label}.npy", features)
        print(f"Saved GLCM features for {label} to glcm_features_{label}.npy")

    print("GLCM feature extraction from patches completed for all subtypes.")
