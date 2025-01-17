# ============================================
# 1. IMPORTS AND GLOBAL VARIABLES
# ============================================

import numpy as np
import cv2
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import GridSearchCV
from PIL import Image, ImageTk
import random
from scipy.interpolate import lagrange

warnings.filterwarnings("ignore")

BLOCK_SIZE = 16
R = 20  # Number of Minutiae points
S = 180  # Number of Chaff points

# ============================================
# 2. PREPROCESSING AND FEATURE EXTRACTION
# ============================================

def do_segmentation(img):
    shape = img.shape
    seg_mask = np.ones(shape)
    threshold = np.var(img) * 0.1
    for i in range(0, img.shape[0], BLOCK_SIZE):
        for j in range(0, img.shape[1], BLOCK_SIZE):
            x, y = min(img.shape[0], i + BLOCK_SIZE), min(img.shape[1], j + BLOCK_SIZE)
            if np.var(img[i:x, j:y]) <= threshold:
                seg_mask[i:x, j:y] = 0
    return np.where(seg_mask == 0, 255, img)

def do_normalization(segmented_image):
    desired_mean, desired_variance = 100.0, 8000.0
    mean, variance = np.mean(segmented_image), np.var(segmented_image)
    normalized_image = (segmented_image - mean) * (desired_variance / variance)**0.5 + desired_mean
    return cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def do_thinning(binarized_image):
    return np.where(skeletonize(binarized_image / 255), 0, 1).astype(np.uint8)

def minutiae_points_computer(img):
    segmented = do_segmentation(img)
    normalized = do_normalization(segmented)
    _, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thinned = do_thinning(binarized)
    minutiae_points = {
        (i, j): (0, 1) for i in range(thinned.shape[0]) for j in range(thinned.shape[1]) if thinned[i, j] == 0
    }
    return minutiae_points

def generate_feature_vector(minutiae_points):
    return np.array([[x, y, orientation] for (x, y), (_, orientation) in minutiae_points.items()]).flatten()

def pad_feature_vectors(feature_vectors, max_length):
    return np.array(
        [np.pad(vector, (0, max_length - len(vector))) if len(vector) < max_length else vector[:max_length] for vector in feature_vectors],
        dtype=np.int32
    )

# ============================================
# 3. FUZZY VAULT FUNCTIONS
# ============================================

def create_fuzzy_vault(minutiae_points, secret_key, degree=3, chaff_points=180):
    """Encodes minutiae points into a Fuzzy Vault"""
    polynomial = np.poly1d(secret_key)
    genuine_points = [(x, polynomial(x)) for x, _ in minutiae_points]
    vault = genuine_points[:]
    while len(vault) < len(minutiae_points) + chaff_points:
        chaff = (random.randint(0, 255), random.randint(0, 255))
        if chaff not in genuine_points:
            vault.append(chaff)
    random.shuffle(vault)
    return vault

def decode_fuzzy_vault(query_minutiae, vault, secret_key):
    """Decodes the Fuzzy Vault using query minutiae and the secret key"""
    matched_points = [point for point in vault if point in query_minutiae]
    if len(matched_points) < len(vault) // 2:
        raise ValueError("Insufficient matching points to reconstruct the polynomial.")
    x_vals, y_vals = zip(*matched_points)
    reconstructed_poly = lagrange(x_vals, y_vals)
    return reconstructed_poly

# ============================================
# 4. DATASET PREPARATION
# ============================================

def load_dataset(dataset_dir, is_fingerprint=True):
    feature_vectors, labels = [], []
    label = 0

    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dataset_dir}. Ensure the directory is structured correctly.")

    for subdir in sorted(subdirs):
        subdir_path = os.path.join(dataset_dir, subdir)
        for img_file in sorted(os.listdir(subdir_path)):
            img_path = os.path.join(subdir_path, img_file)
            if not img_file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {img_path}")
                continue

            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"Invalid or unreadable image: {img_path}")
                continue

            try:
                print(f"Processing image: {img_path}")
                minutiae_points = minutiae_points_computer(img)
                if not minutiae_points:
                    print(f"No minutiae points detected: {img_path}")
                    continue
                feature_vector = generate_feature_vector(minutiae_points)
                if feature_vector.size > 0:
                    feature_vectors.append(feature_vector)
                    labels.append(label)
                else:
                    print(f"Empty feature vector: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        label += 1

    if not feature_vectors:
        raise ValueError(f"No valid feature vectors generated from dataset in {dataset_dir}. Check your images or processing pipeline.")

    max_length = max(len(v) for v in feature_vectors)
    feature_vectors = pad_feature_vectors(feature_vectors, max_length)

    return np.array(feature_vectors), np.array(labels)

# ============================================
# 5. TRAINING AND RECOGNITION
# ============================================

def train_recognition_system(X_train, y_train):
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(f"Training data must contain at least two classes. Found {len(unique_classes)} class(es).")
    
    clf = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    clf_best = grid_search.best_estimator_
    joblib.dump(clf_best, 'recognition_model.pkl')
    return clf_best

def evaluate_recognition_system(clf, X_test, y_test, dataset_type, max_train_len):
    """Ensure X_test features are padded to the same length as the training data."""
    X_test = pad_feature_vectors(X_test, max_train_len)
    accuracy = accuracy_score(y_test, clf.predict(X_test)) * 100
    print(f"{dataset_type} Recognition Accuracy: {accuracy:.2f}%")
    show_accuracy_window(accuracy, dataset_type)
    return accuracy

# ============================================
# 6. GUI, PLOT AND MAIN FUNCTION
# ============================================

def show_accuracy_window(accuracy, dataset_type):
    window = tk.Tk()
    window.title("Recognition Accuracy")
    window.geometry("1000x600")

    background_image_path = "Images/Template Protection.png"
    try:
        bg_image = Image.open(background_image_path)
    except FileNotFoundError:
        print(f"Error: Background image '{background_image_path}' not found.")
        return

    canvas = tk.Canvas(window, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    text_id = None
    close_button_window = None

    def resize_elements(event):
        nonlocal text_id, close_button_window

        new_width, new_height = event.width, event.height
        resized_image = bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        canvas.bg_photo = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=canvas.bg_photo)
        text_id = canvas.create_text(
            new_width // 2, int(new_height * 0.8),
            text=f"{dataset_type} Recognition Accuracy\n{accuracy:.2f}%",
            font=("Arial", 22),
            fill="white",
            anchor="center"
        )
        if close_button_window is None:
            close_button = tk.Button(
                window,
                text="Close",
                command=window.destroy,
                font=("Arial", 12),
                bg="#61dafb",
                fg="#282c34",
                padx=10,
                pady=5
            )
            close_button_window = canvas.create_window(new_width // 2, new_height - 50, anchor="center", window=close_button)
        else:
            canvas.coords(close_button_window, new_width // 2, new_height - 50)
    window.bind("<Configure>", resize_elements)

    window.mainloop()
    
def plot_minutiae_points(img, minutiae_points):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    minutiae_x = [y for x, y in minutiae_points]
    minutiae_y = [x for x, y in minutiae_points]
    plt.scatter(minutiae_x, minutiae_y, color='red', s=10)
    plt.title("Minutiae Points on Fingerprint")
    plt.show()

def select_fingerprint_and_plot():
    file_path = filedialog.askopenfilename(title="Select Fingerprint Image", filetypes=[("Image Files", "*.bmp;*.png;*.jpg;*.jpeg;*.tif")])
    if file_path:
        img = cv2.imread(file_path, 0)
        minutiae_points = minutiae_points_computer(img)
        plot_minutiae_points(img, minutiae_points)
    else:
        messagebox.showwarning("File Selection", "No file selected. Please select a valid fingerprint image.")

def main():
    base_fp_dir = os.path.join("fingerprint_dataset", "SOCOFing")
    train_fp_dirs = ["Train"]
    test_fp_dir = os.path.join(base_fp_dir, "Test")
    print("Loading fingerprint training datasets...")
    fp_train_features, fp_train_labels = [], []
    all_train_features = []

    for train_dir in train_fp_dirs:
        fp_train_features, fp_train_labels = load_dataset(os.path.join(base_fp_dir, train_dir))
        all_train_features.extend(fp_train_features)

    fp_train_features = np.array(all_train_features)
    max_train_len = max(len(v) for v in fp_train_features)  # Get the maximum length of the training vectors
    print("Training Recognition System...")
    clf_fp = train_recognition_system(fp_train_features, fp_train_labels)
    print("Evaluating recognition system on test data...")
    fp_test_features, fp_test_labels = load_dataset(test_fp_dir)
    fp_test_features = np.array(fp_test_features)

    print("Evaluating Fingerprint Dataset Recognition")
    evaluate_recognition_system(clf_fp, fp_test_features, fp_test_labels, "Fingerprint", max_train_len)

# ============================================
# 7. RUN THE PROGRAM
# ============================================

root = tk.Tk()
root.title("Fingerprint Recognition System")
plot_button = tk.Button(root, text="Select and Plot Minutiae Points", command=select_fingerprint_and_plot)
plot_button.pack(padx=20, pady=20)
root.mainloop()

if __name__ == "__main__":
    main()