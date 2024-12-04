# Biometric-Template-Security-using-Fuzzy-Vault-Encryption

**Developed by:** Sagnik

## Overview

This repository implements a robust biometric security system utilising the Fuzzy Vault Technique to lock and encrypt biometric templates such as fingerprints and palmprints with a unique key. The system employs Polynomial Reconstruction and Lagrange interpolation algorithms to securely unlock and access the encrypted biometric data.

## Features

- **Biometric Template Processing:** The system processes biometric images (fingerprints and palmprints) through segmentation, normalization, enhancement, binarization, and thinning.
- **Minutiae Extraction:** Extracts minutiae points from processed biometric images and computes their positions and orientations.
- **Vault Construction:** Uses the extracted minutiae points to construct a Fuzzy Vault, which includes encoding the minutiae data and generating a secure key using polynomial encryption.
- **Visualisation:** Plots a 2D graph of the vault points for fingerprint and palmprint data.
- **Recognition System:** Trains an SVM classifier with a hyperparameter grid search using cross-validation, Evaluates recognition accuracy on test datasets.


## Requirements

- Python 3.x
- NumPy
- OpenCV
- scikit-image
- scipy
- scikit-learn
- joblib
- crc
- matplotlib
- tkinter
- PIL (Python Imaging Library)

## Installation

To install the necessary Python packages, use pip:

```bash
pip install numpy opencv-python scikit-image scikit-learn joblib matplotlib pillow
```

## Usage

1. **Prepare your Dataset:**

   Organize fingerprint images in subdirectories within a dataset folder.
   Each subdirectory represents a class/label.

2. **Run the main script:**

   ```bash
   python fuzzy_vault.py
   ```

   The script opens the GUI for fingerprint minutiae plotting, processes the dataset, extracts features, trains an SVM model, and evaluates its performance.

## Code Explanation

- **Preprocessing Functions:**
  - `do_segmentation(img)`: Segments the image into blocks and removes low-variance regions.
  - `do_normalization(segmented_image)`: Normalizes image intensity to a desired mean and variance.
  - `do_thinning(binarized_image)`: Thins the binary image using skeletonization.
- **Feature Extraction:**
  - `minutiae_points_computer(img)`: Extracts minutiae points from the fingerprint image.
  - `generate_feature_vector(minutiae_points)`: Converts minutiae points into a feature vector.
  - `pad_feature_vectors(feature_vectors, max_length)`: Ensures uniform feature vector lengths.
- **Dataset Handling:**
  - `load_dataset(dataset_dir, is_fingerprint=True)`: Loads fingerprint images, processes them, and generates feature vectors.
- **Training and Evaluation:**
  - `train_recognition_system(X_train, y_train)`: Trains an SVM classifier using GridSearchCV.
  - `evaluate_recognition_system(clf, X_test, y_test, dataset_type)`: Evaluates recognition accuracy on the test dataset.
- **Visualization:**
  - `plot_minutiae_points(img, minutiae_points)`: Visualizes minutiae points on the fingerprint image.
- **Graphical Interface:**
  - GUI to select a fingerprint image and visualize its minutiae points.

## Contributing

Contributions to the project are welcome. Please fork the repository and submit a pull request with your improvements or fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or questions, please reach out to sagnikgraviton847@gmail.com