# Biometric-Template-Security-using-Fuzzy-Vault-Encryption

**Developed by:** Sagnik

## Overview

This repository implements a robust biometric security system utilising the Fuzzy Vault Technique to lock and encrypt biometric templates such as fingerprints and palmprints with a unique key. The system employs Polynomial Reconstruction and Lagrange interpolation algorithms to securely unlock and access the encrypted biometric data.

## Features

- **Biometric Template Processing:** The system processes biometric images (fingerprints and palmprints) through segmentation, normalization, enhancement, binarization, and thinning.
- **Minutiae Extraction:** Extracts minutiae points from processed biometric images and computes their positions and orientations.
- **Vault Construction:** Uses the extracted minutiae points to construct a Fuzzy Vault, which includes encoding the minutiae data and generating a secure key using polynomial encryption.
- **Visualisation:** Plots a 2D graph of the vault points for fingerprint and palmprint data.

## Requirements

- Python 3.x
- NumPy
- OpenCV
- scikit-image
- scipy
- scikit-learn
- galois
- crc
- matplotlib
- fingerprint_enhancer (custom module)

## Installation

To install the necessary Python packages, use pip:

```bash
pip install numpy opencv-python scikit-image scipy scikit-learn galois crc matplotlib
```

## Usage

1. **Prepare your biometric images:**
   Ensure that you have fingerprint and palmprint images in TIFF format named `fingerprint.tif` and `palmprint.tif`, respectively.

2. **Run the main script:**

   ```bash
   python main.py
   ```

   The script will process the images, compute minutiae points, construct the fuzzy vaults, and visualize the vault points in a 2D scatter plot.

## Code Explanation

- **Preprocessing Functions:**
  - `do_segmentation(img)`: Segments the image into regions of interest.
  - `do_normalization(segmented_image)`: Normalizes the image to a desired mean and variance.
  - `do_enhancement(normalized_image)`: Enhances the fingerprint image using the custom `fingerprint_enhancer` module.
  - `do_binarization(enhanced_image)`: Binarizes the enhanced image using Otsu's thresholding.
  - `do_thinning(binarized_image)`: Performs skeletonization to thin the binary image.
  - `preprocessing(img)`: Combines all preprocessing steps.

- **Minutiae Extraction:**
  - `ridge_orientation(img, thinned_image)`: Computes the ridge orientation map.
  - `crossing_number(i, j, thinned_image)`: Calculates the crossing number at a given pixel.
  - `false_minutiae_removal(img, thinned_image)`: Removes false minutiae based on grayscale variance.
  - `minutiae(img, thinned_image, orientation_map)`: Extracts minutiae points with their orientation.
  - `minutiae_points_computer(img)`: Computes minutiae points for a given image.

- **Vault Construction:**
  - `vault_constructor(minutiae_points, img)`: Constructs the fuzzy vault and encodes minutiae points.

- **Visualization:**
  - `main()`: Reads biometric images, computes minutiae points, constructs fuzzy vaults, and plots the 2D graph of vault points.

## Contributing

Contributions to the project are welcome. Please fork the repository and submit a pull request with your improvements or fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or questions, please reach out to sagnikgraviton847@gmail.com