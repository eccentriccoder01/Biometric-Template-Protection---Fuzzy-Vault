import numpy as np
import cv2    
import math
import fingerprint_enhancer as fe
from skimage.morphology import skeletonize
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import galois
from crc import Calculator, Crc16
import matplotlib.pyplot as plt
import warnings
import binascii

block_size = 16
warnings.filterwarnings("ignore")
r = 20 #Minutiae Points
s = 180 #Chaff Points

def do_segmentation(img):
    shap = img.shape
    segmented_image = img.copy()
    seg_mask = np.ones(shap)
    threshold = np.var(img) * 0.1
    row, col = img.shape
    for i in range(0, row, block_size):
        for j in range(0, col, block_size):
            x, y = min(row, i + block_size), min(col, j + block_size)
            loc_gs_var = np.var(img[i:x, j:y])
            if loc_gs_var <= threshold:
                seg_mask[i:x, j:y] = 0
    var = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size * 2, block_size * 2))
    seg_mask = cv2.erode(seg_mask, var, iterations=1)   
    seg_mask = cv2.dilate(seg_mask, var, iterations=1)
    segmented_image[seg_mask == 0] = 255
    return segmented_image

def do_normalization(segmented_image):
    desired_mean, desired_variance = 100.0, 8000.0
    estimated_mean, estimated_variance = np.mean(segmented_image), np.var(segmented_image)
    normalized_image = np.where(segmented_image > estimated_mean,
                                np.sqrt((segmented_image - estimated_mean)**2 * (desired_variance / estimated_variance)) + desired_mean,
                                desired_mean - np.sqrt((segmented_image - estimated_mean)**2 * (desired_variance / estimated_variance)))
    return normalized_image

def do_enhancement(normalized_image):
    return fe.enhance_Fingerprint(normalized_image)

def do_binarization(enhanced_image):
    _, binarized_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized_image

def do_thinning(binarized_image):
    return np.where(skeletonize(binarized_image / 255), 0.0, 1.0)

def preprocessing(img):
    segmented_image = do_segmentation(img)
    normalized_image = do_normalization(segmented_image)
    enhanced_image = do_enhancement(normalized_image)
    binarized_image = do_binarization(enhanced_image)
    thinned_image = do_thinning(binarized_image)
    return img, thinned_image

def ridge_orientation(img, thinned_image):
    scale, delta = 1, 0
    bs = (block_size * 2) + 1
    G_x = cv2.Sobel(thinned_image / 255, cv2.CV_64F, 0, 1, ksize=3, scale=scale, delta=delta)
    G_y = cv2.Sobel(thinned_image / 255, cv2.CV_64F, 1, 0, ksize=3, scale=scale, delta=delta)
    gaussian_directions_x = cv2.GaussianBlur(G_x, (bs, bs), 1.0)
    gaussian_directions_y = cv2.GaussianBlur(G_y, (bs, bs), 1.0)
    orientation_map = 0.5 * (np.arctan2(gaussian_directions_x, gaussian_directions_y)) + (0.5 * np.pi)
    return orientation_map

def crossing_number(i, j, thinned_image):
    if thinned_image[i, j] != 0.0:
        return 2.0
    sum_val = 0.0
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    pixel_values = [thinned_image[i+x, j+y] for x, y in offsets]
    sum_val += np.sum(np.abs(np.diff(pixel_values)))
    return sum_val // 2

def false_minutiae_removal(img, thinned_image):
    global_grayscale_variance = np.var(thinned_image) * 0.1
    seg_mask = np.zeros_like(thinned_image)
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            local_grayscale_variance = np.var(thinned_image[i:i + block_size, j:j + block_size])
            if local_grayscale_variance > global_grayscale_variance:
                seg_mask[i:i + block_size, j:j + block_size] = 1
    return seg_mask

def minutiae(img, thinned_image, orientation_map):
    minutiae_points = {}
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            cn = crossing_number(i, j, thinned_image)
            if cn in [1, 3]:
                minutiae_points[(i, j)] = (cn, orientation_map[i, j])
    seg_mask = false_minutiae_removal(img, thinned_image)
    return minutiae_points

def minutiae_points_computer(img):
    img, thinned_image = preprocessing(img)
    orientation_map = ridge_orientation(img, thinned_image)
    minutiae_points = minutiae(img, thinned_image, orientation_map)
    return minutiae_points

def vault_constructor(minutiae_points, img):
    X_orig = np.array([key[0] for key in minutiae_points.keys()][:r])
    Y_orig = np.array([key[1] for key in minutiae_points.keys()][:r])
    Theta_orig = np.round(np.array([value[1] for value in minutiae_points.values()][:r]) * 100).astype(int)
    if len(X_orig) < r:
        print("Not enough minutiae points")
        return None, None
    
    scaler_x = MinMaxScaler((0, 2**6))
    scaler_y = MinMaxScaler((0, 2**6))
    scaler_theta = MinMaxScaler((0, 2**4))
    
    X_encoded = scaler_x.fit_transform(X_orig.reshape(-1, 1)).flatten().astype(int)
    Y_encoded = scaler_y.fit_transform(Y_orig.reshape(-1, 1)).flatten().astype(int)
    Theta_encoded = scaler_theta.fit_transform(Theta_orig.reshape(-1, 1)).flatten().astype(int)
    
    GF = galois.GF(2**16)
    key = np.random.randint(2, size=16*16, dtype='uint16')
    crc_input = key.tobytes()
    crcsum = binascii.crc_hqx(crc_input, 0xFFFF)
    
    key_dash = np.concatenate((key, np.fromiter(f"{crcsum:016b}", dtype=int)))
    num_groups = len(key_dash) // 16
    key_final = [int(''.join(map(str, key_dash[16*i:16*(i+1)])), 2) for i in range(num_groups)]
    
    encode_poly = galois.Poly(key_final, field=GF)

    vault = {}
    for i in range(r):
        vault[(X_orig[i], Y_orig[i], Theta_orig[i])] = (int(X_encoded[i]), int(Y_encoded[i]), int(Theta_encoded[i]))
    return vault, (X_orig, Y_orig)


def main():
    img_fp = cv2.imread("fingerprint.tif", 0)
    minutiae_points_fp = minutiae_points_computer(img_fp)
    Vault_fp, vault_fp_print = vault_constructor(minutiae_points_fp, img_fp)
    
    img_pp = cv2.imread("palmprint.tif", 0)
    minutiae_points_pp = minutiae_points_computer(img_pp)
    Vault_pp, vault_pp_print = vault_constructor(minutiae_points_pp, img_pp)
    
    x = np.concatenate((vault_fp_print[0], vault_pp_print[0]))
    y = np.concatenate((vault_fp_print[1], vault_pp_print[1]))
    plt.scatter(x, y)
    plt.show()
    
if __name__ == "__main__":
    main()
