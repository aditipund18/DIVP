import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# GET CORRECT FOLDER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Python file:", __file__)
print("Image folder:", BASE_DIR)


# ============================================================
# IMAGE PATHS
# ============================================================

source1_path = os.path.join(BASE_DIR, "source1.jpg")
source2_path = os.path.join(BASE_DIR, "source2.jpg")
reference_path = os.path.join(BASE_DIR, "reference.jpg")


print("Source 1:", source1_path)
print("Source 2:", source2_path)
print("Reference:", reference_path)


# ============================================================
# READ IMAGES
# ============================================================

source1 = cv2.imread(source1_path, cv2.IMREAD_GRAYSCALE)
source2 = cv2.imread(source2_path, cv2.IMREAD_GRAYSCALE)
reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)


# ============================================================
# CHECK IMAGES
# ============================================================

if source1 is None:
    raise FileNotFoundError(
        "Could not load source1.jpg:\n" + source1_path
    )

if source2 is None:
    raise FileNotFoundError(
        "Could not load source2.jpg:\n" + source2_path
    )

if reference is None:
    raise FileNotFoundError(
        "Could not load reference.jpg:\n" + reference_path
    )


print("\nAll images loaded successfully!")

print("Source1:", source1.shape, source1.dtype)
print("Source2:", source2.shape, source2.dtype)
print("Reference:", reference.shape, reference.dtype)


# ============================================================
# PLOT HISTOGRAM
# ============================================================

def plot_hist(img, title):

    hist = cv2.calcHist(
        [img],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.plot(hist)
    plt.title(title)
    plt.xlim([0, 256])
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")


# ============================================================
# HISTOGRAM MATCHING
# ============================================================

def histogram_matching(source, reference):

    src_hist = cv2.calcHist(
        [source],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    ref_hist = cv2.calcHist(
        [reference],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    # Normalize
    src_hist = src_hist / src_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # CDF
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)

    # Lookup table
    lookup = np.zeros(256, dtype=np.uint8)

    j = 0

    for i in range(256):

        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1

        lookup[i] = j

    # Apply lookup table
    matched = lookup[source]

    return matched


# ============================================================
# MATCH SOURCE IMAGES WITH REFERENCE
# ============================================================

matched1 = histogram_matching(source1, reference)
matched2 = histogram_matching(source2, reference)


# ============================================================
# ANALYTICAL TARGET HISTOGRAM
# ============================================================

target_hist = np.zeros(256, dtype=np.float64)

for i in range(256):
    target_hist[i] = np.exp(-i / 50)

target_hist = target_hist / target_hist.sum()

target_cdf = np.cumsum(target_hist)


# ============================================================
# CUSTOM HISTOGRAM MATCHING
# ============================================================

def histogram_match_custom(source, target_cdf):

    src_hist = cv2.calcHist(
        [source],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    src_hist = src_hist / src_hist.sum()

    src_cdf = np.cumsum(src_hist)

    lookup = np.zeros(256, dtype=np.uint8)

    j = 0

    for i in range(256):

        while j < 255 and target_cdf[j] < src_cdf[i]:
            j += 1

        lookup[i] = j

    return lookup[source]


stylized = histogram_match_custom(
    source1,
    target_cdf
)


# ============================================================
# CLAHE
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_img = clahe.apply(source1)


# ============================================================
# CLAHE PARAMETER SWEEP
# ============================================================

clip_limits = [1, 2, 4]
tile_sizes = [4, 8, 16]

plt.figure(figsize=(12, 8))

count = 1

for clip in clip_limits:

    for tile in tile_sizes:

        clahe_test = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(tile, tile)
        )

        clahe_result = clahe_test.apply(source1)

        plt.subplot(3, 3, count)

        plt.imshow(
            clahe_result,
            cmap="gray"
        )

        plt.title(
            "Clip=" + str(clip) +
            " Tile=" + str(tile)
        )

        plt.axis("off")

        count += 1

plt.suptitle("CLAHE Parameter Sweep")
plt.tight_layout()


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

plt.figure(figsize=(12, 8))

# Source image
plt.subplot(2, 3, 1)
plt.imshow(source1, cmap="gray")
plt.title("Source1")
plt.axis("off")

# Reference image
plt.subplot(2, 3, 2)
plt.imshow(reference, cmap="gray")
plt.title("Reference")
plt.axis("off")

# Matched image
plt.subplot(2, 3, 3)
plt.imshow(matched1, cmap="gray")
plt.title("Matched")
plt.axis("off")

# Source histogram
plt.subplot(2, 3, 4)
plot_hist(source1, "Source Histogram")

# Reference histogram
plt.subplot(2, 3, 5)
plot_hist(reference, "Reference Histogram")

# Matched histogram
plt.subplot(2, 3, 6)
plot_hist(matched1, "Matched Histogram")

plt.tight_layout()


# ============================================================
# SHOW ALL RESULTS
# ============================================================

plt.show()


# ============================================================
# OPENCV WINDOWS
# ============================================================

cv2.imshow("Source1", source1)
cv2.imshow("Source2", source2)
cv2.imshow("Reference", reference)

cv2.imshow("Matched1", matched1)
cv2.imshow("Matched2", matched2)

cv2.imshow("Stylized", stylized)

cv2.imshow("CLAHE", clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()