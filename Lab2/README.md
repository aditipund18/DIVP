# Lab 2 – Histogram Processing, Histogram Matching and CLAHE

## Aim

To study and implement **histogram analysis, histogram matching, custom histogram matching, and Contrast Limited Adaptive Histogram Equalization (CLAHE)** for grayscale images using Python, OpenCV, NumPy, and Matplotlib.

---

## Objectives

The objectives of this practical are:

* To understand the concept of an image histogram.
* To calculate and visualize the histogram of grayscale images.
* To understand the distribution of pixel intensity values in an image.
* To perform histogram matching between a source image and a reference image.
* To use cumulative distribution functions (CDFs) for histogram matching.
* To implement histogram matching using a lookup table.
* To generate an analytical target histogram and perform custom histogram matching.
* To study Contrast Limited Adaptive Histogram Equalization (CLAHE).
* To investigate the effect of different CLAHE parameters on image enhancement.
* To compare the source, reference, matched, and enhanced images.

---

# Theory

## 1. Image Histogram

An image histogram represents the **frequency distribution of pixel intensity values** in an image.

For an 8-bit grayscale image, pixel intensity values range from:

**0 to 255**

where:

* `0` represents black.
* `255` represents white.
* Values between 0 and 255 represent different levels of gray.

The histogram shows how frequently each intensity value occurs in the image.

Mathematically, the histogram can be represented as:

**h(r_k) = n_k**

where:

* `r_k` = kth intensity value
* `n_k` = number of pixels having intensity `r_k`

A histogram can provide information about the brightness and contrast of an image.

### Interpretation of Histogram

* Histogram concentrated toward the left → image is generally dark.
* Histogram concentrated toward the right → image is generally bright.
* Narrow histogram → low contrast.
* Wide histogram → higher contrast.

In this practical, histograms are calculated using OpenCV's `cv2.calcHist()` function.

---

# 2. Histogram Normalization

For histogram matching, the histogram is normalized so that it represents a probability distribution.

The normalized histogram is:

**p(r_k) = n_k / N**

where:

* `n_k` = number of pixels having intensity `r_k`
* `N` = total number of pixels in the image.

In the implementation:

```python
src_hist = src_hist / src_hist.sum()
```

and similarly for the reference histogram.

The normalized histogram allows the cumulative distribution function to be calculated.

---

# 3. Cumulative Distribution Function (CDF)

The Cumulative Distribution Function is obtained by calculating the cumulative sum of the normalized histogram.

It is represented as:

**CDF(k) = Σ p(r_j)**

for intensity values from 0 to `k`.

In the implementation, NumPy's `np.cumsum()` function is used:

```python
src_cdf = np.cumsum(src_hist)
ref_cdf = np.cumsum(ref_hist)
```

The CDF is an important component of histogram matching because it provides the mapping between source and reference intensity levels.

---

# 4. Histogram Matching

Histogram matching, also called **histogram specification**, is an image enhancement technique in which the intensity distribution of a source image is modified so that it resembles the histogram of a reference image.

The process involves:

1. Calculate the histogram of the source image.
2. Calculate the histogram of the reference image.
3. Normalize both histograms.
4. Calculate their CDFs.
5. Compare the source CDF with the reference CDF.
6. Generate a mapping or lookup table.
7. Apply the mapping to the source image.

The resulting image has an intensity distribution that is approximately similar to the reference image.

---

# 5. Lookup Table for Histogram Matching

A lookup table is created to map every possible source intensity value from `0` to `255` to an appropriate intensity value in the reference image.

The program initializes the lookup table as:

```python
lookup = np.zeros(256, dtype=np.uint8)
```

For each source intensity value, the algorithm searches for the reference intensity whose CDF is closest to the source CDF.

The mapping is generated using:

```python
while j < 255 and ref_cdf[j] < src_cdf[i]:
    j += 1

lookup[i] = j
```

The lookup table is then applied directly to the source image:

```python
matched = lookup[source]
```

This produces the histogram-matched image.

---

# 6. Custom Histogram Matching with Analytical Target Histogram

In addition to matching a source image with an actual reference image, this practical also implements histogram matching with a mathematically defined target histogram.

The target histogram is defined using an exponential function:

**target_hist(i) = exp(-i / 50)**

for intensity values from `0` to `255`.

The target histogram is then normalized:

```python
target_hist = target_hist / target_hist.sum()
```

The target CDF is calculated using:

```python
target_cdf = np.cumsum(target_hist)
```

The source image is then mapped to this analytical target distribution using a custom histogram matching function.

The resulting image is stored as:

```python
stylized
```

This demonstrates that histogram matching does not require an actual reference image; a desired target distribution can also be mathematically specified.

---

# 7. Contrast Limited Adaptive Histogram Equalization (CLAHE)

CLAHE stands for:

**Contrast Limited Adaptive Histogram Equalization**

It is an image enhancement technique used to improve local contrast while avoiding excessive amplification of noise.

Unlike global histogram equalization, CLAHE divides an image into small local regions called **tiles**.

Histogram equalization is performed separately within each tile.

A contrast limit is applied to prevent excessive amplification of noise.

---

## CLAHE Parameters

Two important parameters are used in this practical:

### 1. Clip Limit

The `clipLimit` controls the amount of contrast enhancement.

A higher clip limit generally allows stronger contrast enhancement.

The practical uses:

```python
clipLimit=2.0
```

for the main CLAHE operation.

---

### 2. Tile Grid Size

The `tileGridSize` determines the number or size of local regions used for adaptive histogram processing.

The practical initially uses:

```python
tileGridSize=(8, 8)
```

This means that the image is divided into a grid of local regions for contrast enhancement.

---

# 8. CLAHE Parameter Sweep

To study the effect of CLAHE parameters, different combinations of:

### Clip Limits

```text
1
2
4
```

and

### Tile Sizes

```text
4
8
16
```

are tested.

This results in:

**3 × 3 = 9 different CLAHE configurations**

The results are displayed using a 3 × 3 subplot arrangement.

This allows the visual effect of different `clipLimit` and `tileGridSize` combinations to be compared.

---

# Software and Libraries Used

The following software and Python libraries were used:

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **OS module**

### Libraries Imported

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```

---

# Input Images

Three grayscale images are used in this practical:

### 1. Source 1

```text
source1.jpg
```

This image is used as the primary source image for histogram analysis, histogram matching, custom histogram matching, and CLAHE.

### 2. Source 2

```text
source2.jpg
```

This image is also matched with the reference image to demonstrate histogram matching on another source image.

### 3. Reference Image

```text
reference.jpg
```

This image provides the desired intensity distribution for histogram matching.

---

# Implementation

## Step 1 – Determine the Correct Image Folder

The program determines the directory in which the Python script is located:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

This ensures that the image files can be loaded correctly regardless of the current working directory.

The image paths are then constructed using:

```python
os.path.join()
```

---

## Step 2 – Load Images

All three images are loaded as grayscale images using OpenCV:

```python
source1 = cv2.imread(source1_path, cv2.IMREAD_GRAYSCALE)

source2 = cv2.imread(source2_path, cv2.IMREAD_GRAYSCALE)

reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
```

The program also checks whether each image has been successfully loaded.

If an image cannot be loaded, a `FileNotFoundError` is generated.

---

# Step 3 – Plot Histogram

A separate function called `plot_hist()` is created to calculate and display an image histogram.

The histogram is calculated using:

```python
cv2.calcHist()
```

The histogram contains 256 bins corresponding to the possible grayscale intensity values.

The X-axis represents:

**Pixel Intensity**

The Y-axis represents:

**Frequency**

---

# Step 4 – Histogram Matching

A function called:

```python
histogram_matching(source, reference)
```

is implemented.

The function performs the following operations:

1. Calculate the source histogram.
2. Calculate the reference histogram.
3. Normalize both histograms.
4. Calculate the source CDF.
5. Calculate the reference CDF.
6. Create a lookup table.
7. Map source intensity values to reference intensity values.
8. Apply the lookup table to the source image.

The function is applied to both source images:

```python
matched1 = histogram_matching(source1, reference)

matched2 = histogram_matching(source2, reference)
```

Therefore, both source images are transformed to have an intensity distribution similar to the reference image.

---

# Step 5 – Generate Analytical Target Histogram

An analytical target histogram is generated using:

```python
target_hist[i] = np.exp(-i / 50)
```

The histogram is normalized and its CDF is calculated.

This creates a mathematically defined target intensity distribution.

---

# Step 6 – Custom Histogram Matching

A function called:

```python
histogram_match_custom(source, target_cdf)
```

is used to match the source image to the analytical target CDF.

The function calculates the source histogram and CDF and generates a lookup table based on the target CDF.

The resulting stylized image is obtained using:

```python
stylized = histogram_match_custom(
    source1,
    target_cdf
)
```

---

# Step 7 – Apply CLAHE

CLAHE is created using:

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
```

The enhancement is applied to `source1` using:

```python
clahe_img = clahe.apply(source1)
```

This produces the locally contrast-enhanced image.

---

# Step 8 – CLAHE Parameter Sweep

Different combinations of clip limits and tile sizes are tested.

The values used are:

```python
clip_limits = [1, 2, 4]

tile_sizes = [4, 8, 16]
```

A nested loop creates nine different CLAHE results.

Each result is displayed with its corresponding parameter values:

```text
Clip=1 Tile=4
Clip=1 Tile=8
Clip=1 Tile=16

Clip=2 Tile=4
Clip=2 Tile=8
Clip=2 Tile=16

Clip=4 Tile=4
Clip=4 Tile=8
Clip=4 Tile=16
```

The resulting images are displayed in a 3 × 3 grid.

---

# Step 9 – Histogram Comparison

The practical creates a comparison figure containing:

### Images

1. Source Image
2. Reference Image
3. Histogram-Matched Image

### Histograms

4. Source Histogram
5. Reference Histogram
6. Matched Histogram

This allows both the visual appearance and intensity distributions of the images to be compared.

---

# Step 10 – Display Results

Matplotlib is used to display the histogram plots and CLAHE parameter sweep.

OpenCV windows are also used to display:

* Source 1
* Source 2
* Reference
* Matched 1
* Matched 2
* Stylized image
* CLAHE image

The program waits for a key press using:

```python
cv2.waitKey(0)
```

and then closes the OpenCV windows using:

```python
cv2.destroyAllWindows()
```

---

# Algorithm / Procedure

1. Start the program.
2. Determine the directory containing the Python script.
3. Construct paths for Source 1, Source 2, and Reference images.
4. Load all images in grayscale format.
5. Verify that all images have been loaded successfully.
6. Calculate and display the histogram of the images.
7. Normalize the source and reference histograms.
8. Calculate their cumulative distribution functions.
9. Generate a lookup table for histogram matching.
10. Apply the lookup table to Source 1.
11. Apply the same process to Source 2.
12. Generate an analytical target histogram using an exponential function.
13. Calculate its CDF.
14. Perform custom histogram matching of Source 1 with the analytical target CDF.
15. Create a CLAHE object using `clipLimit = 2.0` and `tileGridSize = (8,8)`.
16. Apply CLAHE to Source 1.
17. Perform a parameter sweep using clip limits `1, 2, 4` and tile sizes `4, 8, 16`.
18. Generate nine CLAHE results.
19. Display the source, reference, and matched images.
20. Display their corresponding histograms.
21. Display all processed images using OpenCV.
22. Compare the results and observe the effect of histogram processing.
23. End the program.

---

# Results and Observations

## Histogram Analysis

The histograms of the source and reference images show their respective pixel intensity distributions.

The histogram provides information about:

* Brightness
* Contrast
* Distribution of intensity values
* Dominant pixel intensities

---

## Histogram Matching

After histogram matching, the intensity distribution of the source image becomes similar to that of the reference image.

The matched image therefore has a visual appearance influenced by the reference image's intensity distribution.

The same process is performed for both `source1.jpg` and `source2.jpg`.

---

## Custom Histogram Matching

The custom histogram matching demonstrates that an image can also be transformed according to a mathematically defined target histogram.

The target histogram in this practical follows an exponential distribution:

**T(i) = exp(-i / 50)**

This produces a stylized intensity distribution for the source image.

---

## CLAHE

CLAHE enhances local contrast by processing small regions of the image independently.

The parameter sweep demonstrates that changing the `clipLimit` and `tileGridSize` affects the amount and nature of contrast enhancement.

The nine combinations provide a visual comparison of different CLAHE settings.

---

# Parameter Summary

| Technique                 | Parameters Used                 | Purpose                                          |
| ------------------------- | ------------------------------- | ------------------------------------------------ |
| Histogram Matching        | Source + Reference              | Match source intensity distribution to reference |
| Custom Histogram Matching | Exponential target distribution | Match source to an analytical target             |
| CLAHE                     | Clip Limit = 2.0                | Limit contrast amplification                     |
| CLAHE                     | Tile Grid = 8 × 8               | Define local processing regions                  |
| CLAHE Sweep               | Clip = 1, 2, 4                  | Study contrast enhancement                       |
| CLAHE Sweep               | Tile = 4, 8, 16                 | Study local region size                          |

---

# Output

The practical generates the following major outputs:

### Histogram Matching

* Matched version of `source1.jpg`
* Matched version of `source2.jpg`

### Custom Histogram Matching

* Stylized image based on the analytical target histogram

### CLAHE

* CLAHE-enhanced image using `clipLimit=2.0` and `tileGridSize=(8,8)`
* Nine CLAHE results generated using different parameter combinations

### Histogram Comparison

A comparison of:

* Source image
* Reference image
* Matched image
* Source histogram
* Reference histogram
* Matched histogram

---

# Result

Histogram analysis, histogram matching, custom histogram matching, and CLAHE-based contrast enhancement were successfully implemented using Python and OpenCV.

The source images were successfully matched with a reference image using CDF-based intensity mapping. A custom analytical target histogram was also used to generate a stylized output. CLAHE was applied with different clip limits and tile sizes to study their effect on local contrast enhancement.

---

# Conclusion

In this practical, the concepts of **image histograms, histogram normalization, cumulative distribution functions, histogram matching, histogram specification, and CLAHE** were studied and implemented.

The histogram of grayscale images was calculated and visualized to understand their pixel intensity distributions. Histogram matching was then performed by comparing the CDF of a source image with that of a reference image and generating a lookup table for intensity mapping.

A custom histogram matching technique was also implemented using an analytically generated exponential target distribution. This demonstrated how a desired intensity distribution can be specified mathematically rather than relying only on an existing reference image.

Finally, **Contrast Limited Adaptive Histogram Equalization (CLAHE)** was applied to enhance local image contrast. Different combinations of clip limits and tile sizes were tested through a parameter sweep, allowing the effect of these parameters on image enhancement to be observed.

Thus, the practical successfully demonstrated different histogram-based image enhancement and contrast improvement techniques using **Python, OpenCV, NumPy, and Matplotlib**.
