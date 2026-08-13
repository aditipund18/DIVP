# Lab 1 – Image Intensity Transformations

## Aim

To study and implement basic image intensity transformation techniques, namely **Negative Transformation, Power-Law (Gamma) Transformation, and Logarithmic Transformation**, using Python and OpenCV, and to observe their effect on grayscale images.

---

## Objectives

The objectives of this practical are:

* To understand the concept of image intensity transformations.
* To implement the negative transformation of a grayscale image.
* To implement Power-Law (Gamma) transformation.
* To implement logarithmic transformation.
* To understand how different transformation functions modify pixel intensity values.
* To implement the transformations manually using pixel matrices.
* To implement the same transformations using Python libraries such as OpenCV and NumPy.
* To visualize and compare the original image with the transformed images.

---

## Theory

Image intensity transformation is one of the fundamental techniques in **Digital Image Processing**. It is used to modify the intensity values of pixels in an image in order to improve its appearance or highlight specific details.

An intensity transformation maps an input pixel intensity `r` to an output intensity `s`.

The general form can be represented as:

**s = T(r)**

where:

* `r` = input pixel intensity
* `s` = output pixel intensity
* `T` = transformation function

For an 8-bit grayscale image, pixel intensity values generally range from **0 to 255**, where:

* `0` represents black.
* `255` represents white.
* Values between 0 and 255 represent different levels of gray.

In this practical, three important intensity transformations are implemented.

---

## 1. Negative Transformation

Image negative transformation reverses the intensity values of an image.

The transformation is given by:

**s = (L - 1) - r**

For an 8-bit grayscale image:

**s = 255 - r**

where:

* `L = 256` is the number of possible intensity levels.
* `r` is the original pixel value.
* `s` is the transformed pixel value.

For example:

| Original Pixel | Negative Pixel |
| -------------: | -------------: |
|              0 |            255 |
|             50 |            205 |
|            100 |            155 |
|            150 |            105 |
|            255 |              0 |

This transformation converts dark regions into bright regions and bright regions into dark regions.

It is particularly useful when the details of interest are present in dark regions of an image.

---

## 2. Power-Law (Gamma) Transformation

Power-Law transformation is also known as **Gamma transformation**.

The general equation is:

**s = c × r^γ**

In the implementation, the input pixel values are first normalized to the range `[0, 1]`:

**r_norm = r / 255**

The transformation is then applied:

**s_norm = c × (r_norm)^γ**

Finally, the result is scaled back to the range `[0, 255]`.

### Effect of Gamma

The value of `γ` controls the brightness of the output image.

* **γ < 1** → image becomes brighter.
* **γ = 1** → approximately preserves the original intensity.
* **γ > 1** → image becomes darker.

In the OpenCV implementation, a gamma value of:

**γ = 0.5**

is used, which produces a brighter image.

A **Lookup Table (LUT)** is used with OpenCV to efficiently apply the transformation to all possible 8-bit intensity values.

---

## 3. Logarithmic Transformation

Logarithmic transformation is represented by:

**s = c × log(1 + r)**

where:

* `r` = input pixel intensity
* `s` = output pixel intensity
* `c` = scaling constant

The constant is calculated as:

**c = 255 / log(1 + r_max)**

where `r_max` is the maximum pixel intensity in the image.

The logarithmic transformation expands the lower intensity values and compresses the higher intensity values.

Therefore, it can be useful for enhancing details in darker regions of an image.

---

# Software and Libraries Used

The following software and Python libraries were used:

* **Python**
* **OpenCV (`cv2`)**
* **NumPy**
* **Matplotlib**

### Python Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

# Implementation

The practical was implemented in two stages:

1. **Manual implementation using a pixel matrix**
2. **Image processing implementation using OpenCV, NumPy and Matplotlib**

---

## Part A – Manual Matrix Implementation

A user-defined image matrix is taken as input.

The program asks the user to enter:

* Number of rows
* Number of columns
* Pixel values between 0 and 255
* Gamma value for Power-Law transformation

The input matrix is then processed using three transformation functions.

### 1. Manual Natural Logarithm

A custom `manual_ln()` function is implemented using the Taylor series.

The formula used is:

**ln(x) = 2 × [z + z³/3 + z⁵/5 + ...]**

where:

**z = (x - 1) / (x + 1)**

The function uses multiple terms of the series to calculate the natural logarithm without directly using Python's built-in logarithm function.

---

### 2. Negative Transformation

The negative transformation is implemented using:

```python
new_pixel = max_intensity - pixel
```

For an 8-bit image:

```text
new_pixel = 255 - pixel
```

The transformation is applied to every pixel of the input matrix.

---

### 3. Power-Law Transformation

The input pixel is first normalized:

```python
r_norm = pixel / max_intensity
```

The Power-Law transformation is then applied:

```python
s_norm = c * (r_norm ** gamma)
```

The result is converted back to the intensity range `[0, 255]`.

---

### 4. Logarithmic Transformation

A scaling constant is calculated using the maximum intensity value:

```python
c = max_intensity / manual_ln(1 + max_intensity)
```

The logarithmic transformation is then applied to every pixel:

```python
s = c * manual_ln(1 + pixel)
```

The resulting values are clamped to the valid range of 0 to 255.

---

# Part B – Image Processing Using OpenCV

A grayscale image is loaded using OpenCV:

```python
img = cv2.imread("Lab1/image.jpg", cv2.IMREAD_GRAYSCALE)
```

The three transformations are then applied to the actual image.

---

## Negative Transformation Using OpenCV

The image negative is generated using:

```python
neg_img = cv2.bitwise_not(img)
```

This reverses the intensity of every pixel.

---

## Power-Law Transformation Using OpenCV

A Lookup Table (LUT) containing the transformed values for all 256 possible grayscale intensities is generated.

The implementation uses:

```python
gamma = 0.5
```

The lookup table is created using:

```python
lut = np.array(
    [((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]
).astype("uint8")
```

The LUT is then applied using:

```python
gamma_img = cv2.LUT(img, lut)
```

Using a lookup table makes the transformation efficient because the calculation does not need to be repeated independently for every pixel.

---

## Logarithmic Transformation Using OpenCV and NumPy

The maximum pixel value in the image is obtained:

```python
max_pixel_val = float(np.max(img))
```

The scaling constant is calculated as:

```python
c = 255 / np.log(1 + max_pixel_val)
```

The logarithmic transformation is then applied using NumPy:

```python
log_img = np.array(
    c * np.log1p(img.astype(np.float32)),
    dtype=np.uint8
)
```

`np.log1p()` is used to calculate `log(1 + x)` accurately.

---

# Visualization

Matplotlib is used to display the original and transformed images side-by-side.

Four images are displayed:

1. Original Image
2. Negative Transformation
3. Power-Law (Gamma) Transformation
4. Log Transformation

The visualization is created using four subplots:

```python
plt.subplot(1, 4, 1)
plt.subplot(1, 4, 2)
plt.subplot(1, 4, 3)
plt.subplot(1, 4, 4)
```

This allows the effect of each transformation to be compared with the original grayscale image.

---

# Algorithm / Procedure

1. Start the program.
2. Define the input grayscale image or pixel matrix.
3. Accept the required matrix dimensions and pixel values for the manual implementation.
4. Accept a Gamma value from the user.
5. Apply Negative Transformation.
6. Apply Power-Law (Gamma) Transformation.
7. Apply Logarithmic Transformation.
8. Display the transformed pixel matrices.
9. Load the grayscale image using OpenCV.
10. Apply Negative Transformation using `cv2.bitwise_not()`.
11. Create a Lookup Table for Power-Law transformation.
12. Apply the LUT using `cv2.LUT()`.
13. Calculate the scaling constant for logarithmic transformation.
14. Apply the logarithmic transformation using NumPy.
15. Display the original and transformed images using Matplotlib.
16. Compare the visual changes produced by each transformation.
17. End the program.

---

# Input

The input for the practical consists of:

* A grayscale image stored as `image.jpg`.
* Pixel intensity values ranging from **0 to 255**.
* A user-defined Gamma value for the manual Power-Law transformation.

For the OpenCV implementation, the image is loaded from:

```text
Lab1/image.jpg
```

---

# Output

The program produces the following results:

### 1. Original Image

The original grayscale image is displayed without any intensity transformation.

### 2. Negative Image

The intensity values are reversed, producing a negative version of the original image.

### 3. Power-Law Transformed Image

With `γ = 0.5`, the image becomes brighter, particularly enhancing lower-intensity regions.

### 4. Logarithmically Transformed Image

The logarithmic transformation expands lower intensity values and compresses higher intensity values, enhancing details in darker regions.

---

# Results and Observations

The three intensity transformation techniques produce different effects on the grayscale image.

| Transformation        | Effect                                                         |
| --------------------- | -------------------------------------------------------------- |
| Negative              | Reverses dark and bright intensity values                      |
| Power-Law (`γ = 0.5`) | Brightens the image and enhances lower intensity values        |
| Logarithmic           | Expands darker intensity levels and compresses brighter levels |

### Observations

* Negative transformation produces a complete reversal of image intensities.
* Power-Law transformation provides control over image brightness through the Gamma value.
* A Gamma value less than 1 brightens the image.
* Logarithmic transformation is useful for enhancing details in darker regions.
* Different transformations can be selected depending on the type of image enhancement required.
* The manual matrix implementation helps understand the mathematical operation at the pixel level.
* The OpenCV implementation demonstrates how the same transformations can be efficiently applied to an actual grayscale image.

---

# Result

The Negative, Power-Law (Gamma), and Logarithmic intensity transformations were successfully implemented using both manual pixel-matrix operations and Python image-processing libraries.

The transformed images were successfully generated and visualized alongside the original grayscale image.

---

# Conclusion

In this practical, three fundamental image intensity transformation techniques — **Negative, Power-Law (Gamma), and Logarithmic transformation** — were studied and implemented.

The negative transformation was used to reverse the intensity levels of the image. The Power-Law transformation demonstrated how the Gamma parameter can be used to control image brightness, with `γ = 0.5` producing a brighter image. The logarithmic transformation demonstrated the enhancement of lower intensity values while compressing higher intensity values.

The practical also provided an understanding of how intensity transformations can be implemented manually using pixel matrices as well as efficiently using **OpenCV, NumPy, and Matplotlib**.

Thus, the practical successfully demonstrated the importance of intensity transformation techniques in **Digital Image Processing** for image enhancement and analysis.
