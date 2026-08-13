def manual_ln(x, terms=50):
    """
    Computes natural logarithm ln(x) manually using Taylor Series.
    Formula: ln(x) = 2 * [ z + (z^3)/3 + (z^5)/5 + ... ] where z = (x-1)/(x+1)
    """
    if x <= 0:
        return 0.0
    
    z = (x - 1) / (x + 1)
    z_squared = z * z
    current_term = z
    result = 0.0
    
    for n in range(terms):
        result += current_term / (2 * n + 1)
        current_term *= z_squared
        
    return 2 * result

def clamp_and_convert(val, max_intensity=255):
    """Ensures pixel values stay between 0 and 255 and converts to integer."""
    if val > max_intensity:
        val = max_intensity
    elif val < 0:
        val = 0
    return int(val)

def negative_transform(matrix, max_intensity=255):
    """Negative: s = (L - 1) - r"""
    result = []
    for row in matrix:
        new_row = []
        for pixel in row:
            new_pixel = max_intensity - pixel
            new_row.append(new_pixel)
        result.append(new_row)
    return result


def power_law_transform(matrix, gamma, c=1.0, max_intensity=255):
    """Power Law (Gamma): s = c * r^gamma"""
    result = []
    for row in matrix:
        new_row = []
        for pixel in row:
            # Normalize to [0, 1] to prevent massive numbers
            r_norm = pixel / max_intensity
            
            # Apply transformation
            s_norm = c * (r_norm ** gamma)
            
            # Scale back to [0, max_intensity]
            new_pixel = clamp_and_convert(s_norm * max_intensity, max_intensity)
            new_row.append(new_pixel)
        result.append(new_row)
    return result


def log_transform(matrix, max_intensity=255):
    """Logarithmic: s = c * log(1 + r)"""
    # Calculate constant 'c' to map max input to max output
    c = max_intensity / manual_ln(1 + max_intensity)
    
    result = []
    for row in matrix:
        new_row = []
        for pixel in row:
            # Apply transformation using our manual natural log
            s = c * manual_ln(1 + pixel)
            
            new_pixel = clamp_and_convert(s, max_intensity)
            new_row.append(new_pixel)
        result.append(new_row)
    return result

def print_matrix(title, matrix):
    """Helper to print a 2D list neatly."""
    print(f"\n--- {title} ---")
    for row in matrix:
        # Align columns to 4 spaces for readability
        formatted_row = [str(pixel).rjust(4) for pixel in row]
        print(" ".join(formatted_row))


# Take user input for the matrix
print("Define your image matrix.")
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

print(f"\nEnter the pixel values (0-255) for each row, separated by spaces.")
print(f"Example for 3 columns: 10 150 255")

original_matrix = []
for i in range(rows):
    row_input = input(f"Row {i + 1}: ")
    # Split the string by spaces and convert each to an integer
    row_values = [int(val) for val in row_input.split()]
    original_matrix.append(row_values)

gamma_input = float(input("\nEnter a Gamma value for the Power Law (e.g., 0.5 to lighten, 2.0 to darken): "))

# Run transformations
neg_matrix = negative_transform(original_matrix)
pow_matrix = power_law_transform(original_matrix, gamma=gamma_input)
log_matrix = log_transform(original_matrix)

# Display results
print_matrix("Original Matrix", original_matrix)
print_matrix("Negative Transformation", neg_matrix)
print_matrix(f"Power Law (Gamma={gamma_input})", pow_matrix)
print_matrix("Log Transformation", log_matrix)
import cv2
import numpy as np

img = cv2.imread("Lab1/image.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Negative
neg_img = cv2.bitwise_not(img)

# 2. Power-Law (using OpenCV Lookup Table / cv2.LUT)
# Precalculating 0-255 values in a 1D table is faster than calculating exponents for every pixel
lut = np.array([((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_img = cv2.LUT(img, lut)

# 3. Log
# Cast the max value to a float to prevent the 8-bit integer overflow
max_pixel_val = float(np.max(img))
c = 255 / np.log(1 + max_pixel_val)

# Cast the image to float32 before applying log1p to ensure safe calculations across the whole array
log_img = np.array(c * np.log1p(img.astype(np.float32)), dtype=np.uint8)
import matplotlib.pyplot as plt

# Set up a figure with 4 subplots side-by-side
plt.figure(figsize=(20, 5))

# 1. Show Original
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off') # Hides the x and y axes

# 2. Show Negative
plt.subplot(1, 4, 2)
plt.imshow(neg_img, cmap='gray')
plt.title('Negative Transformation')
plt.axis('off')

# 3. Show Power-Law (Gamma)
plt.subplot(1, 4, 3)
plt.imshow(gamma_img, cmap='gray')
plt.title('Power-Law (Gamma)')
plt.axis('off')

# 4. Show Log
plt.subplot(1, 4, 4)
plt.imshow(log_img, cmap='gray')
plt.title('Log Transformation')
plt.axis('off')

# Render the plots
plt.show()