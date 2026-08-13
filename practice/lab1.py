import cv2
import numpy as np


img = cv2.imread("practice/babbon.png")

if img is None:
    print("Error: babbon.png not found!")
    exit()

while True:
    print("\n===== BABBON IMAGE PROCESSING MENU =====")
    print("1. Display Original Image")
    print("2. Convert to Grayscale")
    print("3. Increase Brightness")
    print("4. Decrease Brightness")
    print("5. Increase Contrast")
    print("6. Decrease Contrast")
    print("7. Negative Image")
    print("8. Blur Image")
    print("9. Sharpen Image")
    print("10. Edge Detection")
    print("11. Resize Image")
    print("12. Rotate Image")
    print("13. Histogram Equalization")
    print("14. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        cv2.imshow("Original Image", img)

    elif choice == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", gray)

    elif choice == 3:
        bright = cv2.convertScaleAbs(img, alpha=1, beta=50)
        cv2.imshow("Brightness Increased", bright)

    elif choice == 4:
        dark = cv2.convertScaleAbs(img, alpha=1, beta=-50)
        cv2.imshow("Brightness Decreased", dark)

    elif choice == 5:
        contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        cv2.imshow("Contrast Increased", contrast)

    elif choice == 6:
        contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
        cv2.imshow("Contrast Decreased", contrast)

    elif choice == 7:
        negative = 255 - img
        cv2.imshow("Negative Image", negative)

    elif choice == 8:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.imshow("Blurred Image", blur)

    elif choice == 9:
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
        sharp = cv2.filter2D(img, -1, kernel)
        cv2.imshow("Sharpened Image", sharp)

    elif choice == 10:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imshow("Edge Detection", edges)

    elif choice == 11:
        resized = cv2.resize(img, (300, 300))
        cv2.imshow("Resized Image", resized)

    elif choice == 12:
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("Rotated Image", rotated)

    elif choice == 13:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_eq = cv2.equalizeHist(gray)
        cv2.imshow("Histogram Equalization", hist_eq)

    elif choice == 14:
        print("Exiting...")
        break

    else:
        print("Invalid Choice!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()