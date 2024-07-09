import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
import math

directory = 'Test_images/Slight under focus'
background_path = os.path.join(directory, 'background.tiff')
files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
blur_background = cv2.GaussianBlur(background, (3, 3), 0)

def process_image(filename):
    img_path = os.path.join(directory, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    subtract = cv2.subtract(blur_background, blur_img)
    _, binary = cv2.threshold(subtract, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    erode1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    erode2 = cv2.morphologyEx(erode1, cv2.MORPH_OPEN, kernel)
    
    edges = cv2.Canny(erode2, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(img)
    cv2.drawContours(contour_image, contours, -1, 255, 1)

    deformation = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter 
        deformation = 1 - circularity

    return filename, contour_image, deformation


def main():
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_image, files))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(files):.6f} seconds")

    for filename, contour_image, deformation in results:
        cv2.imshow(f'Processed Image: {filename}', contour_image)
        if deformation is not None:
            print(f"Deformability for {filename}: {deformation:.6f}")
        print(f"processed image {filename}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()