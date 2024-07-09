import cv2
import numpy as np
import math
import time
import os
from concurrent.futures import ThreadPoolExecutor
#import threading

def process_image(filename):
    img_path = os.path.join('Test_images/Slight under focus', filename)
    background_path = os.path.join('Test_images/Slight under focus', 'background.tiff')
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

    single_start_time = time.time()

    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    blur_background = cv2.GaussianBlur(background, (3, 3), 0)
    substract = cv2.subtract(blur_background, blur_img)
    ret, binary = cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode1 = cv2.erode(binary, kernel)
    dilate1 = cv2.dilate(erode1, kernel)
    dilate2 = cv2.dilate(dilate1, kernel)
    erode2 = cv2.erode(dilate2, kernel)
    edge = cv2.Canny(erode2, 50, 150)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    deformation = None
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter
        deformation = 1 - circularity

    draw_image = np.zeros_like(img)
    contour_image = cv2.drawContours(draw_image, contours, -1, (255,255,255), 1)

    single_end_time = time.time()
    single_duration = single_end_time - single_start_time

    return filename, deformation, single_duration, contour_image

def main():
    start_time = time.time()
    
    image_files = [f for f in os.listdir('Test_images/Slight under focus') if f.endswith('.tiff') and f != 'background.tiff']

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, image_files))

    end_time = time.time()
    total_time = end_time - start_time
    total_duration = sum(duration for _, _, duration, _ in results)
    avg_time = total_duration / len(results) if results else 0

    print(f'Total images processed: {len(results)}')
    print(f'Total processing time: {total_time:.6f} seconds')
    print(f'Average processing time per image: {avg_time:.6f} seconds')

    # 顯示結果
    for filename, deformation, duration, contour_image in results:
        print(f'Showing results for {filename}')
        if deformation is not None:
            print(f'Deformability: {deformation:.6f}')
        print(f'Duration: {duration:.6f} seconds')
        
        cv2.imshow(f'Processed image - {filename}', contour_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()