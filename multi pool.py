import cv2
import numpy as np
import math
import time
import os
from multiprocessing import Pool, cpu_count

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

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, image_files)

    total_duration = 0
    count = 0

    for filename, deformation, duration, contour_image in results:
        if deformation is not None:
            print(f'Deformability for {filename}: {deformation:.6f}')
        
        cv2.imshow(f'Processed image - {filename}', contour_image)
        cv2.waitKey(1)  # 顯示圖片，但不等待按鍵

        total_duration += duration
        count += 1

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_duration / count if count > 0 else 0

    print(f'Total images processed: {count}')
    print(f'Total processing time: {total_time:.6f} seconds')
    print(f'Average processing time per image: {avg_time:.6f} seconds')
    #print(f'Total duration (sum of individual processing times): {total_duration:.6f} seconds')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    