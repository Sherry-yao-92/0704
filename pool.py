import cv2
import numpy as np
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def blur_image(img):
    return cv2.GaussianBlur(img, (3, 3), 0)

def subtract_and_threshold(blur_background, blur_img):
    substract = cv2.subtract(blur_background, blur_img)
    return cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY)[1]

def morphological_operations(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode1 = cv2.erode(binary, kernel)
    dilate1 = cv2.dilate(erode1, kernel)
    dilate2 = cv2.dilate(dilate1, kernel)
    return cv2.erode(dilate2, kernel)

def edge_detection(erode2):
    return cv2.Canny(erode2, 50, 150)

def find_contours(edge):
    return cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

def calculate_deformation(contours):
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter
        return 1 - circularity
    return None

def draw_contours(img_shape, contours):
    draw_image = np.zeros(img_shape, dtype=np.uint8)
    return cv2.drawContours(draw_image, contours, -1, (255,255,255), 1)

def main():
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        # 並行讀取圖片
        future_img = executor.submit(read_image, 'Test_images/Slight under focus/0066.tiff')
        future_background = executor.submit(read_image, 'Test_images/Slight under focus/background.tiff')

        img = future_img.result()
        background = future_background.result()

        # 並行模糊化
        future_blur_img = executor.submit(blur_image, img)
        future_blur_background = executor.submit(blur_image, background)

        blur_img = future_blur_img.result()
        blur_background = future_blur_background.result()

        # 並行二值化
        future_binary = executor.submit(subtract_and_threshold, blur_background, blur_img)

        # 並行膨脹侵蝕
        binary = future_binary.result()
        future_erode2 = executor.submit(morphological_operations, binary)

        # 並行邊緣檢測
        erode2 = future_erode2.result()
        future_edge = executor.submit(edge_detection, erode2)

        # 並行輪廓查找
        edge = future_edge.result()
        future_contours = executor.submit(find_contours, edge)

        # 並行計算變形度和繪製輪廓
        contours = future_contours.result()
        future_deformation = executor.submit(calculate_deformation, contours)
        future_contour_image = executor.submit(draw_contours, img.shape, contours)

        # 獲取最終結果
        deformation = future_deformation.result()
        contour_image = future_contour_image.result()

    if deformation is not None:
        print('Deformability: ', deformation)

    cv2.imshow('Processed image', contour_image)

    end_time = time.time()
    dif_time = end_time - start_time
    print('Duration: ', dif_time)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()