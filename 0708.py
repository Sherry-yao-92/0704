import cv2
import numpy as np
import math
import time
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# 設置為僅使用 CPU
USE_GPU = False

# 預加載和預處理背景圖像
background_path = os.path.join('Test_images/Slight under focus', 'background.tiff')
background_cpu = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
background_cpu = cv2.GaussianBlur(background_cpu, (3, 3), 0)

# 創建線程本地存儲
thread_local = threading.local()

def process_image_cpu(filename):
    # 讀取圖像
    img_path = os.path.join('Test_images/Slight under focus', filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 對圖像進行高斯模糊
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    # 從背景中減去模糊後的圖像
    subtract = cv2.subtract(background_cpu, blur_img)
    # 二值化處理
    _, binary = cv2.threshold(subtract, 10, 255, cv2.THRESH_BINARY)
    
    # 創建形態學操作的核
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 先膨脹後侵蝕
    morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # 先侵蝕後膨脹
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel)
    
    # 邊緣檢測
    edge = cv2.Canny(morphology, 50, 150)
    # 尋找輪廓
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    deformation = None
    if contours:
        # 取第一個輪廓
        cnt = contours[0]
        # 計算面積
        area = cv2.contourArea(cnt)
        # 計算周長
        perimeter = cv2.arcLength(cnt, True)
        # 計算圓形度
        circularity = float(2 * math.sqrt(math.pi * area)) / perimeter if perimeter > 0 else 0
        # 計算變形度
        deformation = 1 - circularity
    
    # 創建一個與原圖像大小相同的黑色圖像
    contour_image = np.zeros_like(img)
    # 在黑色圖像上繪製輪廓
    cv2.drawContours(contour_image, contours, -1, 255, 1)
    
    return filename, deformation, contour_image

def main():
    start_time = time.time()
    
    # 獲取所有圖像文件名
    image_files = [f for f in os.listdir('Test_images/Slight under focus') if f.endswith('.tiff') and f != 'background.tiff']
    
    process_func = process_image_cpu
    
    # 使用線程池並行處理圖像(代替Pool)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_func, image_files))
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(results) if results else 0

    # 印出處理結果統計
    print(f'Total images processed: {len(results)}')
    print(f'Total processing time: {total_time:.6f} seconds')
    print(f'Average processing time per image: {avg_time:.6f} seconds')

    # 顯示每張圖像的處理結果
    for filename, deformation, contour_image in results:
        print(f'Results for {filename}')
        if deformation is not None:
            print(f'Deformability: {deformation:.6f}')
        
        # 顯示處理後的圖像
        cv2.imshow(f'Processed image - {filename}', contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()