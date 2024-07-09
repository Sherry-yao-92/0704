import cv2
import numpy as np
import math
import time
import os
from multiprocessing import Process, Queue, cpu_count

def read_images(input_queue, output_queue):
    while True:
        try:
            filename = input_queue.get(timeout=1)  # 從輸入隊列獲取檔案名,超時時間1秒
            if filename is None:  # 如果收到None,表示處理結束
                break
            img_path = os.path.join('Test_images/Slight under focus', filename)  
            background_path = os.path.join('Test_images/Slight under focus', 'background.tiff')  
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)  
            
            output_queue.put((filename, img, background))  # 將檔案名、圖片和背景放入輸出隊列
        except Queue.Empty:  # 如果隊列為空,繼續循環
            continue
    output_queue.put(None)  # 待處理完所有圖片結束

def preprocess_images(input_queue, output_queue):
    while True:
        try:
            data = input_queue.get(timeout=1)  # 從輸入隊列獲取數據,超時時間1秒
            if data is None:  # 如果收到None,表示處理結束
                break
            filename, img, background = data  # 假設 data 是一個包含三個元素的數據結構,並將這三個元素分別賦值給 filename, img, 和 background
            
            # 圖像預處理步驟
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
            
            output_queue.put((filename, edge))  # 將檔案名和處理後的邊緣圖像放入輸出隊列
        except Queue.Empty:  # 如果隊列為空,繼續循環
            continue
    output_queue.put(None)  # 待處理完所有圖片結束

def analyze_images(input_queue, output_queue):
    while True:
        try:
            data = input_queue.get(timeout=1)  # 從輸入隊列獲取數據,超時時間1秒
            if data is None:  # 如果收到None,表示處理結束
                break
            filename, edge = data  
            
            contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找輪廓
            
            deformation = None
            if contours:  # 如果找到輪廓
                cnt = contours[0]  # 取第一個輪廓
                area = cv2.contourArea(cnt)  
                perimeter = cv2.arcLength(cnt, True)  
                circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter  
                deformation = 1 - circularity  
            
            draw_image = np.zeros_like(edge)  
            contour_image = cv2.drawContours(draw_image, contours, -1, (255,255,255), 1)  
            
            output_queue.put((filename, deformation, contour_image))  # 將結果放入輸出隊列
        except Queue.Empty:  # 如果隊列為空,繼續循環
            continue
    output_queue.put(None)  # 待處理完所有圖片結束

def main():
    start_time = time.time()  # 記錄開始時間

    # 獲取所有.tiff文件,排除background.tiff
    image_files = [f for f in os.listdir('Test_images/Slight under focus') if f.endswith('.tiff') and f != 'background.tiff']

    input_queue = Queue()  # 創建輸入隊列
    read_queue = Queue()  # 創建讀取隊列
    preprocess_queue = Queue()  # 創建預處理隊列
    output_queue = Queue()  # 創建輸出隊列

    for filename in image_files:
        input_queue.put(filename)  # 將所有文件名放入輸入隊列

    # 創建並啟動三個處理進程
    read_process = Process(target=read_images, args=(input_queue, read_queue))
    preprocess_process = Process(target=preprocess_images, args=(read_queue, preprocess_queue))
    analyze_process = Process(target=analyze_images, args=(preprocess_queue, output_queue))

    processes = [read_process, preprocess_process, analyze_process]
    for p in processes:
        p.start()  # 啟動所有進程

    input_queue.put(None)  # 添加結束信號

   
    results = {}

    while len(results) < len(image_files):
        try:
            result = output_queue.get(timeout=10)
            if result is None:
                break
            filename, deformation, contour_image = result  
            results[filename] = (deformation, contour_image)
        except Queue.Empty:
            print("Timeout waiting for results. Breaking loop.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(image_files)

    print(f'Total images processed: {len(image_files)}')
    print(f'Total processing time: {total_time:.6f} seconds')
    print(f'Average processing time per image: {avg_time:.6f} seconds')

    for filename, (deformation, contour_image) in results.items():
        if deformation is not None:
            print(f'Deformability for {filename}: {deformation:.6f}')
        cv2.imshow(f'Processed image - {filename}', contour_image)
        cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for p in processes:
        p.terminate()
        p.join()

if __name__ == '__main__':
    main()