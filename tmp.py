import cv2
import numpy as np
import pytesseract
from Accuracy import accuracy_metric 

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
path = './bgt/2.jpg'
image = cv2.imread(path)

# config area max, min cần nhận diện
# chỗ này đang phải làm bằng tay

# # bgt/6
# MIN_AREA = 5000
# MAX_AREA = 20000

# # bgt/3
# MIN_AREA = 1000
# MAX_AREA = 3000

# bgt/2
MIN_AREA = 1000
MAX_AREA = 5000

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#  tạo ra structure element (giống như kernel) để dilation cho ảnh 
se = cv2.getStructuringElement(cv2.MORPH_RECT , (6,6))
im_morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)

out_gray=cv2.divide(gray, im_morph, scale=255)
im_norm = cv2.normalize(out_gray,  out_gray, 0, 255, cv2.NORM_MINMAX)
#  lấy ngưỡng dựa trên trung bình của ảnh cũ
out_binary=cv2.threshold(im_morph, np.mean(im_morph)+10, 255, cv2.THRESH_OTSU )[1] 

canny = cv2.Canny(gray, 100,200)
origH, origW = gray.shape[:2]
results = []
counter = 0
#   chạy contour để tìm ra đc các vùng nghi ngờ là có chữ
contours, hierarchy = cv2.findContours(out_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area_cnt = cv2.contourArea(cnt)
    x, y, h, w = cv2.boundingRect(cnt)
    #  so sánh xem vùng đó có nằm trong khoảng min max đang xét hay khong, nếu có lưu lại x, y, w, h
    if area_cnt > MIN_AREA and area_cnt <= MAX_AREA:
        # lấy tọa độ startX, startY, endX, endY xem nó có vượt quá giới hạn của ảnh hay không


        # cv2.rectangle(image, (x, y), (x + h, y + w), (0, 255, 0), 2)
        #  config pytesserect
        config = ("-l eng --oem 1 --psm 7") 
        #  cắt vùng cần lấy ra 
        roi = image[ y: y+w ,x:x + h]

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(roi,config=config)
        # print(text)
        
        results.append(((x, y, h, w), text))
        

#  mảng lưu lại danh sách các từ detect đc
list_words_detected = []

#  show + vẽ kết quả sau khi detect đc các string
for ((x, y, h, w), text) in results:
    
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    
    if text != "" and len(text) != 0:
        list_words_detected.append(text)
        cv2.rectangle(image, (x, y), (x + h, y + w), (0, 0, 255), 2)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 222, 0), 2)


print(list_words_detected)


for ((x, y, h, w), text) in results:
    roi = image[ y: y+w ,x:x + h]
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    thresh =   cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    boxes = pytesseract.image_to_boxes(roi)
    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        print(b)
        # lấy ra tọa độ theo x và y, chiều rộng chiều dài
        # b[0] là ký tự
        # b[1] b[2] là x và y
        # b[3] b[4] la chiều rộng và chiều dài
        if b[0]  != "~": 
            x_roi, y_roi, w_roi, h_roi = int(b[1]) , int(b[2]) , int(b[3]), int(b[4])
            # # Đóng các ô chử nhật theo từ ký tự
            cv2.rectangle(image, (x_roi + x, y_roi + y), ( x + w_roi,  y+ h_roi), (0,0,0),1)
            # ghi thêm text vào trong ký tự đã nhân diện
            # cv2.putText(img,b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255),1 )

cv2.imshow('result', image)
cv2.waitKey(0)

#  Giả sử ma các từ đúng của ảnh bgt/2.jpg
guess_text = ["DUONG", "DUONG", "NGO", "MINH"]

# TP: đúng thật
# TN : sai thật
# FP : đúng giả sử
# FN : sai giả sử

#  giả sử sai 1 từ
FN = 1
#  giả sử đúng 1 từ
FP = 1

TP = 0
for word in list_words_detected:
    if word in guess_text:
        TP += 1

TN = len(guess_text) - TP

print("accuracy_metric of "+path + " : " + str(accuracy_metric(TP, TN, FP, FN) * 100) + "%")