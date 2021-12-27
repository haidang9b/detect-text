# frozen_east_detection .pb cung cấp bởi open cv

# add các thư viện cần thiết
from random import randrange
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
import argparse

from Accuracy import accuracy_metric 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

def decode_predictions(scores, geometry):
    #lấy hàng và cột từ scores
    # khởi tạo các hình chữ nhật theo trị số của nó
    # confidences scores

    (numRows, numCols) = scores.shape[2:4]
    rects =[]
    confidences = []

    # chạy vòng lặp qua các rows
    for y in range(0, numRows):
        # trích xuất các scores
        # geometry được sử dụng để lấy hộp giới hạn tiềm năng
        # tọa độ bao quanh văn bản
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]

        anglesData = geometry[0, 4, y]

        #chạy vòng lặp qua các cột
        for x in range(0, numCols):
            # nếu scoresData không đủ tin cậy thì ignore
            if scoresData[x] < args["min_confidence"]:
                continue
            # tính các offset factor như các kết quả tương ứng
            # maps sẽ nhỏ hơn 4 lần vs ảnh đầu vào

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # xoay theo góc dự đoán

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # sử dụng các khối hình học để tính chiều cao và chiều rộng
            # của bounding box

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # tính toán điểm start và end của x, y
            # cho các vùng chưa text
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add các tọa độ và xác suất của chúng
            # vào danh sách lưu trữ

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # trả về kết quả là dạng tuple bao gồm các vị trí của text và xcas suất của chúng
    return (rects, confidences)




# load ảnh đầu vào với path
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# set width + height mới dựa trên input đầu vào
(newH, newW) = (args["height"], args["width"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize image theo newH và newW

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Định nghĩa 2 layer output cho EAST 
# Đầu tiên - xác suất đầu ra
# Thứ 2 - Lấy tọa độ của các vùng chứa văn bản

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])


# cTạo một màu đốm sau đó thực hiện chuyển tiếp
# Modal có tập hợp 2 lớp đầu ra
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode các dự đoán, sau đó triệu tiêu các khoảng không cực đại
# chặn các giới hạn chồng chéo, yêu
(rects, confidences) = decode_predictions(scores, geometry)

boxes = non_max_suppression(np.array(rects), probs=confidences)
# nhập ảnh đầu vào 
# initialize the list of results
results = []
output = orig.copy()
counter = 0
# The Bounding Rectangles will be stored here:

# chạy vòng for qua các vùng có chữ
for (startX, startY, endX, endY) in boxes:
    # Scale bounding box
    # xoay trục (ratios)
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Để detect bằng ocr tốt hơn thì cần bounding đầu vào
    # tính toán các hướng theo cả hướng x và y
    
    dX = int((endX - startX) * args["padding"])
    dY = int((endY - startY) * args["padding"])

    tmpX = startX
    tmpY = startY

    # Áp dụng padding cho các hướng
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # lấy chuẩn xác các padded ROI
    roi = orig[startY:endY, startX:endX]
    # Áp dụng Tessaract OCR cho văn bản thì cần phải cung cấp
    # (1) là ngôn ngữ (language) (2) là OEM flag of 4, để biết rằng
    # sử dụng mô hình LSTM cho OCR cuối cùng
    # (3) giá trị OEM, giả sử ở đây là 7, có thể thay đổi
    # coi ROI là một dòng văn bản
    config = ("-l eng --oem 1 --psm 7")
    
    text = pytesseract.image_to_string(roi,config=config)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    boxes_roi = pytesseract.image_to_boxes(roi)
    for b in boxes_roi.splitlines():
        # print(b)
        b = b.split(' ')
        print(b)
        if b[0]  != "~": 
            x_roi, y_roi, w_roi, h_roi = int(b[1]) , int(b[2]) , int(b[3]), int(b[4])
            # # Đóng các ô chử nhật theo từ ký tự
            cv2.rectangle(output, (x_roi + startX, y_roi + startY), ( startX + w_roi,  startY+ h_roi), (0,255,0),1)
    results.append(((startX, startY, endX, endY), text))

results = sorted(results, key=lambda r:r[0][1])
# chạy vòng lăp for trong mảng results
list_words_detected = []
for ((startX, startY, endX, endY), text) in results:
   
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    list_words_detected.append(text)
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)


    cv2.putText(output, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # show the output image


    
print(list_words_detected)
cv2.imshow("Text Detection", output)
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

print("accuracy_metric  : " + str(accuracy_metric(TP, TN, FP, FN) * 100) + "%")