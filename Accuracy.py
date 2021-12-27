import cv2
import numpy as np


# TP: đúng thật
# TN : sai thật
# FP : đúng giả sử
# FN : sai giả sử
def accuracy_metric(TP, TN, FP, FN):
    return (TP + TN)/(TP + TN + FP + FN)