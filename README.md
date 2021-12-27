# detect text with Pytesseract

## Method 1

run cmd : `python east.py --image bgt/7.png --east frozen_east_text_detection.pb`
- `bgt/7.png` is path image
- line 201 -> end is Accuracy
## Method 2

run cmd: `python tmp.py`
- edit line 7 with path `path = './bgt/2.jpg'`
- add `MIN_AREA` & `MAX_AREA` with min, max area text object
- line 105 -> end is Accuracy
