import cv2
import argparse
import glob


def detect_blur(image):
    threshold = 15 # 100 is default, last was 15
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
#    print(fm)
    if fm < threshold:
        return True
    else:
        return False

def background_people(image): #reject start frames indicating person size smaller threshold
    # do not implement
    # solved in features.start_frame since related
    print('person size smaller threshold')

