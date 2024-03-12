import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# stags = cv2.imread('pig.jpg')
stags = cv2.imread('123.png')


def select_colorsp(img, colorsp='gray'):
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Split BGR.
    red, green, blue = cv2.split(img)
    # Convert to HSV.
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split HSV.
    hue, sat, val = cv2.split(im_hsv)
    # Store channels in a dict.
    channels = {'gray': gray, 'red': red, 'green': green,
                'blue': blue, 'hue': hue, 'sat': sat, 'val': val}

    return channels[colorsp]


def display(im_left, im_right, name_l='Left', name_r='Right', figsize=(10, 7)):
    # Flip channels for display if RGB as matplotlib requires RGB.
    im_l_dis = im_left[..., ::-1] if len(im_left.shape) > 2 else im_left
    im_r_dis = im_right[..., ::-1] if len(im_right.shape) > 2 else im_right

    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(im_l_dis)
    plt.title(name_l)
    plt.axis(False)
    plt.subplot(122)
    plt.imshow(im_r_dis)
    plt.title(name_r)
    plt.axis(False)
    plt.show()


def threshold(img, thresh=127, mode='inverse'):
    im = img.copy()

    if mode == 'direct':
        thresh_mode = cv2.THRESH_BINARY
    else:
        thresh_mode = cv2.THRESH_BINARY_INV

    ret, thresh = cv2.threshold(im, thresh, 255, thresh_mode)

    return thresh

def get_bboxes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort according to the area of contours in descending order.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    bboxes = []
    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        bboxes.append((x, y, x+w, y+h))
    return bboxes


def morph_op(img, mode='open', ksize=5, iterations=1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)

    return morphed
def draw_annotations(img, bboxes, thickness=2, color=(0, 255, 0)):
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)

    return annotations

def get_filtered_bboxes(img, min_area_ratio=0.001):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours according to area, larger to smaller.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    # Container to store filtered bboxes.
    bboxes = []
    # Image area.
    im_area = img.shape[0] * img.shape[1]
    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        # Remove very small detections.
        if cnt_area > min_area_ratio * im_area:
            bboxes.append((x, y, x+w, y+h))
    return bboxes

if __name__ == '__main__':
    # Select colorspace.
    blue = select_colorsp(stags, colorsp='blue')
    green = select_colorsp(stags, colorsp='green')
    red = select_colorsp(stags, colorsp='red')
    gray = select_colorsp(stags, colorsp='gray')
    hue = select_colorsp(stags, colorsp='hue')
    sat = select_colorsp(stags, colorsp='sat')
    val = select_colorsp(stags, colorsp='val')
    # Perform thresholding.
    thresh_stags = threshold(gray, thresh=70)
    morphed_stags = morph_op(thresh_stags)
    # bboxes = get_bboxes(morphed_stags)
    # ann_morphed_stags = draw_annotations(stags, bboxes, thickness=5, color=(0, 0, 255))
    bboxes = get_filtered_bboxes(thresh_stags, min_area_ratio=0.001)
    filtered_ann_stags = draw_annotations(stags, bboxes, thickness=2, color=(0, 0, 255))
    # Display.
    display(stags, filtered_ann_stags,
            name_l='Stags original infrared',
            name_r='Thresholded Stags',
            figsize=(20, 14))