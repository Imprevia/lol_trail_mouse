import cv2
import numpy as np
import matplotlib.pyplot as plt

CHAMPION_HIGHT_OFFSET, CHAMPION_WIDTH_OFFSET = 190,130
ENEMY_COLOR_OFFSET_X,ENEMY_COLOR_OFFSET_Y=5,15

stags = cv2.imread('lol.png')

lower_gray = np.array([60, 60, 60])
upper_gray = np.array([100, 100, 100])

hsv = cv2.cvtColor(stags, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(stags, lower_gray, upper_gray)


def is_ememy_color(color):
    B,G,R = color
    if B > R or G > R :
        return False
    if G > 30 or B > 30:
        return False
    return R>50 and R< 70
def get_bboxes(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort according to the area of contours in descending order.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    bboxes = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        bboxes.append((x, y, x+w, y+h))
    return bboxes

def draw_annotations(img, bboxes, thickness=2, color=(0, 255, 0)):
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)

    return annotations

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


if __name__ == '__main__':
    bboxes = get_bboxes(mask)
    filtered_ann_stags = draw_annotations(stags, bboxes, thickness=5, color=(0, 0, 255))
    # Display.
    display(stags, mask,
            name_l='Stags original infrared',
            name_r='Thresholded Stags',
            figsize=(10, 7))