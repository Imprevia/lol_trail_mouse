import cv2
import pyautogui
import numpy as np

template_path = 'icon.png'
def desktop_screenshots():
    img = pyautogui.screenshot().convert('RGB')
    frame = np.array(img)

    # 显示结果
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyWindow()


if __name__ == '__main__':
    desktop_screenshots()