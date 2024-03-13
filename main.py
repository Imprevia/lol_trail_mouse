import cv2
import numpy as np
import dxcam
import pyautogui
from pynput import mouse
import threading

pyautogui.PAUSE = 0
CHAMPION_HIGHT_OFFSET, CHAMPION_WIDTH_OFFSET = 190,130
ENEMY_COLOR_OFFSET_X,ENEMY_COLOR_OFFSET_Y=5,15

camara = None
is_bug_on = False

def is_ememy_color(color):
    B,G,R = color
    if B > R or G > R :
        return False
    if G > 30 or B > 30:
        return False
    return R>50 and R< 70

def start():
    global is_bug_on
    if is_bug_on:
        return
    is_bug_on = True
    capture()

def capture():
    global camara
    global is_bug_on
    camara = dxcam.create(output_color="BGR")
    camara.start()

    while is_bug_on:
        if not is_bug_on:
            break

        frame = camara.get_latest_frame()

        lower_gray = np.array([60,60,60])
        upper_gray = np.array([100,100,100])

        mask =cv2.inRange(frame, lower_gray, upper_gray)

        contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mouse_pos = pyautogui.position()
        closest = None

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)

            if w/h>0.85 and w/h<1.05 and w>10:
                if y+ENEMY_COLOR_OFFSET_Y > frame.shape[0]-1 or x+ENEMY_COLOR_OFFSET_X > frame.shape[1]-1:
                    continue

                if not is_ememy_color(frame[y+ENEMY_COLOR_OFFSET_Y,x+ENEMY_COLOR_OFFSET_X]):
                    continue

                if closest is None:
                    closest = (x,y,w,h)
                else:
                    if abs(x-mouse_pos[0]) < abs(closest[0]-mouse_pos[0]):
                        closest = (x,y,w,h)

        if closest is None:
            continue

        pyautogui.moveTo(closest[0] + (closest[2] + CHAMPION_WIDTH_OFFSET)//2, closest[1] + closest[3] + CHAMPION_HIGHT_OFFSET)

    camara.stop()
    del camara

def on_click(x,y,button,pressed):
    global is_bug_on
    if pressed and button == mouse.Button.x1:
        if is_bug_on:
            is_bug_on = False
        else:
            print("Starting...")
            threading.Thread(target=start).start()

if __name__ == "__main__":
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()