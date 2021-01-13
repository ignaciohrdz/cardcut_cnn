import cv2
import os
import numpy as np
import pandas as pd
import glob
import time

# Some parameters for putText
font = cv2.FONT_HERSHEY_SIMPLEX
org = (25, 50)
fontScale = 1
color = (0, 255, 0)
thickness = 2

# Load the frames and label them
folder = "data/images"

# Global coordinates (for mouse events)
mouse_x = 0
mouse_y = 0

def save_mouse_coords(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Saving mouse coords: {},{}'.format(x,y))
        mouse_y = x
        mouse_x = y

image_files = pd.read_csv("data/cardpoints.csv", delimiter=",")
image_files = image_files.fillna(0)
columns = list(image_files.columns.values)
visible_list = [k for k in columns if '_visible' in k]

k = None

for index, row in image_files.iterrows():
    time_start = time.process_time()
    if row[1] == 0:
        path = os.path.join(folder,row[0])
        img = cv2.imread(path,1)
        window = 'Frame:'+row[0]
        cv2.namedWindow(window)
        cv2.setMouseCallback(window,save_mouse_coords)

        for point_name in visible_list:
            mouse_x = 0
            mouse_y = 0
            title = "({})".format(row[2]) +point_name.replace('_visible',':')
            img_now = cv2.putText(img.copy(), title, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
            cv2.imshow(window,img_now)
            k = cv2.waitKey()

            if k == ord('q'):
                break

            if mouse_x > 0 or mouse_y > 0:
                image_files.iloc[index,1] = 1
                image_files.loc[index,point_name] = 1
                image_files.loc[index,point_name.replace('_visible','_x')] = round(mouse_x/img.shape[0],6)
                image_files.loc[index,point_name.replace('_visible','_y')] = round(mouse_y/img.shape[1],6)
                print('Saving coordinates for {}'.format(point_name))
            img_now = img.copy()

        cv2.destroyAllWindows()

        if k == ord('q'):
            break
    
    time_elapsed = round(time.process_time() - time_start,6)
    print('Time elapsed: {} seconds.'.format(time_elapsed))

images_files = image_files.fillna(0)
image_files.to_csv("data/cardpoints.csv", index=False)    
