import cv2
import os
import numpy as np
import pandas as pd
import glob


def draw_vertices(img_vertices, vertices, colour=(255, 255, 255)):
    for v in vertices:
        if row[v[0]] == 1:
            x0_vert = int(image_files.loc[index, v[0].replace('_visible', '_x')] * img.shape[0])
            y0_vert = int(image_files.loc[index, v[0].replace('_visible', '_y')] * img.shape[1])

            if row[v[1]] == 1:
                x1_vert = int(image_files.loc[index, v[1].replace('_visible', '_x')] * img.shape[0])
                y1_vert = int(image_files.loc[index, v[1].replace('_visible', '_y')] * img.shape[1])
                img_vertices = cv2.line(img_vertices, (y0_vert, x0_vert), (y1_vert, x1_vert), colour, 1)

            if row[v[2]] == 1:
                x2_vert = int(image_files.loc[index, v[2].replace('_visible', '_x')] * img.shape[0])
                y2_vert = int(image_files.loc[index, v[2].replace('_visible', '_y')] * img.shape[1])
                img_vertices = cv2.line(img_vertices, (y0_vert, x0_vert), (y2_vert, x2_vert), colour, 1)

            if row[v[3]] == 1:
                x3_vert = int(image_files.loc[index, v[3].replace('_visible', '_x')] * img.shape[0])
                y3_vert = int(image_files.loc[index, v[3].replace('_visible', '_y')] * img.shape[1])
                img_vertices = cv2.line(img_vertices, (y0_vert, x0_vert), (y3_vert, x3_vert), colour, 1)

    return img_vertices


# Some parameters for putText
font = cv2.FONT_HERSHEY_SIMPLEX
org = (25, 50)
fontScale = 1
color = (0, 255, 0)
thickness = 2

# Show the labelled frames
folder = "data/images"

image_files = pd.read_csv("data/cardpoints.csv", delimiter=",")
image_files = image_files.fillna(0)
columns = list(image_files.columns.values)
visible_list = [k for k in columns if '_visible' in k]

points_colours = {'upper_left_face_visible': (0, 0, 255),
                  'upper_right_face_visible': (0, 255, 0),
                  'upper_left_back_visible': (255, 0, 0),
                  'upper_right_back_visible': (0, 255, 255),
                  'lower_left_face_visible': (255, 0, 255),
                  'lower_right_face_visible': (255, 255, 0),
                  'lower_left_back_visible': (50, 128, 255),
                  'lower_right_back_visible': (255, 128, 0)}

vertices = [
    ['upper_left_face_visible', 'upper_left_back_visible', 'upper_right_face_visible', 'lower_left_face_visible'],
    ['upper_left_back_visible', 'upper_left_face_visible', 'upper_right_back_visible', 'lower_left_back_visible'],
    ['upper_right_face_visible', 'upper_right_back_visible', 'upper_left_face_visible', 'lower_right_face_visible'],
    ['upper_right_back_visible', 'upper_right_face_visible', 'upper_left_back_visible', 'lower_right_back_visible'],
    ['lower_left_face_visible', 'lower_left_back_visible', 'lower_right_face_visible', 'upper_left_face_visible'],
    ['lower_left_back_visible', 'lower_left_face_visible', 'lower_right_back_visible', 'upper_left_back_visible'],
    ['lower_right_face_visible', 'lower_right_back_visible', 'lower_left_face_visible', 'upper_right_face_visible'],
    ['lower_right_back_visible', 'lower_right_face_visible', 'lower_left_back_visible', 'upper_right_back_visible']]

k = None

i = 0
prev_value = 52

for index, row in image_files.iterrows():
    path = os.path.join(folder, row[0])
    img = cv2.imread(path, 1)
    window = 'Frame'  # 'Frame:'+row[0]
    window2 = 'Frame scaffold'  # 'Frame scaffold:'+row[0]
    cv2.namedWindow(window)
    cv2.namedWindow(window2)

    img_draw = img.copy()

    img_vertices = np.zeros_like(img_draw)
    img_vertices = draw_vertices(img_vertices, vertices)
    img_draw = draw_vertices(img_draw, vertices, colour=(0, 0, 0))

    for point_name in visible_list:
        x = int(image_files.loc[index, point_name.replace('_visible', '_x')] * img.shape[0])
        y = int(image_files.loc[index, point_name.replace('_visible', '_y')] * img.shape[1])

        img_draw = cv2.circle(img_draw, (y, x), 3, points_colours[point_name], thickness=-1)
        img_vertices = cv2.circle(img_vertices, (y, x), 3, points_colours[point_name], thickness=-1)

    if row[2] == prev_value:
        i += 1
    else:
        prev_value = row[2]
        i = 1

    img_draw = cv2.putText(img_draw, str(row[2]) + " - " + str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)
    img_vertices = cv2.putText(img_vertices, str(row[2]) + " - " + str(i), org, font, fontScale, color, thickness,
                               cv2.LINE_AA)

    cv2.imshow(window, img_draw)
    cv2.imshow(window2, img_vertices)
    k = cv2.waitKey()
    # cv2.destroyAllWindows()

    if k == ord('q'):
        break
