import cv2
import os
import numpy as np
import pandas as pd
import glob

# Load the videos and subsample the frames

sampling = 20

video_files = pd.read_csv("data/videos/videofiles.csv", delimiter=";")
image_files = pd.read_csv("data/cardpoints.csv", delimiter=";")

for index, row in video_files.iterrows():
    videoname = row[0]
    counted = row[1]
    total = row[2]

    video = cv2.VideoCapture(os.path.join("data/videos", videoname))
    if not video.isOpened():
        print("The video can't be opened.")

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if i % sampling == 0:
                cv2.imshow('Frame from: ' + videoname, frame)
                cv2.waitKey(10)
                frame_name = videoname.replace(".MP4", "_") + "frame_" + str(i) + ".jpg"
                new_data = {'imageName': frame_name, 'cards_counted': counted, 'total_cards': total}
                cv2.imwrite(os.path.join("data/images", frame_name), frame)
                image_files = image_files.append(new_data, ignore_index=True)
            i += 1
        else:
            break

    cv2.destroyAllWindows()
    video.release()

images_files = image_files.fillna(0)
image_files.to_csv("data/cardpoints.csv", index=False)
