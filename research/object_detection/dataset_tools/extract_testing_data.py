import os
import cv2
from shutil import copy
import json
from copy import deepcopy

PATH_TO_READ_TXT_VAL = "/media/dyve/BigData/HandDetection/Sequence_Dataset/signLanguage_WithoutNumbers_val.txt"
PATH_TO_SAVE_FOLDER = "/media/dyve/BigData/HandDetection/test_data_image_detection_API/NO_right_hand/"
def debugImage(image,json_path):


    with open(json_path) as file:
        data = json.load(file)

        xs = [float(data["Annotations"][0]['PointTopLeft'].split(',')[0]),
              float(data["Annotations"][0]['PointTopRight'].split(',')[0]),
              float(data["Annotations"][0]['PointBottomLeft'].split(',')[0]),
              float(data["Annotations"][0]['PointBottomRight'].split(',')[0])]
        ys = [float(data["Annotations"][0]['PointTopLeft'].split(',')[1]),
              float(data["Annotations"][0]['PointTopRight'].split(',')[1]),
              float(data["Annotations"][0]['PointBottomLeft'].split(',')[1]),
              float(data["Annotations"][0]['PointBottomRight'].split(',')[1])]

        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)

        class_name = data["Annotations"][0]["Label"]
        class_name = class_name.replace(" ", "_")

        image_to_draw = deepcopy(image)
        cv2.rectangle(image_to_draw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
        cv2.putText(image_to_draw, class_name,
                    (int(xmin) + 2, int(ymin) + 12),
                    0, 8 * 1.2e-4 * image_to_draw.shape[0],
                    (0, 255, 0), 2)
        cv2.imshow("image", image_to_draw)
        cv2.waitKey(0)
    f.close()

listOfFiles= []
with open(PATH_TO_READ_TXT_VAL,"r") as file:
    content = file.readlines()
    for c in content:
        listOfFiles.append((c.split(" NNN ")[0],c.split(" NNN ")[1].split("\n")[0]))
        # copyfile(src, dst)
print(listOfFiles[0][1])

for item in listOfFiles:
    with open(item[1]) as f:
        data = json.load(f)
        class_name = data["Annotations"][0]["Label"]
        class_name = class_name.replace(" ", "_")
        if class_name == "NO_right_hand":
            print(class_name)
            print(item[0])
            copy(item[0], PATH_TO_SAVE_FOLDER)


