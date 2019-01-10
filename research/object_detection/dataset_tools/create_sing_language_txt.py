import os
import json
import cv2
from copy import deepcopy


PATH_TO_FOLDER_TXT_TRAIN ="/media/dyve/BigData/HandDetection/Sequence_Dataset/newTrainVal/signLanguage_folder_train.txt"
PATH_TO_FOLDER_TXT_VAL = "/media/dyve/BigData/HandDetection/Sequence_Dataset/newTrainVal/signLanguage_folder_val.txt"
PATH_TO_SAVE_TXT_TRAIN = "/media/dyve/BigData/HandDetection/Sequence_Dataset/newTrainVal/signLanguage_train.txt"
PATH_TO_SAVE_TXT_VAL = "/media/dyve/BigData/HandDetection/Sequence_Dataset/newTrainVal/signLanguage_val.txt"

listOfTrainingFolders = []
with open(PATH_TO_FOLDER_TXT_TRAIN,"r") as file:
    content = file.readlines()
    for c in content:
        listOfTrainingFolders.append(c.split("\n")[0])

file.close()

listOfUnwantedLabels = ['HELP_right_hand','HELP_left_hand','NO_right_hand','NO_left_hand','YES_right_hand','YES_left_hand','none']
print(listOfTrainingFolders)
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


savedImagesPaths = []
savedJsonPaths = []
visitedNoneImagesPaths= []
visitedNoneJsonPaths = []
previousClassName = []
previousJson = []
previousImagesPath = []

finalListOfItems = []
for folder in listOfTrainingFolders:
    print(folder)
    listOfTrainImagesInEachFolder = os.listdir(folder + "/")
    listOfTrainImagesInEachFolder.sort()
    for item in listOfTrainImagesInEachFolder:
        if item.endswith(".jpg"):
            imagePath = folder + "/" + item
            image = cv2.imread(imagePath)
            json_path = folder + "/CombinedAnnotations/" + item.split(".")[0] + "_combined.json"
            if os.path.exists(json_path):
                # print(json_path)
                with open(json_path) as f:
                    data = json.load(f)
                    isBadImage = data["IsBadImage"]
                    if not isBadImage:
                        # print(isBadImage)

                        if not len(data["Annotations"])>0:
                            continue
                        class_name = data["Annotations"][0]["Label"]
                        class_name = class_name.replace(" ", "_")
                        if class_name ==  "face" or class_name == "n10_right_hand" or class_name == "n10_left_hand":
                            continue
                        else:
                            finalListOfItems.append((imagePath, json_path))
                        # if class_name in listOfUnwantedLabels:
                        #
                        #     previousClassName.append(class_name)
                        #     previousJson.append(json_path)
                        #     previousImagesPath.append(imagePath)
                        #
                        # else:
                        #     for i,x in reversed(list(enumerate(previousClassName))):
                        #         if x =="none" and i<8:
                        #             img = cv2.imread(previousImagesPath[i])
                        #             print(previousImagesPath[i])
                        #             print(previousClassName[i])
                        #             print(previousJson[i])
                        #             finalListOfItems.append((previousImagesPath[i],previousJson[i]))
                        #             # debugImage(img,previousJson[i])
                        #         else:
                        #             previousClassName.clear()
                        #             previousJson.clear()
                        #             previousImagesPath.clear()
                        #             break
                        #     previousClassName.clear()
                        #     previousJson.clear()
                        #     previousImagesPath.clear()
                        #     finalListOfItems.append((imagePath, json_path))






with open(PATH_TO_SAVE_TXT_TRAIN,"w") as f:
    for it in finalListOfItems:
        f.write(it[0] + " NNN " + it[1] + "\n")