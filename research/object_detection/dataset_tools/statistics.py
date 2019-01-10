import os
import json
import cv2
import numpy as np
from copy import deepcopy
framesPerLabel = {

    'n0_right_hand' :[],
    'n0_left_hand':[],
    'n1_right_hand':[],
    'n1_left_hand':[],
    'n2_right_hand':[],
    'n2_left_hand':[],
    'n3_right_hand':[],
    'n3_left_hand':[],
    'n4_right_hand':[],
    'n4_left_hand':[],
    'n5_right_hand':[],
    'n5_left_hand':[],
    'n6_right_hand':[],
    'n6_left_hand':[],
    'n7_right_hand':[],
    'n7_left_hand':[],
    'n8_right_hand':[],
    'n8_left_hand':[],
    'n9_right_hand':[],
    'n9_left_hand':[],
    'n10_right_hand':[],
    'n10_left_hand':[],
    'HELP_right_hand':[],
    'HELP_left_hand':[],
    'NO_right_hand':[],
    'NO_left_hand':[],
    'YES_right_hand':[],
    'YES_left_hand':[],
    'none':[]
}

def get_name_from_folder_name (folder):
    if folder == "Daniel Basic ASL Signs Olive Shirt-SEQUENCE": return "DANIEL"
    if folder == "14-0 hearing caucasian male mostly clear signs_SEQUENCE": return "Rick&Morty"
    if folder == "Random 0-9 video1_SEQUENCE": return "DANIEL"
    if folder == "Basic ASL Signs Randomized 5_SEQUENCE": return "DANIEL"
    if folder == "02-0 hispanic male 30s deaf slow clear and good signs with synced voiceover-SEQUENCE": return "Hispanic Male"
    if folder == "13-1 indian female hearing clear signs_SEQUENCE": return "Indian Female"
    if folder == "Random 0-9 video3_SEQUENCE": return "DANIEL"
    if folder == "Daniel Basic ASL signs sleeveless blue-SEQUENCE": return "DANIEL"
    if folder == "Basic ASL Signs Randomized 1_SEQUENCE": return "DANIEL"
    if folder == "Daniel Basic ASL Signs Hoodie-SEQUENCE": return "DANIEL"
    if folder == "18-0 caucasian female middle age deaf clear signs_SEQUENCE": return "Caucasian Female"
    if folder == "Daniel Basic ASL Plaid Shirt_SEQUENCE": return "DANIEL"
    if folder == "Daniel Basic ASL Signs Hat-SEQUENCE": return "DANIEL"
    if folder == "19-0 caucasian female hearing clear signs_SEQUENCE": return "PINK GIRL"
    if folder == "Random 0-9 video2_SEQUENCE": return "DANIEL"
    if folder == "Daniel Basic ASL Signs Beanie-SEQUENCE": return "DANIEL"
    if folder == "Daniel Basic ASL Signs White Shirt-SEQUENCE": return "DANIEL"
    if folder == "High Quality Basic ASL Signs with audio explanation-SEQUENCE": return "DANIEL"
    if folder == "19-1 caucasian female hearing clear signs_SEQUENCE": return "PINK GIRL"
    if folder == "Daniel Basic ASL Black Shirt_SEQUENCE": return "DANIEL"
    if folder == "Daniel Basic ASL Red Shirt-SEQUENCE": return "DANIEL"
    if folder == "18-1 caucasian female middle age deaf clear signs with synced voiceover_SEQUENCE": return "Caucasian Female"


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
    file.close()

print(framesPerLabel)


def only_values_statistics():
    PATH_TO_FOLDER_TXT_TRAIN ="/media/dyve/BigData/HandDetection/Sequence_Dataset/signLanguage_folder_train.txt"
    PATH_TO_FOLDER_TXT_VAL = "/media/dyve/BigData/HandDetection/Sequence_Dataset/signLanguage_folder_val.txt"

    listOfTrainingFolders = []
    with open(PATH_TO_FOLDER_TXT_TRAIN,"r") as file:
        content = file.readlines()
        for c in content:
            listOfTrainingFolders.append(c.split("\n")[0])

    file.close()
    print(listOfTrainingFolders)

    previousClassName= ""
    index = 0
    listOfPeople = []
    for folder in listOfTrainingFolders:
        print(folder.split("/")[-1])

        character = get_name_from_folder_name(folder.split("/")[-1])
        print(character)

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
                            print(class_name)
                            if class_name =='face':
                                continue
                            if class_name != previousClassName:

                                if index > 3:
                                    framesPerLabel[previousClassName].append(index)
                                    index = 0
                                else:
                                    if index ==0 :
                                        print(folder)
                                        print(item)
                                        print(json_path)
                                        debugImage(image,json_path)
                                        index = 0
                                previousClassName = class_name

                            else:
                                index = index+1
    print(framesPerLabel)
    with open('/media/dyve/BigData/HandDetection/Sequence_Dataset/Statistics/firstStatisticsTrainFolders.txt','w') as file:
        for key in framesPerLabel:
            minValue = min(framesPerLabel[key])
            maxValue = max(framesPerLabel[key])
            meanValue = np.mean(framesPerLabel[key])
            file.write("Min " + key + " = "+ str(minValue) + "\n")
            file.write("Max " + key + " = " + str(maxValue)+ "\n")
            file.write("Mean " + key + " = " + str(meanValue)+ "\n")
            file.write("#######################################################"+ "\n")

            print("Min " + key + " = "+ str(minValue))
            print("Max " + key + " = " + str(maxValue))
            print("Mean " + key + " = " + str(meanValue))
            print("#######################################################")

    file.close()




def values_for_every_person():
    PATH_TO_FOLDER_TXT_TRAIN = "/media/dyve/BigData/HandDetection/Sequence_Dataset/signLanguage_folder_train.txt"
    PATH_TO_FOLDER_TXT_VAL = "/media/dyve/BigData/HandDetection/Sequence_Dataset/signLanguage_folder_val.txt"

    listOfTrainingFolders = []
    with open(PATH_TO_FOLDER_TXT_VAL, "r") as file:
        content = file.readlines()
        for c in content:
            listOfTrainingFolders.append(c.split("\n")[0])

    file.close()
    print(listOfTrainingFolders)

    previousClassName = ""
    index = 0
    listOfPeople = {
        "DANIEL": deepcopy(framesPerLabel),
        "Rick&Morty":deepcopy(framesPerLabel),
        "Hispanic Male":deepcopy(framesPerLabel),
        "Indian Female":deepcopy(framesPerLabel),
        "Caucasian Female":deepcopy(framesPerLabel),
        "PINK GIRL":deepcopy(framesPerLabel)

    }
    print(listOfPeople)
    for folder in listOfTrainingFolders:
        print(folder.split("/")[-1])

        character = get_name_from_folder_name(folder.split("/")[-1])
        print(character)

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

                            if not len(data["Annotations"]) > 0:
                                continue
                            class_name = data["Annotations"][0]["Label"]
                            class_name = class_name.replace(" ", "_")
                            # print(class_name)
                            if class_name == 'face':
                                continue
                            if class_name != previousClassName:

                                if index > 3:
                                    print(listOfPeople[character][previousClassName])
                                    listOfPeople[character][previousClassName].append(index)
                                    index = 0
                                # else:
                                    # if index == 0:
                                        # print(folder)
                                        # print(item)
                                        # print(json_path)
                                        # debugImage(image, json_path)
                                        # index = 0
                                previousClassName = class_name

                            else:
                                index = index + 1
        print(framesPerLabel)
        meanValuesWithoutNone = []
        meanValuesWithNone = []
        meanValuesLeftHand= []
        meanValuesRightHand= []
        with open('/media/dyve/BigData/HandDetection/Sequence_Dataset/Statistics/'+"SequenceTimeFor_"+character+"_Val.txt", 'w') as file:
            file.write(character + "\n\n")
            for key in framesPerLabel:
                if  len(listOfPeople[character][key])>0:
                    minValue = min(listOfPeople[character][key])
                    maxValue = max(listOfPeople[character][key])
                    meanValue = np.mean(listOfPeople[character][key])
                    meanValuesWithNone.append(meanValue)
                    if key != "none":
                        meanValuesWithoutNone.append(meanValue)
                        if key.split('_')[1] == 'left':
                            meanValuesLeftHand.append(meanValue)

                        if key.split('_')[1] =='right':
                            meanValuesRightHand.append(meanValue)


                    file.write("Min " + key + " = " + str(minValue) + "\n")
                    file.write("Max " + key + " = " + str(maxValue) + "\n")
                    file.write("Mean " + key + " = " + str(meanValue) + "\n")
                    file.write("#######################################################" + "\n")

                    print("Min " + key + " = " + str(minValue))
                    print("Max " + key + " = " + str(maxValue))
                    print("Mean " + key + " = " + str(meanValue))
                    print("#######################################################")
            totalMeanWithNone = np.mean(meanValuesWithNone)
            totalMeanWithoutNone = np.mean(meanValuesWithoutNone)
            totalMeanLeftHand = np.mean(meanValuesLeftHand)
            totalMeanRightHand = np.mean(meanValuesRightHand)
            file.write("Mean of all sequences with none=" +str(totalMeanWithNone) + "\n")
            file.write("Mean of all sequences without none=" + str(totalMeanWithoutNone) + "\n")
            file.write("Mean of all sequences LEFT HAND=" + str(totalMeanLeftHand) + "\n")
            file.write("Mean of all sequences RIGHT HAND=" + str(totalMeanRightHand) + "\n")
        file.close()


if __name__== "__main__":
    values_for_every_person()