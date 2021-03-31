import cv2
import numpy as np
import os
import glob
import time
import sys

np.set_printoptions(threshold=sys.maxsize)

image1 = 'detected/stamp_detected_daihoccantho1.jpeg'
image2 = 'detected/stamp_detected_daihoccantho2.jpeg'


# def compare_images(image1, image2, output_folder):
def compare_images(image1, image2):
    img1 = cv2.imread(image1, 0)
    img2 = cv2.imread(image2, 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)

    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matches_ex = bf.knnMatch(des2, des1, k=2)
    good = []
    good_ex = []
    matched = False
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    for m, n in matches_ex:
        if m.distance < 0.75 * n.distance:
            good_ex.append([m])
    if len(good) > 95:
        matched = True
        print(len(good))
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    elif len(good_ex) > 95:
        matched = True
        print(len(good_ex))
        img3 = cv2.drawMatchesKnn(img2, kp2, img1, kp1, good_ex, None, flags=2)
    else:
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    timestr = time.time()
    matching_image_name = str(timestr) + '.png'
    # matching_image = str(output_folder)+'/'+matching_image_name
    #
    # cv2.imwrite(matching_image, img3)
    print(matched)
    return matched, matching_image_name


def identify_images(image_test, folder_path, output_path):
    kp_max = ''
    label_max = ''
    image_max = ''
    point_max = 0
    array_max = []
    for image in glob.glob(folder_path):
        img_test = cv2.imread(image_test, 0)
        img = cv2.imread(image, 0)

        sift = cv2.SIFT_create()
        kp_test, des_test = sift.detectAndCompute(img_test, None)

        kp, des = sift.detectAndCompute(img, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_test, des, k=2)
        point = []
        matched = False
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                point.append([m])
        if len(point) > point_max:
            point_max = len(point)
            array_max = point
            kp_max = kp
            image_max = image
    print(point_max)
    if point_max > 100:
        matched = True
    label_max = repr(image_max).split('\\')[2]
    matching_image_name = str(time.time()) + '.png'
    print(image_max)
    img_matching = cv2.drawMatchesKnn(img_test, kp_test, cv2.imread(image_max), kp_max, array_max, None, flags=2)
    cv2.imwrite(os.path.join(output_path, matching_image_name), img_matching)
    return matched, label_max, image_max, matching_image_name


if __name__ == '__main__':
    compare_images(image1, image2)
# # Test file's name
# testName = 'ImageQuery/giaothong2.png'
#
# # Import image train to a list then take class name
# path = 'ImageTrain'
# images = []
# classnames = []
# list_images = os.listdir(path)
# for cl in list_images:
#   imgCur = cv2.imread(f'{path}/{cl}',0)
#   images.append(imgCur)
#   classnames.append(os.path.splitext(cl)[0])
#
# # Take a descriptor per images train and store in a new list
# orb = cv2.ORB_create(nfeatures=1000)
# bf = cv2.BFMatcher()
# def findDes(images):
#   desList = []
#   for img in images:
#     kp, des = orb.detectAndCompute(img, None)
#     desList.append(des)
#   return desList
#
#
# def findID(img, desList):
#   kp2, des2 = orb.detectAndCompute(img, None)
#   matchList = []
#   finalVal = -1
#   try:
#     for des in desList:
#       matches = bf.knnMatch(des, des2, k=2)
#       good = []
#       for m, n in matches:
#         if (m.distance < 0.75 * n.distance):
#           good.append([m])
#       matchList.append(len(good))
#     print(matchList)
#   except:
#     pass
#   if len(matchList)!=0:
#     finalVal = matchList.index(max(matchList))
#   return finalVal
#
# # Test with new image
# desList = findDes(images)
# imgTestOrigin = cv2.imread(testName)
# imgTest = cv2.imread(testName, 0)
# id = findID(imgTest, desList)
# if id != -1:
#   print(classnames[id])
#   cv2.putText(imgTestOrigin, classnames[id],(15,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
# cv2.imwrite('prediction/'+testName, imgTestOrigin)
# cv2.imshow('imgTest', imgTestOrigin)
#
# kpTest, desTest = orb.detectAndCompute(imgTest, None)
# print(desTest.shape)
# Result = 'ImageTrain/'+str(classnames[id])+'.png'
# imgResult = cv2.imread(Result, 0)
# kpResult, desResult = orb.detectAndCompute(imgResult, None)
# matches = bf.knnMatch(desResult, desTest, k=2)
# goodResult = []
# for m, n in matches:
#   if (m.distance < 0.75 * n.distance):
#     goodResult.append([m])
# img3 = cv2.drawMatchesKnn(imgTest, kpTest, imgResult, kpResult, goodResult, None, flags=2 )
# cv2.imshow('img3', img3)
# cv2.waitKey(0)
