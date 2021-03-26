import cv2
import numpy as np
import os
import glob2
import time


image1 = 'detected/stamp_detected_daihoccantho1.jpeg'
image2 = 'detected/stamp_detected_daihoccantho2.jpeg'
def compare_images(image1, image2):
  img1 = cv2.imread(image1, 0)
  img2 = cv2.imread(image2, 0)

  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(img1, None)
  kp2, des2 = orb.detectAndCompute(img2, None)

  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2)
  good = []
  matched = False
  for m, n in matches:
    if m.distance < 0.75*n.distance:
      good.append([m])
  if len(good)>13:
    matched = True
  print(len(good))
  img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
  timestr = time.strftime("%Y%m%d-%H%M%S")
  matching_image = 'static/matching_img/'+str(timestr)+'.png'
  cv2.imwrite(matching_image, img3)
  print(matched)
  return matched, matching_image

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

