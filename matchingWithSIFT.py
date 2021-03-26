import cv2
import numpy as np

img1 = cv2.imread('detected/stamp_detected_dothicamranh1.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('detected/stamp_detected_dothicamranh3.jpeg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
orb = cv2.ORB_create()
# surf = cv2.xfeatures2D.SURF_create()

kp1, des1 = sift.detectAndCompute(img1, None)
# kp12, des12 = surf.detectAndCompute(img12, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
img = cv2.drawKeypoints(img1, kp1, None)
img2 = cv2.drawKeypoints(img2, kp2, None)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# img12 = cv2.drawKeypoints(img12, kp12, None)
cv2.imshow("image", img)
cv2.imshow("image2", img2)
cv2.imshow('matching_image', img3)
cv2.waitKey(0)