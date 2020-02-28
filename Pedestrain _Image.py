from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


path = "ped4.jpeg"

# loop over the image paths
image = cv2.imread(path)
orig = image
# load the image and resize it to (1) reduce detection time
# and (2) improve detection accuracy

# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
   		padding=(8, 8), scale=1.05)

# draw the original bounding boxes
for (x, y, w, h) in rects:
   	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
   	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# show some information on the number of bounding boxes
   	
# show the output images
#cv2.imshow("Before NMS", orig)
cv2.imshow("After NMS", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
