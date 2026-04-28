from mnist import load, predict
import cv2
import csv
import os
 
model = load('my_model.keras')
directory = os.fsencode("media\\tiles")

for file in os.listdir(directory):
    filename = "media\\tiles\\" + os.fsdecode(file)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    image = image[(int(height/8)):(int(7*height/8)), (int(width/8)):(int(7*width/8))]
    predict(image, model)