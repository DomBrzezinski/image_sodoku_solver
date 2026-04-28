from cropPuzzle import *
import os
import csv

# in_omr = cv2.imread("media/sodokutest.jpeg", cv2.IMREAD_COLOR)
# page = find_page(in_omr, "media/sodokutest.jpeg" )
# lines, page = get_lines(page)
# intersections = get_line_intersections(lines, page)
# get_squares(page, intersections)

directory = os.fsencode("media\\tiles")

with open("media/data.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    for file in os.listdir(directory):
        filename = "media\\tiles\\" + os.fsdecode(file)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        check_image = normalize(image)
        # check_image = cv2.cvtColor(check_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(check_image, (5, 5), 0)
        T, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

        height, width = thresh.shape
        thresh = thresh[(int(height/4)):(int(3*height/4)), (int(width/4)):(int(3*width/4))]
        average_intensity = cv2.mean(thresh)[0]
        if average_intensity > 250:
            num = 0
        else:
            cv2.imshow("image", cv2.resize(image, (500,500)))
            cv2.resizeWindow("image", 500, 500)
            while True:
                num = cv2.waitKey(1)
                if num != -1: break
            num = num - 48
        writer.writerow([filename, num])
