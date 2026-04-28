from cropPuzzle import *
import os

directory = os.fsencode("media\puzzles")

for file in os.listdir(directory):
    directoryname = os.fsdecode(directory)
    filename = os.path.join(directoryname, os.fsdecode(file))
    in_omr = cv2.imread(filename, cv2.IMREAD_COLOR)
    page = find_page(in_omr)
    lines, page = get_lines(page)

    cv2.imshow("page", cv2.resize(page, (500,500)))
    cv2.resizeWindow("page", 500, 500)
    wait_q()
    
    intersections = get_line_intersections(lines, page)
    print(len(intersections))
    if len(intersections) == 100:
        get_squares(page, intersections, os.fsdecode(file).split(".")[0])