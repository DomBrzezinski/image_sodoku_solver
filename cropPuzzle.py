import cv2
import numpy as np
import copy

DEFAULT_WHITE_COLOR = 255
DEFAULT_BLACK_COLOR = 0
DEFAULT_NORMALIZE_PARAMS = {"alpha": 0, "beta": 255}
DEFAULT_LINE_WIDTH = 2
DEFAULT_MARKER_LINE_WIDTH = 4
DEFAULT_CONTOUR_COLOR = (0, 255, 0)
DEFAULT_CONTOUR_LINE_WIDTH = 2
DEFAULT_CONTOUR_FILL_COLOR = (255, 255, 255)
DEFAULT_CONTOUR_FILL_WIDTH = 10
DEFAULT_BORDER_REMOVE = 5

DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER = {"kernel_size": (5, 5), "sigma_x": 0}

# CropPage constants
MIN_PAGE_AREA_THRESHOLD = 80000
MAX_COSINE_THRESHOLD = 0.35
DEFAULT_GAUSSIAN_BLUR_KERNEL = (3, 3)
PAGE_THRESHOLD_PARAMS = {"threshold_value": 200, "max_pixel_value": 255}
CANNY_PARAMS = {
    # lower_threshold: lower bound for Canny edge detection
    # upper_threshold: upper bound for Canny edge detection
    "lower_threshold": 185,
    "upper_threshold": 55,
}
APPROX_POLY_EPSILON_FACTOR = 0.1

# CropOnMarkers constants
QUADRANT_DIVISION = {"height_factor": 3, "width_factor": 2}
MARKER_RECTANGLE_COLOR = (150, 150, 150)
ERODE_RECT_COLOR = (50, 50, 50)
NORMAL_RECT_COLOR = (155, 155, 155)
EROSION_PARAMS = {"kernel_size": (5, 5), "iterations": 5}

# FeatureBasedAlignment constants
DEFAULT_MAX_FEATURES = 500
DEFAULT_GOOD_MATCH_PERCENT = 0.15

# Builtin processor constants
DEFAULT_MEDIAN_BLUR_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_BLUR_PARAMS = {"kernel_size": (3, 3), "sigma_x": 0}


options = {
        "morphKernel": [
          10,
          10
        ]
      }

def normalize(image):
    return cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)

def grab_contours(cnts):
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    return cnts

def wait_q():
    esc_key = 27
    while cv2.waitKey(1) & 0xFF not in [ord("q"), esc_key]:
        pass
    cv2.destroyAllWindows()


def angle(p_1, p_2, p_0):
    dx1 = float(p_1[0] - p_0[0])
    dy1 = float(p_1[1] - p_0[1])
    dx2 = float(p_2[0] - p_0[0])
    dy2 = float(p_2[1] - p_0[1])
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt(
        (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
    )


def check_max_cosine(approx):
    # assumes 4 pts present
    max_cosine = 0
    min_cosine = 1.5
    for i in range(2, 5):
        cosine = abs(angle(approx[i % 4], approx[i - 2], approx[i - 1]))
        max_cosine = max(cosine, max_cosine)
        min_cosine = min(cosine, min_cosine)

    if max_cosine >= MAX_COSINE_THRESHOLD:
        return False
    return True


def find_page(image, file_path):

    image = normalize(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("thresh", cv2.resize(thresh, (500, 800)))
    cv2.resizeWindow("thresh", 500, 800)
    wait_q()

    comment_box = None
    boxes = []
    for cnt in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # The box should have 4 corners and a significant area
        
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > 10000:  # Adjust this threshold based on your image resolution
                boxes.append(approx)

    # 5. Draw the result
    if boxes is not None:
        cv2.drawContours(image, boxes, -1, (0, 255, 0), 3)

    cv2.imshow("image", cv2.resize(image, (500, 800)))
    cv2.resizeWindow("image", 500, 800)
    wait_q()

    pts = np.reshape(boxes[0], (4, -1))

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    max_width = max(int(width_a), int(width_b))
    # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # max_height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    cv2.imshow("warped", cv2.resize(warped, (500, 500)))
    cv2.resizeWindow("warped", 500, 500)
    wait_q()

    return warped

def get_lines(image):
    length = min(image.shape[0:1])
    image = cv2.resize(image, (length, length))
    canny = cv2.GaussianBlur(image, (9, 9), 0)
    canny = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 5, 2)
    cv2.imshow("canny", cv2.resize(canny, (500, 500)))
    cv2.resizeWindow("canny", 500, 500)
    wait_q()

    lines = cv2.HoughLinesP(canny , 1, np.pi/180, threshold=250, minLineLength=300, maxLineGap=10)
    colour_image = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(colour_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("colour_image", cv2.resize(colour_image, (500, 500)))
    cv2.resizeWindow("colour_image", 500, 500)
    wait_q()

    return lines, image



def get_squares(image, intersections):
    length = image.shape[0]
    correct_rows = [[],[],[],[],[],[],[],[],[],[]]
    for intersection in intersections:
        row = round((intersection[1]/length)*9)
        correct_rows[row].append(intersection)
    all_rows = np.array([])
    all_rows = np.zeros((10,10,2))
    for i in range(len(correct_rows)):
        row = np.array(correct_rows[i])
        row = row[row[:, 0].argsort()]
        all_rows[i] = row

    tiles = []

    for row_num in range(0,9):
        for col_num in range(0, 9):
            # Define the coordinates for slicing: [y1:y2, x1:x2]
            x1, y1 = all_rows[row_num, col_num]
            x2, y2 = all_rows[row_num+1, col_num+1]
            
            tile = image[int(y1):int(y2), int(x1):int(x2)]
            tiles.append(tile)
            
            # Optional: Save each tile
            cv2.imshow("tile", cv2.resize(tile, (500, 500)))
            cv2.resizeWindow("tile", 500, 500)
            wait_q()
            cv2.imwrite(f'media/tiles/tile_{row_num}_{col_num}.jpg', tile)

def remove_from_array(base_array, test_array):
    # print(base_array) 
    # print(test_array)
    # print(base_array[2])
    # print(np.array_equal(base_array[2], test_array))
    for index in range(len(base_array)):
        if np.array_equal(base_array[index], test_array):
            base_array.pop(index)
            return base_array
    raise ValueError('remove_from_array(array, x): x not in array')



def get_line_intersections(lines, image):
    colour_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    length = image.shape[0]
    lines_copy = copy.deepcopy(lines)
    horizontal = []
    vertical = []
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines_copy[i][0]
        if abs(x1 - x2) > abs(y1 - y2):
        # x2 is larger than x1
            new_y1 = int(y1 - (y2-y1) * x1/(x2-x1))
            new_y2 = int(y2 + (y2-y1) * (length-x2)/(x2-x1))
            lines[i][0] = np.array([0, new_y1, length, new_y2])
            horizontal.append(lines[i])
        else:
            # y2 is smaller than y1
            new_x1 = int(x1 - (x2-x1) * y2/(y1-y2))
            new_x2 = int(x2 + (x2-x1) * (length-y1)/(y1-y2))
            lines[i][0] = np.array([new_x1, length, new_x2, 0])
            vertical.append(lines[i])
    

    for line_1 in horizontal[:]:
        for line_2 in horizontal:
            x1_1, y1_1, x1_2, y1_2 = line_1[0]
            x2_1, y2_1, x2_2, y2_2 = line_2[0]
            if [x1_1, y1_1, x1_2, y1_2] != [x2_1, y2_1, x2_2, y2_2]:
                if abs(y1_1 - y2_1) < length/20 and abs(y2_1 - y2_2) < length/20:
                    if abs(y1_1 - y1_2) <  abs(y2_1 - y2_2): 
                        try: horizontal = remove_from_array(horizontal, line_2)
                        except ValueError: 
                            print("cant remove")
                            pass
                    else:
                        try: horizontal = remove_from_array(horizontal, line_1)
                        except ValueError: pass
        

    for line_1 in vertical[:]:
        for line_2 in vertical:
            x1_1, y1_1, x1_2, y1_2 = line_1[0]
            x2_1, y2_1, x2_2, y2_2 = line_2[0]
            if [x1_1, y1_1, x1_2, y1_2] != [x2_1, y2_1, x2_2, y2_2]:
                if abs(x1_1 - x2_1) < length/20 and abs(x2_1 - x2_2) < length/20:
                    if abs(x1_1 - x1_2) <  abs(x2_1 - x2_2):
                        try: vertical = remove_from_array(vertical, line_2)
                        except ValueError: pass
                    else:
                        try: vertical = remove_from_array(vertical, line_1)
                        except ValueError: pass
            
    print(len(horizontal), len(vertical))
    lines = horizontal + vertical

    intersections = []
    for h_line in horizontal:
        for v_line in vertical:
            x1, y1, x2, y2 = h_line[0]
            x3, y3, x4, y4 = v_line[0]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue

            # Determinant formula for intersection
            intersect_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            intersect_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

            intersections.append((int(intersect_x), int(intersect_y)))


    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(colour_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for intersection in intersections:
        cv2.circle(colour_image, intersection, 5, (0, 0, 255), -1)
    cv2.imshow("colour_image", cv2.resize(colour_image, (500, 500)))
    cv2.resizeWindow("colour_image", 500, 500)
    wait_q()

    return intersections
        

    

in_omr = cv2.imread("media/sodokutest.jpeg", cv2.IMREAD_COLOR)
page = find_page(in_omr, "media/sodokutest.jpeg" )
lines, page = get_lines(page)
intersections = get_line_intersections(lines, page)
get_squares(page, intersections)