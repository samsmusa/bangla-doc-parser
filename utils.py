import logging
import math
import time
from enum import Enum
from functools import wraps
from typing import List
from typing import NamedTuple
from typing import Tuple, Union

import cv2
import imutils
import numpy as np
from deskew import determine_skew
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from skimage.transform import rotate

logger = logging.getLogger(__name__)


class DocType(str, Enum):
    Old = "Old"
    New = "New"

    def __str__(self):
        return self.value


class Point(NamedTuple):
    x: int
    y: int


class Line(NamedTuple):
    p1: Point
    p2: Point


class HLine(NamedTuple):
    l: Line
    r: float
    t: float


class Box(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


class CRange(NamedTuple):
    """
    cluster range. min max value of a cluster points.
    """
    min: int
    max: int


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


class Processor(BaseModel):
    crop: Tuple[int, int]


class Config(BaseModel):
    preprocess: Union[Processor, None]


config = {
    "preprocess": {
        "crop": [700, 1500]
    }
}

CONFIG = Config(**config)


# Box = namedtuple('Box', ("x1", "y1", "x2", "y2"))
# Point = namedtuple('Point', ('x', 'y'))
# Line = namedtuple("Line", ('p1', 'p2'))
# HLine = namedtuple("HLine", ('l', 'r', 't'))


def draw_lines(image, lines: List[Line]):
    res_img = image.copy()
    for i, line in enumerate(lines):
        cv2.line(res_img, line.p1, line.p2, (0, 0, 255) if i % 2 == 0 else (255, 0, 0), 2)
    return res_img


def draw_Hline(image, lines: List[HLine], color=(0, 0, 255), thickness=2):
    res_img = image.copy()
    for line in lines:
        cv2.line(res_img, line.l.p1, line.l.p2, color, thickness=thickness)
    return res_img


def is_aligned_box(ref_box: Box, box: Box):
    center = Point((box.x1 + box.x2) // 2, (box.y1 + box.y2) // 2)
    if ref_box.y1 - 5 <= center.y <= ref_box.y2 + 5:
        return True

    if ref_box.y1 <= box.y1 <= ref_box.y2 or ref_box.y1 <= box.y2 <= ref_box.y2:
        return True
    return False


def get_rows_from_columns(columns: List[List[Box]]) -> List[List[Box]]:
    sorted_columns = [sorted(column, key=lambda x: x.y1) for column in columns]
    rows = [[] for _ in range(len(sorted_columns[0]))]
    for r, first_box in enumerate(sorted_columns[0]):
        rows[r].append(first_box)
        pre_box = first_box

        for next_col in sorted_columns[1:]:
            next_box = next((box for box in next_col if is_aligned_box(pre_box, box)), None)

            if next_box:
                pre_box = next_box
                rows[r].append(next_box)
            else:
                rows[r].append(None)
    return rows


def show_table(data: List[List[Box]], title=""):
    cols = len(data[0])
    table = Table(title=title)
    for i in range(cols):
        table.add_column(f"col-{i + 1}", justify="right", style="cyan", no_wrap=True)
    for row in data:
        table.add_row(*list(map(str, row)))

    console = Console()
    console.print(table)


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
    if event == cv2.EVENT_RBUTTONDOWN:
        pass


def cv2_imshow(image, winname='window', wait=1000, window_size=(400, 560)):
    img = image.copy()
    cv2.namedWindow(f'{winname}', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(f'{winname}', mouse_click)
    cv2.resizeWindow(f'{winname}', window_size[0], window_size[1])
    cv2.imshow(f'{winname}', img)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()


def get_rows(bounding_boxes, img):
    image = img.copy()
    dy, dx = image.shape[:2]
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[1])
    current_box = sorted_boxes[0]
    merged_boxes = []
    for i, next_box in enumerate(sorted_boxes[1:]):
        if next_box[1] - current_box[1] < 25:  # Adjust the threshold as needed
            # Combine boxes if their y-values are close
            current_box = (
                min(current_box[0], next_box[0]), min(current_box[1], next_box[1]), current_box[2] + next_box[2],
                max(current_box[3], next_box[3]))
            if i == len(sorted_boxes) - 2:
                merged_boxes.append(current_box)
        else:
            # If y-values are not close, start a new merged box
            merged_boxes.append(current_box)
            current_box = next_box
    test_image = image[int(0.12 * dy):, :]
    line_segment = []
    for i, box in enumerate(merged_boxes):

        if i == 0:
            line_segment.append(test_image[0:merged_boxes[i + 1][1], :])
        elif i == len(merged_boxes) - 1:
            line_segment.append(test_image[merged_boxes[i - 1][3]:, :])
        else:
            line_segment.append(test_image[merged_boxes[i - 1][3]:merged_boxes[i + 1][1], :])


def get_bounding_idx_column(img):
    image = img.copy()
    dy, dx = image.shape[:2]
    col_1 = image[int(0.12 * dy):, 0:int(0.06 * dx)]
    gray = cv2.cvtColor(col_1, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 55 and 4 < w < 50:
            bounding_boxes.append((x, y, x + w, y + h))
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return bounding_boxes


def get_image(filename='12.png', debug=False):
    img_color = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = np.copy(img_color)
    img = cv2.resize(img, CONFIG.preprocess.crop)
    return img


def resize_image(image, width=300):
    return imutils.resize(image, width=width)


def get_lines(h_lines) -> List[Line]:
    lines = []
    for line in h_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        lines.append(Line(Point(x1, y1), Point(x2, y2)))
    return lines


def get_Hline(h_lines) -> List[HLine]:
    lines = []
    for line in h_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        lines.append(HLine(Line(Point(x1, y1), Point(x2, y2)), rho, theta))
    return lines


def point_between_lines(x, y, line1: HLine, line2: HLine):
    r1, theta1 = line1.r, line1.t
    r2, theta2 = line2.r, line2.t

    # Calculate the distances from the point to the lines
    dist1 = x * np.cos(theta1) + y * np.sin(theta1) - r1
    dist2 = x * np.cos(theta2) + y * np.sin(theta2) - r2

    # Check if the point lies between the two lines
    return np.sign(dist1) != np.sign(dist2)


def point_intersected_lines(x, y, _lines: List[HLine]):
    lines = _lines
    for i, line in enumerate(lines[:-1]):
        if point_between_lines(x, y, line, lines[i + 1]):
            return i


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bitwise_not_img = cv2.bitwise_not(gray_image)
    return cv2.adaptiveThreshold(bitwise_not_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, -2)


def get_sorted_column_lines(lines: List[HLine], threshold=10, center=False) -> List[HLine]:
    sorted_lines = sorted(lines, key=lambda x: x.l.p1.x)
    column_lines = []
    ref_line = sorted_lines[0]
    last_line = None
    for a in sorted_lines[1:]:
        distance = abs(ref_line.l.p1.x - a.l.p1.x) / 2 if center else abs(ref_line.l.p1.x - a.l.p1.x)
        if distance > threshold:
            column_lines.append(ref_line)
            last_line = a
        ref_line = a
    column_lines.append(last_line)
    return column_lines

    # for a, b in zip(sorted_lines, sorted_lines[1:]):
    #     distance = abs(b.l.p1.x - a.l.p1.x) / 2 if center else abs(b.l.p1.x - a.l.p1.x)
    #     if distance > threshold:
    #         column_lines.append(b)
    #     else:
    #         pass
    #         # column_lines.append(a)
    # return column_lines


def get_tangent_c(line: Line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    m = (y2 - y1) / (x2 - x1 + 1e-10)  # Avoid division by zero
    c = y1 - m * x1
    return m, c


def get_vertical_p_lines(img, kernel_ratio=30, threshold=230, theta=1, pi=np.pi, minLineLength=100, maxLineGap=100):
    vertical = np.copy(img)

    rows = vertical.shape[0]
    vertical_size = rows // kernel_ratio
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    lines = cv2.HoughLinesP(vertical, cv2.HOUGH_PROBABILISTIC, pi, threshold=threshold, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    return lines, vertical


def get_min_line_from_center(src, lines):
    h, w = src.shape[:2]
    min_y = 100000
    min_id = 0
    for idx, line in enumerate(lines):
        hh = h / 2 - line[0][1]
        if 0 < hh < min_y:
            min_y = hh
            min_id = idx

    return [lines[min_id]]


def get_first_col(image):
    pass


def divide_image_hr(img, lines):
    image = img.copy()
    h, w = image.shape[:2]
    if not lines or len(lines[0]) != 2:
        return image, None

    line_y = (lines[0][0][1] + lines[0][1][1]) // 2

    upper_portion = image[:line_y + 3, :]
    lower_portion = image[line_y + 3:, :]

    return upper_portion, lower_portion


def get_vertical_lines(img, kernel_ratio=30, threshold=230, theta=1, pi=np.pi):
    vertical = np.copy(img)

    rows = vertical.shape[0]
    vertical_size = rows // kernel_ratio
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    lines = cv2.HoughLines(vertical, theta, np.pi, threshold=threshold)
    return lines, vertical


def get_horizontal_lines(img, kernel_ratio=30, threshold=230):
    horizontal = img.copy()
    cols = horizontal.shape[1]
    horizontal_size = cols // kernel_ratio
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    lines = cv2.HoughLines(horizontal, 1, np.pi / 90, threshold=threshold)
    return lines, horizontal


def create_text_heatmap(img):
    image = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Dilation to make text regions more prominent
    kernel = np.ones((7, 7), np.uint8)
    kernel[5:, 5:] = 0
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Create a gradient from right to left
    gradient = np.tile(np.arange(image.shape[1], 0, -1), (image.shape[0], 1))
    gradient = gradient / gradient.max()  # Normalize to [0, 1]

    # Apply the gradient to the dilated image
    heatmap = gradient * dilated
    return heatmap


def get_deskew_image(image, width=1000):
    image = imutils.resize(image, width=width)
    gray_image = preprocess_image(image)
    _, vertical = get_vertical_lines(gray_image, 30)

    grayscale = vertical.copy()
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle) * 255

    image = rotated.astype(np.uint8)
    new = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray, new


def remove_v_lines(gray, image):
    gray_image = gray.copy()
    thick_kernel = np.ones((1, 3))
    vertical = np.copy(gray_image)
    rows = vertical.shape[0]
    vertical_kernel = rows // 30
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure, iterations=5)
    vertical = cv2.dilate(vertical, thick_kernel)
    image_without_v_lines = image.copy()
    image_without_v_lines[np.where(vertical == 255)] = 255
    return image_without_v_lines


def get_columns(text_boxes, v_lines):
    column_data = [[] for _ in range(len(v_lines))]
    for box in text_boxes:
        center_box = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        idx = point_intersected_lines(*center_box, v_lines)
        if idx is not None:
            column_data[idx].append(box)
    return column_data


def get_rows_data(column_data):
    column_data_new = [column for i, column in enumerate(column_data[1:]) if (2 < i < len(column_data) - 2) or column]
    rows = get_rows_from_columns(column_data_new)
    return rows, column_data_new


def draw_boxes(image, boxes, color=(0, 0, 255), thickness=2):
    for box in boxes:
        if type(box) == list:
            draw_boxes(image, box)
        elif box is None:
            continue
        else:
            cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
    return image


def draw_points(image, points: List[Union[List[Point] | Point]], odd_color=(0, 0, 255), even_color=(255, 0, 0),
                thickness=-1, odd=True):
    for i, item in enumerate(points):
        if type(item) == list:
            draw_points(image, item, odd=i % 2 == 0)
        elif item is None:
            pass
        else:
            cv2.circle(image, (int(item.x), int(item.y)), 8, odd_color if odd else even_color, thickness)
    return image


def get_table_box(vertical_image, horizontal_image, vertical_sp=50, horizontal_sp=50, padding=10):
    v_image = cv2.rotate(vertical_image, cv2.ROTATE_90_CLOCKWISE)
    v_histogram = np.sum(v_image[int(v_image.shape[0] / 2):, :], axis=0)
    h_histogram = np.sum(horizontal_image[int(horizontal_image.shape[0] / 2):, :], axis=0)
    h_derivative = np.gradient(h_histogram)
    v_derivative = np.gradient(v_histogram)
    w_range = max(0, np.argmax(h_derivative[:vertical_sp]) - padding), max(0, padding + len(h_derivative) - (
            vertical_sp - np.argmin(h_derivative[-vertical_sp:])))
    h_range = max(0, np.argmax(v_derivative[:horizontal_sp]) - padding), max(0, padding + len(v_derivative) - (
            horizontal_sp - np.argmin(v_derivative[-horizontal_sp:])))

    return h_range, w_range


def merge_boxes_x(bounding_boxes: List[Box], threshold=8) -> List[Box]:
    sorted_boxes_x = sorted(bounding_boxes, key=lambda box: box[0])
    current_box = sorted_boxes_x[0]
    merged_boxes = []
    for i, next_box in enumerate(sorted_boxes_x[1:]):
        if abs(next_box.x1 - current_box.x1) < threshold:
            current_box = Box(min(current_box.x1, next_box.x1), min(current_box.y1, next_box.y1),
                              max(current_box.x2, next_box.x2), max(current_box.y2, next_box.y2))
        else:
            merged_boxes.append(current_box)
            current_box = next_box
        if i == len(sorted_boxes_x) - 2:
            merged_boxes.append(current_box)

    return merged_boxes


def merge_boxes_y(bounding_boxes: List[Box], threshold=25):
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box.y1)
    current_box = sorted_boxes[0]
    merged_boxes = []
    for i, next_box in enumerate(sorted_boxes[1:]):
        if abs(next_box.y1 - current_box.y1) < threshold:
            current_box = Box(min(current_box.x1, next_box.x1), min(current_box.y1, next_box.y1),
                              max(current_box.x2, next_box.x2), max(current_box.y2, next_box.y2))
        else:
            merged_boxes.append(current_box)
            current_box = next_box
        if i == len(sorted_boxes) - 2:
            merged_boxes.append(current_box)

    return merged_boxes


def extract_data(indexes, predictions):
    extract_text = []
    for row_key, row_value in indexes.items():
        row = []
        for col_key, col_value in row_value.items():
            row.append(''.join(predictions[col_value[0]: col_value[1]]))
        extract_text.append(row)
    return extract_text


@timeit
def apply_morph(image, erode_kernel=(1, 50), dialate_kernel=(1, 50), eiter=1, diter=2):
    erosion_kernel = np.ones(erode_kernel, np.uint8)
    dialation_kernel = np.ones(dialate_kernel, np.uint8)
    morphed_image = cv2.erode(image, erosion_kernel, iterations=eiter)
    morphed_image = cv2.dilate(morphed_image, dialation_kernel, iterations=diter)
    return morphed_image


def get_y(x, x1, y1, x2, y2):
    y = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1
    return int(y)


def get_point(line: Line, x, width):
    x1, y1 = line.p1
    x2, y2 = line.p2
    y_left = get_y(x, x1, y1, x2, y2)
    y_right = (get_y(width, x1, y1, x2, y2))
    return Line(Point(x, y_left), Point(width, y_right))


def four_point_transform(image, pts):
    pts = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    return warped


def find_intersection(line1: Line, line2: Line):
    x1, y1 = line1.p1
    x2, y2 = line1.p2
    x3, y3 = line2.p1
    x4, y4 = line2.p2
    # Calculate slopes (m) and intercepts (b) for each line
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    b1 = y1 - m1 * x1 if m1 != float('inf') else x1  # b = y - mx for finite slope, b = x for vertical line

    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')
    b2 = y3 - m2 * x3 if m2 != float('inf') else x3  # b = y - mx for finite slope, b = x for vertical line

    # Handle vertical lines (infinite slope)
    if m1 == float('inf') and m2 == float('inf'):
        return None  # Parallel vertical lines, no intersection
    elif m1 == float('inf'):  # Line 1 is vertical
        x_intersect = x1
        y_intersect = m2 * x_intersect + b2
    elif m2 == float('inf'):  # Line 2 is vertical
        x_intersect = x3
        y_intersect = m1 * x_intersect + b1
    else:
        # Calculate intersection point
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1

    return Point(int(x_intersect), int(y_intersect))


def make_gray(image: np.ndarray, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 15)
    gray[thresh == 255] = (255)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    dst = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = np.ones((1, 1))
    dst = cv2.erode(dst, kernel)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    dst = cv2.filter2D(dst, -1, kernel)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    if kwargs.get('debug', None):
        cv2_imshow(dst, "making gray", 0)
    return dst


@timeit
def get_horizontal_ratio(horizontal_image):
    h_histogram = np.sum(horizontal_image, axis=0)
    indices_greater_than_3000 = np.where(h_histogram > 3000)[0]
    min_index = np.min(indices_greater_than_3000)
    max_index = np.max(indices_greater_than_3000)
    threshold = int(abs(max_index - min_index) // 2.5)
    return min_index, max_index, threshold


@timeit
def get_vertical_ratio(vertical_image):
    v_image = cv2.rotate(vertical_image, cv2.ROTATE_90_CLOCKWISE)
    v_histogram = np.sum(v_image[:int(v_image.shape[0] / 2), :], axis=0)
    indices_greater_than_5000 = np.where(v_histogram > 5000)[0]
    min_index = np.min(indices_greater_than_5000)
    max_index = np.max(indices_greater_than_5000)
    threshold = (max_index - min_index) // 3
    return min_index, max_index, threshold


def half_percentage(x, a=0.5, b=200, c=-1.005):
    x = x // 100
    return round(a / (1 + b * math.exp(c * x)), 2)
