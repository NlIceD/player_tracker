import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from skimage.util import pad, crop

### Functions for Baseline/Sideline
def get_dominant_colorset(_bgr_img, thresh=0.02):
    '''
    input: image
    Returns: dominant colorset using YCR
    '''

    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)[80:340]
    hist = cv2.calcHist([img], [1,2], None, [256,256], [0,256,0,256])

    peak1_flat_idx = np.argmax(hist)
    peak1_idx = np.unravel_index(peak1_flat_idx, hist.shape)
    peak1_val = hist[peak1_idx]
    connected_hist1, sum1, subtracted_hist = get_connected_hist(hist,peak1_idx,thresh)

    return connected_hist1



def create_court_mask(_bgr_img, dominant_colorset, binary_gray=True):
    '''
    Inputs: image, dominant_colorset
    Returns: binary mask of image based on dominant colorset
    using YCR as the color filter
    '''

    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)
    #img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2HSV)
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            idx = (row, col)
            h, cr, cb = img[idx]
            #print img[idx]
            if (cr,cb) not in dominant_colorset:
                img[idx] = (0,128,128) #BLACK
            elif binary_gray:
                img[idx] = (255,128,128) #WHITE

    return ycbcr_to_gray(img) if binary_gray else img

def get_baseline_sideline(bgr_img, thresh1 = .015, thresh2 = 40):
    '''
    Inputs: BGR Image, Threshold for color hist, Threshold for HoughLines
    Returns: Sideline, Baseline if two lines exist.  Otherwise Return False
    '''

    color_set = get_dominant_colorset(bgr_img, thresh1)
    newest_img = create_court_mask(bgr_img,color_set, binary_gray=True)
    flooded_img = get_double_flooded_mask(newest_img)
    flooded_img = dilate_erode(flooded_img)
    lines = get_lines(flooded_img,thresh2)
    if not lines:
        return False
    average_lines = average_line_group(group_lines(lines))
    if len(average_lines) == 2:
        return average_lines[0], average_lines[1]
    else:
        return False
### Functions for baseline/sideline and FT/paint line

def get_connected_hist(hist, peak_idx, thresh):
    '''
    Input: histogram, peak index, threshold percentage of max peak
    Output: Histogram of colors near the peak index based on threshold
    '''
    connected_hist = set()
    sum_val = 0
    subtracted_hist = np.copy(hist)

    min_passing_val = thresh * hist[peak_idx]

    connected_hist.add(peak_idx)
    sum_val += hist[peak_idx]
    subtracted_hist[peak_idx] = 0
    queue = deque([peak_idx])

    while queue:
        x, y = queue.popleft()
        toAdd = []
        if x > 1:
            toAdd.append((x-1,y))
        if x < hist.shape[0] - 1:
            toAdd.append((x+1,y))
        if y > 1:
            toAdd.append((x, y-1))
        if y < hist.shape[1] - 1:
            toAdd.append((x, y+1))

        for idx in toAdd:
            if idx not in connected_hist and hist[idx] >= min_passing_val:
                connected_hist.add(idx)
                sum_val += hist[idx]
                subtracted_hist[idx] = 0
                queue.append(idx)

    return connected_hist, sum_val, subtracted_hist

def fill_holes_with_contour_filling(gray_mask,inverse=False):
    '''
    Input: Grayscale image
    Returns: Image with holes filled by contour mapping
    '''

    filled = gray_mask.copy()
    filled = pad(filled,((5,5),(0,0)),'constant',constant_values=255)
    if inverse:
        filled = cv2.bitwise_not(filled)
    image, contour, _ = cv2.findContours(filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
            cv2.drawContours(filled,[cnt], 0, 255, -1)
    if inverse:
        filled = cv2.bitwise_not(filled)
    filled = crop(filled,((5,5),(0,0)))
    return filled

def get_double_flooded_mask(gray_mask):
    '''
    Input: Grayscale image
    Returns: Image with holes filled by contour mapping

    Purpose: Inverts the map and fills the contours again.
    '''
    gray_flooded = fill_holes_with_contour_filling(gray_mask)
    gray_flooded2 = fill_holes_with_contour_filling(gray_flooded, inverse=True)
    return gray_flooded2

def dilate_erode(image):
    kernel = np.ones((5,5),np.uint8)

    dilated = cv2.dilate(image,kernel,iterations=10)
    dilated = cv2.erode(dilated,kernel,iterations=10)
    dilated = cv2.erode(dilated,kernel,iterations=10)
    dilated = cv2.dilate(dilated,kernel,iterations=10)

    return dilated

def erode_dilate(image):
    kernel = np.ones((5,5),np.uint8)

    dilated = cv2.erode(image,kernel,iterations=10)
    dilated = cv2.dilate(dilated,kernel,iterations=10)
    dilated = cv2.dilate(dilated,kernel,iterations=10)
    dilated = cv2.erode(dilated,kernel,iterations=10)

    return dilated

def get_lines(gray, thresh=55):
    '''
    Input: Grayscale image
    Returns: Possible lines for that image
    '''

    #flooded = fill_holes_with_contour_filling(gray, inverse=True)
    canny = cv2.Canny(gray.copy(), 50, 200)
    lines = cv2.HoughLines(canny[0:0.79*canny.shape[0]], 1, np.pi/180, thresh)
    normal_lines = []
    if lines is None:
        return False
    for rho,theta in lines.reshape(lines.shape[0],2):
        if rho < 0:
            rho = -rho
            theta = theta - np.pi
        normal_lines.append([rho,theta])
    return normal_lines

def put_lines_on_img(bgr_img, lines_rho_theta,y_shift = 0, x_shift = 0):
    '''
    Input: Image, lines
    Returns: Image with lines on it
    '''

    lined = bgr_img.copy()
    redness = np.linspace(0, 255, len(lines_rho_theta))
    redness = np.floor(redness)
    blueness = 255 - redness
    for i, (rho, theta) in enumerate(lines_rho_theta):
        # print 'The parameters of the line: rho = %s, theta = %s' %(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b)+x_shift)
        y1 = int(y0 + 1000*(a)+y_shift)
        x2 = int(x0 - 1000*(-b)+x_shift)
        y2 = int(y0 - 1000*(a)+y_shift)
        red = redness[i]
        blue = blueness[i]
        cv2.line(lined,(x1,y1),(x2,y2),(100,0,200),2)
    return lined

def group_lines(lines_rho_theta):
    '''
    Input: list of lines
    Returns: Groups of lines based on which are close to eachother
    '''

    line_groups = []
    rho, theta = lines_rho_theta[0]
    line_groups.append([[rho,theta]])

    for rho, theta in lines_rho_theta[1:]:
        new_group = True
        if rho < 0:
            rho = -rho
            theta = theta - np.pi
        for key in range(len(line_groups)):
            # Append to list if close to existing theta
            if abs(line_groups[key][0][1] - theta) < 0.2 and abs(line_groups[key][0][0] - rho) < 50:
                line_groups[key].append([rho, theta])
                new_group = False
                break
        if new_group:
            line_groups.append([[rho, theta]])

    return line_groups

def average_line_group(line_groups):
    '''
    Input: List of Groups of lines
    Returns: Returns averages for every group_lines
    '''

    averages = []
    votes_appended = []
    votes = map(len,line_groups)

    for key in range(len(line_groups)):
        average = np.average(line_groups[key],axis=0)
        rho, theta = average

        if abs(average[1]) < .1 or abs(average[1]-np.pi/2) < .0001:
            pass
        else:
            for idx, stored_avg in enumerate(averages):
                thetas = stored_avg[1]

                if abs(theta-thetas) < .3 and votes_appended[idx] > votes[key]:
                    #don't replace
                    #print 'DON"T REPLACE'
                    break
                elif abs(theta-thetas) < .3:
                    #replace existing value
                    #print 'REPLACE'
                    averages[idx] = average
                    votes_appended[idx] = votes[key]
                    break
                elif idx == len(averages)-1:
                    #print 'APPEND'
                    #append value to list
                    averages.append(average)
                    votes_appended.append(votes[key])
                    break
            if not averages:
                #print 'INITIALIZE'
                averages.append(average)
                votes_appended.append(votes[key])

    return averages



### Functions for freethrow line and paint line

def get_hsv_dominant_colorset(_bgr_img, thresh=0.02):
    '''
    input: image
    Returns: dominant colorset using HSV
    '''

    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2HSV)[40:340]
    hist = cv2.calcHist([img], [0], None, [256], [0,256])

    peak1_flat_idx = np.argmax(hist)
    peak1_idx = np.unravel_index(peak1_flat_idx, hist.shape)
    peak1_val = hist[peak1_idx]
    connected_hist1, sum1, subtracted_hist = get_connected_hist(hist,peak1_idx,thresh)

    return connected_hist1

def create_hsv_court_mask(_bgr_img, dominant_colorset, binary_gray=False):
    '''
    Inputs: image, dominant_colorset
    Returns: binary mask of image based on dominant colorset
    using HSV as the color filter
    '''

    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2HSV)[40:340]
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            idx = (row, col)
            h, cr, cb = img[idx]
            #print img[idx]
            if (h,0) not in dominant_colorset:
                img[idx] = (0,128,128) #BLACK
            elif binary_gray:
                img[idx] = (255,128,128) #WHITE

    return ycbcr_to_gray(img) if binary_gray else img

def get_freethrow_line(gray, baseline, thresh=30):
    '''
    Takes flooded HSV image and baseline vector as inputs.
    Returns a single line that should be the free throw lines.
    If multiple lines exist, or no line exists returns False
    '''

    lines = get_lines(gray,thresh)
    if not lines:
        return False

    possible_lines = []

    for rho,theta in lines:
        if abs(theta - baseline[1]) < .2 and abs(rho - baseline[0]) > 50:
            possible_lines.append([rho,theta])
    ## Just use try except next time...
    if not possible_lines:
        return False
    average_ft = average_line_group(group_lines(np.array(possible_lines)))

    if len(average_ft) > 1:
        ft_line = []
        for rho, theta in average_ft:
            if abs(baseline[1])-abs(theta) > 0:
                ft_line.append([rho,theta])
        if len(ft_line) == 1:
            return np.array(ft_line[0])
        else:
            return False

    if len(average_ft) == 1:
        return average_ft[0]
    else:
        return False

def get_paint_line(gray, sideline, thresh=75):
    '''
    Takes flooded HSV image and sideline vector as inputs.
    Returns a single line that should be the bottom part of the painted box.
    If multiple lines exist, or no line exists returns False
    '''

    lines = get_lines(gray,thresh)
    if not lines:
        return False

    possible_lines = []

    for rho,theta in lines:
        if abs(theta - sideline[1]) < .25 and rho - sideline[0] > 40:
            possible_lines.append([rho,theta])
    if not possible_lines:
        return False
    avg_lines = average_line_group(group_lines(np.array(possible_lines)))
    if len(avg_lines) == 1:
        return np.array(average_line_group(group_lines(np.array(possible_lines)))[0])
    else:
        return False

def get_box_lines(bgr_img, hist1_thresh = .015, hough1 = 30, hist2_thresh=.107, ft_thresh = 20, paint_thresh = 30):
    '''
    Input: Image, Hist thresh for sideline/baseline, hough for sideline/baseline,
        hist thresh for FT/paint line, hough thresh for ft line, hough thresh for paint line
    Returns: array of four lines, sideline, baseline, ft_line, paint_line
    Or False if all four lines don't exist.
    '''
    side_base = get_baseline_sideline(bgr_img, hist1_thresh, hough1)
    if side_base:
        sideline, baseline = side_base
    else:
        return False

    hsv_color_set = get_hsv_dominant_colorset(bgr_img, hist2_thresh)
    hsv_binary = create_hsv_court_mask(bgr_img, hsv_color_set, binary_gray=True)
    flooded_hsv = get_double_flooded_mask(hsv_binary)
    hsv_ft_line = get_freethrow_line(flooded_hsv, baseline, ft_thresh)
    hsv_paint_line = get_paint_line(flooded_hsv,sideline, paint_thresh)
    if type(hsv_ft_line) == bool or type(hsv_paint_line) == bool:
        flooded_hsv = erode_dilate(flooded_hsv)
        hsv_ft_line = get_freethrow_line(flooded_hsv, baseline, ft_thresh)
        hsv_paint_line = get_paint_line(flooded_hsv,sideline, paint_thresh)

    if type(hsv_ft_line) != bool and type(hsv_paint_line) != bool:
        return sideline, baseline, hsv_ft_line, hsv_paint_line
    else:
        #Maybe include sideline baseline in case it can be used to interpolate future
        #return sideline, baseline
        return False

# Color Helpers
def ycbcr_to_bgr(ycbcr_img):
    img = ycbcr_img.copy()
    return cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

def ycbcr_to_gray(ycbcr_img):
    img = ycbcr_img.copy()
    img = ycbcr_to_bgr(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ycbcr_to_binary(ycbcr_img):
    img = ycbcr_img.copy()
    return ycbcr_to_gray(img) > 128

def ycbcr_to_bgr(ycbcr_img):
    img = ycbcr_img.copy()
    return cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

def ycbcr_to_gray(ycbcr_img):
    img = ycbcr_img.copy()
    img = ycbcr_to_bgr(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
