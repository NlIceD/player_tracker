import numpy as np
import court
from skimage import transform as tf
from joblib import Parallel, delayed

def get_intersection(line_1, line_2, y_offset1 = 0, y_offset2 = 0):
    rho_1, theta_1 = line_1
    rho_2, theta_2 = line_2
    a_1 = np.cos(theta_1)
    b_1 = np.sin(theta_1)
    slope_1 = a_1 / -b_1
    intercept_1 = rho_1 / b_1 + y_offset1

    a_2 = np.cos(theta_2)
    b_2 = np.sin(theta_2)
    slope_2 = a_2 / -b_2
    intercept_2 = rho_2 / b_2 + y_offset2

    x_intersection = (intercept_2 - intercept_1) / (slope_1 - slope_2)
    y_intersection = x_intersection * slope_1 + intercept_1

    return (x_intersection, y_intersection)

def get_points(side, base, ft, paint, padding = 60):
    '''
    Returns:
    intersection of base, side
    intersection of side, ft
    intersection of paint, ft
    intersection of base, paint
    '''
    points = [get_intersection(base,side, y_offset1=padding, y_offset2=padding),
              get_intersection(side, ft, y_offset1=padding, y_offset2=40+padding),
              get_intersection(paint,ft,y_offset1 = 40+padding, y_offset2 = 40+padding),
              get_intersection(base,paint, y_offset1=padding, y_offset2 = 40+padding)]
    return np.array(points)

def get_all_lines(vid, cores = 1):
    all_lines = Parallel(n_jobs = cores)(delayed(court.get_box_lines)(frame, hist1_thresh = .015, hough1 = 30, hist2_thresh=.107, ft_thresh = 24, paint_thresh = 65) for frame in vid)
    return all_lines

def get_all_points(all_lines):
    all_points = []
    for line_set in all_lines:
        if line_set == False:
            all_points.append(line_set)
        else:
            all_points.append(get_points(*line_set))
    return all_points

def interpolate_points(all_points):
    maxes = []
    initial = np.mean(np.array([point for point in all_points if type(point) != bool]), axis=0)
    new_points = []
    new_points.append(initial)
    for i in range(1,len(all_points)):
        if type(all_points[i]) == bool or np.amax(abs(all_points[i]-new_points[i-1])) > 40:
            new_points.append(new_points[i-1])
        else:
            new_points.append(all_points[i])
        maxes.append(np.amax(new_points[i]-new_points[i-1]))
    return new_points


def get_homography(points):
    '''Input: Array of points
    Returns: Transform Matrix
    '''
    CONSTANT = 0
    if points[0][0]>=points[3][0]:
        # print 'left side'
        src = np.array((
            (0, 0+CONSTANT),
            (228, 0+CONSTANT),
            (228, 396+CONSTANT),
            (0, 396+CONSTANT)
            ))
    else:
        # print 'Right side'
        src = np.array((
            (900, 0+CONSTANT),
            (1128, 0+CONSTANT),
            (1128, 396+CONSTANT),
            (900, 396+CONSTANT)
            ))

    dst = points
    tform = tf.ProjectiveTransform()
    tform.estimate(dst, src)
    return tform.params

def video_to_court(xs, ys, homography):
    '''
    Input: list of x, list of y, homography
    Output: transformed x and y
    '''
    xst, yst = zip(*tf.matrix_transform(zip(xs,ys),homography))
    return xst, yst
