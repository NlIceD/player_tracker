import numpy as np
from skimage import transform as tf

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


def get_homography(points):
    CONSTANT = 0
    src = np.array((
        (0, 0+CONSTANT),
        (228, 0+CONSTANT),
        (228, 396+CONSTANT),
        (0, 396+CONSTANT)
    ))

    dst = points
    tform = tf.ProjectiveTransform()
    tform.estimate(dst, src)
    return tform.params
