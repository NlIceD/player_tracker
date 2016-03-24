import numpy as np
import court
import court_warp
import cv2

def get_all_lines(vid):
    all_lines = []
    for frame in vid:
        #lines = court.get_box_lines(frame, hist1_thresh = .015, hough1 = 30, hist2_thresh=.107, ft_thresh = 20, paint_thresh = 30)
        lines = court.get_box_lines(frame, hist1_thresh = .015, hough1 = 30, hist2_thresh=.107, ft_thresh = 24, paint_thresh = 65)
        all_lines.append(lines)
    return all_lines

def get_all_points(all_lines):
    all_points = []
    for line_set in all_lines:
        if line_set == False:
            all_points.append(line_set)
        else:
            all_points.append(court_warp.get_points(*line_set))
    return all_points

def interpolate_points(all_points):
    maxes = []
    initial = np.mean(np.array([point for point in all_points if type(point) != bool]), axis=0)
#     last_valid = next(points for points in all_points if points is not False)
    new_points = []
    new_points.append(initial)
    for i in range(1,len(all_points)):
#         print "ALL POINTS",all_points[i], "\n", "NEW POINTS",new_points[i-1]
#         print "WHAT", all_points[i]-new_points[i-1]
#         print 'MAX', np.amax(abs(all_points[i]-new_points[i-1]))
        if type(all_points[i]) == bool or np.amax(abs(all_points[i]-new_points[i-1])) > 40:
            new_points.append(new_points[i-1])
        else:
            new_points.append(all_points[i])
        maxes.append(np.amax(new_points[i]-new_points[i-1]))
    return new_points

def color_percentage(image, boxes):
    '''
    Input: image, Predicted boxes
    Output: percentage of white, percentage of purple in those boxes
    '''
    classification = []
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for box in boxes:
        WHITE_MIN = np.array([130, 0, 100],np.uint8)
        WHITE_MAX = np.array([150, 100, 255],np.uint8)

        PURPLE_MIN = np.array([140, 100, 5],np.uint8)
        PURPLE_MAX = np.array([200, 255, 255],np.uint8)

        white_threshed = cv2.inRange(img[box[1]:box[3],box[0]:box[2]], WHITE_MIN, WHITE_MAX)
        purple_threshed = cv2.inRange(img[box[1]:box[3],box[0]:box[2]], PURPLE_MIN, PURPLE_MAX)
        total = float(255*(box[3]-box[1])*(box[2]-box[0]))
        white_ratio = np.sum(white_threshed)/total
        purple_ratio = np.sum(purple_threshed)/total
        classification.append((white_ratio,purple_ratio))

    return classification

def box_overlap(boxes):
    '''
    Input: Predicted boxes
    Output: Sets of boxes that overlap
    '''
    overlap = []
    for i, box in enumerate(boxes):
        sublist = boxes[i+1:]
        box_set = set([])
        for j, subbox in enumerate(sublist):
            x = min(box[2],subbox[2]) - max(box[0],subbox[0])
            y = min(box[3],subbox[3]) - max(box[1],subbox[1])
            if x > 0 and y > 0:
#                 area = x * y
#                 a.append(area)
                box_set.add(i)
                box_set.add(i+1+j)
        if box_set:
            overlap.append(box_set)
    return overlap

def team_classify(image, boxes):
    '''
    Input: image, Predicted boxes
    Output: Team based on color.  -1 is neither, 1 is white, 2 is purple, 3 is mixed
    '''
    colors = color_percentage(image, boxes)
    overlaps = box_overlap(boxes)

    team = []
    over = [{-1:0,1:0,2:0,3:0}]*len(overlaps)

    for i, color in enumerate(colors):
        white, purple = color
        print white, purple
        if white < .1 and purple < .05:
            print 'MADE IT'
            team.append(-1)
        else:
            if white - purple > .1:
                team.append(1)
            elif purple - white > .1:
                team.append(2)
            else:
                #if i in overlap_set:
                team.append(3)
    for ind, group in enumerate(overlaps):
        for temp in group:
            over[ind][team[temp]] = over[ind][team[temp]] + 1
    for i, val in enumerate(team):
        if val == 3:
            for ind, group in enumerate(overlaps):
                if i in group:
                    if over[ind][1] > over[ind][2]:
                        team[i] = 2
                        over[ind][2] = over[ind][2] + 1
                    elif over[ind][1] < over[ind][2]:
                        team[i] = 1
                        over[ind][1] = over[ind][1] + 1
    return team
