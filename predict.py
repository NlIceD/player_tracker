import numpy as np
# import court
# import court_warp
import cv2

def get_box_coordinates(rects,conf=0.2):
    x_centers = [x.centerX() for x in [rect for rect in rects if rect.score > conf]]
    y_bottoms = [x.y2 for x in [rect for rect in rects if rect.score > conf]]
    corners = [(x.x1,x.y1,x.x2,x.y2) for x in [rect for rect in rects if rect.score > conf]]
    return x_centers, y_bottoms, corners


### FUNCTIONS TO DETERMINE THE TEAM OF PLAYERS
def color_percentage(image, boxes):
    ### Consider changing to flat number of pixels rather than percentage
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
        total = float((box[3]-box[1])*(box[2]-box[0]))
        white_total = np.sum(white_threshed)/255.
        purple_total = np.sum(purple_threshed)/255.
        white_ratio = white_total/total
        purple_ratio = purple_total/total
        classification.append((white_ratio, purple_ratio, white_total, purple_total))

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
    over = [{-1:0,1:0,2:0,3:0} for i in range(len(overlaps))]

    for i, color in enumerate(colors):
        white, purple, wt, pt = color
        # print white, purple
        if (white < .1 and purple < .05) and not (wt > 100 or pt > 50):
            team.append(-1)
        else:
            if white - purple > .09:
                team.append(1)
            elif purple - white > .05:
                team.append(2)
            else:
                # print colors[i]
                #if i in overlap_set:
                team.append(3)
    for ind, group in enumerate(overlaps):
        for box_ind in group:
            over[ind][team[box_ind]] = over[ind][team[box_ind]] + 1

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
                    else:
                        white, purple, wt, pt = colors[i]
                        if white > purple:
                            team[i] = 1
                        else:
                            team[i]=2
    for i, val in enumerate(team):
        if val == 3:
            white, purple, wt, pt = colors[i]
            if white > purple:
                team[i] = 1
            else:
                team[i] = 2
    return team

def new_team_classify(image, boxes):
    '''
    Input: image, Predicted boxes
    Output: Team based on color.  -1 is neither, 1 is white, 2 is purple, 3 is mixed
    '''
    colors = color_percentage(image, boxes)
    overlaps = box_overlap(boxes)

    team = [0 for i in range(len(boxes))]
    over = [{-1:0,1:0,2:0,3:0} for i in range(len(overlaps))]

    #first classify the easy ones (non overlapping)
    #add those to total counters
    #then classify the overlaps



    for i, color in enumerate(colors):
        white, purple = color
        # print white, purple
        if white < .1 and purple < .05:
            team.append(-1)
        else:
            if white - purple > .09:
                team.append(1)
            elif purple - white > .05:
                team.append(2)
            else:
                # print colors[i]
                #if i in overlap_set:
                team.append(3)
    for ind, group in enumerate(overlaps):
        for box_ind in group:
            over[ind][team[box_ind]] = over[ind][team[box_ind]] + 1
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
                    else:
                        white, purple = colors[i]
                        if white > purple:
                            team[i] = 1
                        else:
                            team[i]=2
                else:
                    white, purple = colors[i]
                    if white > purple:
                        team[i] = 1
                    else:
                        team[i]=2
    return team


def color_mask(team_color):
    '''Creates a mask for when the value is the list -1'''
    return [num!=-1 for num in team_color]

def out_of_bounds(boxes, xst,yst,team_color):
    mask = [ys > -100  and xs > -50 for xs,ys in zip(xst,yst)]
    return mask_values(boxes,mask), mask_values(xst,mask), mask_values(yst,mask), mask_values(team_color,mask)


def mask_values(val, mask):
    '''Returns a list with only the values in the mask which are true'''
    return [j for j, i in zip(val,mask) if i]

def mask_by_color(boxes, xst, yst, team_color):
    '''Removes the boxes and colors that are correspond to team_color of -1'''
    mask = color_mask(team_color)

    new_boxes = mask_values(boxes, mask)
    new_xs = mask_values(xst, mask)
    new_ys = mask_values(yst, mask)
    new_colors = mask_values(team_color, mask)

    return new_boxes, new_xs, new_ys, new_colors

def add_rectangles(image, acc_rects):
    '''
    Input: Image and boxes
    Output: image with boxes on it
    '''
    temp = np.copy(image)
    for rect in acc_rects:
            cv2.rectangle(image,
                (int(rect[0]), int(rect[1])),
                (int(rect[2]), int(rect[3])),
                (0,255,0),
                2)
    return image
