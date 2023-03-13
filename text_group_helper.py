import math
import operator
import pdb

def get_max_key(dictionary):
    return max([key for key in dictionary.keys()])

def get_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_center(group):
    x = 0
    y = 0
    for point in group:
        x+=point[0]
        y+=point[1]
    return (x/len(group), y/len(group))

def get_distance_from_center(point, center):
    return math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)

def get_min_distance_from_center(point, group):
    center = get_center(group)
    return get_distance_from_center(point, center)

def get_min_distance(point, group):
    min_distance = 100000000
    for group_point in group:
        distance = get_distance(point, group_point)
        if distance < min_distance:
            min_distance = distance
    return min_distance

# def get_closest_group(point, groups,min_distance = 100000000):
#     closest_group = None
#     for group in groups.values():
#         distance = get_min_distance(point, group)
#         if distance < min_distance:
#             min_distance = distance
#             closest_group = group
#     return closest_group

def get_closest_group(point, groups, max_x_distance, max_y_distance):
    closest_group = None
    for k, group in groups.items():
        for gp in group:
            if abs(point[0] - gp[0]) < max_x_distance and abs(point[1] - gp[1]) < max_y_distance:
                closest_group = k
                break
    return closest_group


def make_text_groups(top_lefts, xd=600, yd=65):
    
    top_ls = top_lefts.copy()
    groups = {0:[top_ls.pop()]}
    while len(top_ls) > 0:
        p = top_ls.pop()
        # pdb.set_trace()
        closest_group = get_closest_group(p, groups, xd, yd)
        if closest_group is None:
            max_k = get_max_key(groups)
            groups[max_k +1] = [p]
        else:
            groups[closest_group].append(p)
    
    # check singleton groups
    merges = [] # 0 into 1
    for k in groups.keys():
        if len(groups[k]) != 1: continue # if the group has more than 1 point, skip it
        v = groups[k][0] # otherwise, get the first (and only) point in the group
        temp_groups = groups.copy()
        temp_groups.pop(k)
        cg = get_closest_group(v, temp_groups, xd, yd) # get the closest group
        if cg is None: continue
        else:
            merges.append((k, cg))
        # for kp in ov: # for each point in the other group
        #     if all([abs(v[0]-kp[0]) < xd, abs(v[1] - kp[1]) < yd]): # see if it's close enough
        #         groups[ok].append(v) # if so, add it to the other group
        #         found = True
        #         merges.append((k,ok)) # keep track of the merge
        #         break
    
    for m in merges:
        v = groups.pop(m[0])[0]
        groups[m[1]].append(v)
    
    return groups


def groups_by_index(groups, base_list):
    index_groups = {}
    top_lefts_base = [p[0][0] for p in base_list]
    for k, v in groups.items():
        index_groups[k] = [top_lefts_base.index(p) for p in v]
        
    return index_groups

def convert_ocr_to_text_groups(ocr_results, xthresh=600, ythresh=65):
    top_lefts = [ocrr[0][0] for ocrr in ocr_results]
    s_top_lefts = sorted(top_lefts, key=operator.itemgetter(1,0))
    tl_groups = make_text_groups(s_top_lefts, xthresh, ythresh)
    idx_groups = groups_by_index(tl_groups, ocr_results)
    output = []
    for v in idx_groups.values():
        d = {}
        d['detections'] = [ocr_results[i] for i in v]
        # now we need to get the min and maxs of all the boundaries
        min_x = min([p[0][0][0] for p in d['detections']])
        max_x = max([p[0][1][0] for p in d['detections']])
        min_y = min([p[0][0][1] for p in d['detections']])
        max_y = max([p[0][2][1] for p in d['detections']])
        d['bounds'] = [(min_x, min_y), (max_x, max_y)]
        d['upper_left'] = (min_x, min_y)
        d['upper_right'] = (max_x, min_y)
        d['lower_right'] = (max_x, max_y)
        d['lower_left'] = (min_x, max_y)
        d['box'] = [d['upper_left'], d['upper_right'], d['lower_right'], d['lower_left']]
        d['height'] = max_y - min_y
        d['width'] = max_x - min_x
        group_top_lefts = [p[0][0] for p in d['detections']]
        s_group_top_lefts = sorted(group_top_lefts, key=operator.itemgetter(1,0))
        group_index_top_lefts = [group_top_lefts.index(p) for p in s_group_top_lefts]
        # break the group into lines
        lines = [[group_index_top_lefts.pop()]]
        line_thresh = int(math.floor(ythresh/2))
        while len(group_index_top_lefts) > 0:
            idx = group_index_top_lefts.pop()
            tl = s_group_top_lefts[idx]
            last_line_y = max([ s_group_top_lefts[lines[-1][i]][1] for i in range(len(lines[-1])) ])
            if abs(tl[1] - last_line_y) < line_thresh:
                lines[-1].append(idx)
            else:
                lines.append([idx])
        text = ''
        for line in lines[::-1]:
            # print([  d['detections'][lidx][1][0] for lidx in line ])
            text += ' '.join([  d['detections'][lidx][1][0] for lidx in line ])
            text += '\n'
        print('text: ', text)
        d['text'] = text
        output.append(d)
    
    # last sort on the output, based on the upper left of the group
    return sorted(output, key= lambda x: (x['upper_left'][1], x['upper_left'][0]) )
            
    