def tracking_update(tracking, transp,dt):
    delete_track = []

    for i in range(0,len(tracking)):
        threshold = 0.7
        iou_max = 0
        id_max = 0
        iou=[]

        for j in range(0, len(transp)):
            if(tracking[i][-1] == transp[j][-1]):
                score = cal_iou(tracking[i], transp[j])
                if (score > threshold):
                    iou.append([score,j])
        
        # find iou max
        if(len(iou)>0):
            for k in range(0,len(iou)):
                if(iou[k][0] > iou_max):
                    iou_max = iou[k][0]
                    id_max = iou[k][1]

            tracking[i] = transp[id_max]
            dt[i] += 1
        else:
            delete_track.append(i)


    tracking = delete_list(delete_track,tracking)
    dt = delete_list(delete_track,dt)

def delete_list(index,list):
    ix = index
    ix.sort()
    num = len(ix)
    if(len(list)>0 and num > 0):
        for i in range(0,num):
            del list[ix[i]]
            
            for j in range((i+1), len(ix)):
                ix[j] -= 1

    return list

        
def cal_iou(tracking, transp):
    # box = (x1, y1, x2, y2)
    box1_area = (tracking[2] - tracking[0] + 1) * (tracking[3] - tracking[1] + 1)
    box2_area = (transp[2] - transp[0] + 1) * (transp[3] - transp[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(tracking[0], transp[0])
    y1 = max(tracking[1], transp[1])
    x2 = min(tracking[2], transp[2])
    y2 = min(tracking[3], transp[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def in_ROI(cd,roi):
    if(cal_iou(cd,roi) > 0):
        return True
    else:
        return False

def is_near(cd,x,y):
    cd_x = (cd[0]+cd[2])/2
    cd_y = (cd[1]+cd[3])/2

    for i in range(0,len(x)):
        length = abs(cd_x - x[i]) + abs(cd_y - y[i])
        if(length < 100):
            return True

def not_existence(tl1,tl2):
    for i in range(0,len(tl1)):
        if(cal_iou(tl1[i],tl2) < 0.4):
            return True
        else:
            return False

def count_person(det):
    n = 0
    for i in range(0, len(det)):
        class_id = int(det[i,-1])
        if(class_id == 0):
            n += 1

    return n

def classify(det,person_x,person_y,ROI,transp):
    for j in range(0,len(det)):
        class_id = int(det[j,-1])
        class_cd = [det[j, 0],det[j, 1],det[j, 2],det[j, 3]]

        if(class_id == 0) : # person
            if(in_ROI(class_cd, ROI)) :
                person_x.append((class_cd[0] + class_cd[2])/2)
                person_y.append((class_cd[1] + class_cd[3])/2)

        elif(class_id == 2): # car or bus or truck # 2,5,7
            if(in_ROI(class_cd, ROI)):
                class_cd.append(class_id)
                transp.append(class_cd)

def choose_tracking(person_x, person_y,transp,tracking,dt):
    for k in range(0, len(transp)):
        if(is_near(transp[k], person_x, person_y)):
            if(len(tracking) > 0):
                if(not_existence(tracking,transp[k])):
                    tracking.append(transp[k])
                    dt.append(0)
            else:
                tracking.append(transp[k])
                dt.append(0)

