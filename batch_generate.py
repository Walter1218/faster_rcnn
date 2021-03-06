import pandas as pd
import numpy as np
import cv2, utils
import random
import copy
import threading
import itertools
import numpy.random as npr
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def get_img_by_name(df,ind,size=(960,640), dataset = 'PASCAL_VOC'):
    file_name = df['FileName'][ind]
    #print(file_name)
    img = cv2.imread(file_name)
    img_size = np.shape(img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)
    if(dataset == 'PASCAL_VOC'):
        name_str = file_name.split('/')
        name_str = name_str[-1]
    else:
        name_str = file_name.split('/')
        #print(name_str)
        name = name_str[2] + '/' + name_str[3] + '/' + name_str[4] + '/' + name_str[5] + '/'
        #print(name)
        name_str = name + name_str[-1]
    #print(name_str)
    #print(file_name)
    bb_boxes = df[df['Frame'] == name_str].reset_index()
    img_size_post = np.shape(img)
    #TODO,(add data augment support)

    return name_str,img,bb_boxes

def target_label_generate(gta, anchor_box ,mc ,is_multi_layer = False, DEBUG = False):
    """
    target label generate function,
    input:
        gta: ground truth
        anchor_box: anchor box, default is anchor_box[0]
        is_multi_layer: later will support multi feature detector
    returmn:
        target label(fg/bg) array
        target bbox regression array(target bbox delta, bbox_in_weight, bbox)out_weight)
    """
    anchor_box = anchor_box[0]
    #default anchor_box[0] is H:40, W:60, because the rf is 16 in vgg(conv5_3)
    H, W = (40, 60)
    num_anchor_box_per_grid = mc.ANCHOR_PER_GRID
    gta = bbox_transform(gta, is_df = True)
    _gta = gta
    gta = bbox2cxcy(gta)
    #target_label = np.zeros((H, W, num_anchor_box_per_grid), dtype=np.float32)
    #target_bbox_delta = np.zeros((H, W, num_anchor_box_per_grid, 4), dtype=np.float32)
    #bbox_in_w = np.zeros((H, W, num_anchor_box_per_grid, 4), dtype=np.float32)
    #bbox_out_w = np.zeros((H, W, num_anchor_box_per_grid, 4), dtype=np.float32)

    #is valid, only inside anchor box is valid
    img_size = (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT)
    #transfer center_x, center_y, w, h to xmin, ymin, xmax, ymax
    bbox_xy = bbox_transform(anchor_box, is_df = False)
    _allowed_border = 0
    inds_inside = np.where(
    (bbox_xy[:, 0] >= -_allowed_border) &
    (bbox_xy[:, 1] >= -_allowed_border) &
    (bbox_xy[:, 2] < img_size[0] + _allowed_border) &  # width
    (bbox_xy[:, 3] < img_size[1] + _allowed_border)    # height
    )[0]
    out_inside = np.where(
    (bbox_xy[:, 0] < -_allowed_border) &
    (bbox_xy[:, 1] < -_allowed_border) &
    (bbox_xy[:, 2] >= img_size[0] + _allowed_border) &  # width
    (bbox_xy[:, 3] >= img_size[1] + _allowed_border)    # height
    )[0]
    #if(DEBUG):
    #    print('the valid anchors have ',len(inds_inside))
    #valid_anchors
    valid_anchors = anchor_box[inds_inside]
    anchors = coord2box(valid_anchors)
    groundtruth = coord2box(gta)
    #print(anchors)
    num_of_anchors = len(anchors)
    num_of_gta = len(groundtruth)
    overlaps_table = np.zeros((num_of_anchors, num_of_gta))
    for i in range(num_of_anchors):
        for j in range(num_of_gta):
            overlaps_table[i,j] = utils.box_iou(anchors[i], groundtruth[j])
    #overlaps_table = utils.bbox_overlaps(anchors, groundtruth)
    if(DEBUG):
        print('the shape of overlaps table {0}'.format(overlaps_table.shape))
        print('the number of groundtruth is', len(groundtruth))
        print('the number of valid anchor box is', len(anchors))

    #pick the postive and negative samples referenced from overlaps table

    #argmax overlaps for each groundtruth
    gt_argmax_overlaps = overlaps_table.argmax(axis=0)
    if(DEBUG):
        print('the shape of gt_argmax_overlaps is ',gt_argmax_overlaps.shape)
        print('the value in gt_argmax_overlaps is ',gt_argmax_overlaps)

    argmax_overlaps = overlaps_table.argmax(axis = 1)
    if(DEBUG):
        print('the shape of argmax_overlaps is ', argmax_overlaps.shape)
        print('the value in argmax_overlaps is ', argmax_overlaps)

    #overlaps groundtruth
    gt_max_overlaps = overlaps_table[gt_argmax_overlaps,np.arange(overlaps_table.shape[1])]
    gt_argmax_overlaps = np.where(overlaps_table == gt_max_overlaps)[0]
    if(DEBUG):
        print('the shape of processed gt_argmax_overlaps is ', gt_argmax_overlaps.shape)
        print('the value in processed gt_argmax_overlaps is ', gt_argmax_overlaps)

    #used this to select postive/ negative/ no care samples
    max_overlaps = overlaps_table[np.arange(len(valid_anchors)), argmax_overlaps]
    if(DEBUG):
        print('the shape of max overlaps table is ', max_overlaps.shape)

    target_labels = pick_samples(max_overlaps, gt_argmax_overlaps, mc)
    #target_labels[out_inside] = -1
    if(DEBUG):
        num_pos_samples = len(np.where(target_labels == 1)[0])
        num_neg_samples = len(np.where(target_labels == 0)[0])
        print('the number of postive samples is ', num_pos_samples)
        print('the number os negative samples is ', num_neg_samples)
    #subsampling, default subsampling methods is random sample
    target_labels = subsampling(target_labels, mc)
    if(DEBUG):
        num_pos_samples = len(np.where(target_labels == 1)[0])
        num_neg_samples = len(np.where(target_labels == 0)[0])
        print('After subsampling, the number of postive samples is ', num_pos_samples)
        print('After subsampling, the number os negative samples is ', num_neg_samples)

    #target cls label
    #if(mc.cls = True):


    #bbox delta label
    target_delta, bbox_in_w, bbox_out_w = target_bbox(out_inside, valid_anchors, gta[argmax_overlaps,:], target_labels, mc)
    if(DEBUG):
        print('the shape of target_delta is ',target_delta.shape)
        print('the shape of bbox_in_w is ',bbox_in_w.shape)
        print('the shape of bbox_out_w is ',bbox_out_w.shape)
    #UNMAP TO original feature images
    total_anchors = num_anchor_box_per_grid * H * W
    labels = unmap2original(target_labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap2original(target_delta, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = unmap2original(bbox_in_w, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = unmap2original(bbox_out_w, total_anchors, inds_inside, fill=0)
    #labels = target_labels
    #bbox_targets = target_delta
    #bbox_inside_weights = bbox_in_w
    #bbox_outside_weights = bbox_out_w
    #bbox_targets[out_inside] = 0
    #bbox_inside_weights[out_inside] = 0
    #bbox_outside_weights[out_inside] = 0
    if(DEBUG):
        print('the shape of target labels is ', labels.shape)
        print('the shape of bbox_target is', bbox_targets.shape)
        print('the shape of bbox_in_w is', bbox_inside_weights.shape)
        print('the shape of bbox_out_w is', bbox_outside_weights.shape)
    #classification map generate
    target_anchor_box_labels = target_classification_label_generate(total_anchors, anchor_box, labels, _gta, gta, groundtruth, mc, DEBUG)
    target_anchor_box_labels = target_anchor_box_labels.reshape((mc.H, mc.W, mc.ANCHOR_PER_GRID, mc.CLASSES))
    #reshape
    #labels = labels.reshape((H, W, num_anchor_box_per_grid))
    #bbox_targets = bbox_targets.reshape((H, W, num_anchor_box_per_grid * 4))
    #bbox_inside_weights = bbox_inside_weights.reshape((H, W, num_anchor_box_per_grid * 4))
    #bbox_outside_weights = bbox_outside_weights.reshape((H, W, num_anchor_box_per_grid * 4))
    labels = labels.reshape((mc.H , mc.W , mc.ANCHOR_PER_GRID))
    rpn_labels = labels
    #print(rpn_labels.shape)

     # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((mc.H , mc.W , mc.ANCHOR_PER_GRID * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((mc.H , mc.W , mc.ANCHOR_PER_GRID * 4))
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((mc.H , mc.W , mc.ANCHOR_PER_GRID * 4))
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights
    if(DEBUG):
        print('overlaps_table', overlaps_table.shape)
        print('overlaps_table displayed', overlaps_table)
        num_gta = overlaps_table.shape[1]
        print('number of gta',num_gta)
        print('groundtruth display : ', gta[:, 4])

    target_cls_label = cls_mapping(overlaps_table, max_overlaps, 20, anchor_box, gta, target_labels, mc, DEBUG = False)
    #print(target_cls_label.shape)
    target_cls_label = unmap2original(target_cls_label, total_anchors , inds_inside, fill=-1)
    #print(target_cls_label.shape)
    target_cls_label = target_cls_label \
        .reshape((mc.H, mc.W, mc.ANCHOR_PER_GRID , 20))
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, _gta, target_cls_label#target_anchor_box_labels,

def cls_mapping(overlaps_table, max_overlaps, num_of_cls, anchor_box, gta, target_labels, mc, DEBUG = False):
    #used one hot vector to generate target classification labels
    label = utils.one_hot_encode(gta[:,4], num_of_cls = num_of_cls)
    if(DEBUG):print('one hot encode vector ', label)
    #fg/bg + one hot vector, so 1 + num_of_cls
    target_cls_label = np.zeros((len(max_overlaps), num_of_cls))
    if(DEBUG):print('the initialize shape of target classification labels :', target_cls_label.shape)
    postive_index = np.where(target_labels == 1)[0]
    if(DEBUG):print('postive index is', postive_index)
    #calculate iou for all postive anchor box and groundtruth
    ref_anchors = anchor_box[postive_index]
    if(DEBUG):print('the center_x, center_y, w and h parameters of referenced postive anchors', ref_anchors)
    cls_overlaps_table = np.zeros((len(postive_index), overlaps_table.shape[1]))
    cls_overlaps_table.fill(-1)
    if(DEBUG):print('the shape of initialized classification overlaps_table ' , cls_overlaps_table.shape)
    anchor_coords = coord2box(ref_anchors)
    groundtruth = coord2box(gta)
    for i in range(len(postive_index)):
        for j in range(overlaps_table.shape[1]):
            cls_overlaps_table[i,j] = utils.box_iou(anchor_coords[i], groundtruth[j])
    if(DEBUG):print('the cls overlaps_table is', cls_overlaps_table)
    cls_idx = cls_overlaps_table.argmax(axis = 1)
    if(DEBUG):print('the label for each postive samples is', cls_idx)
    target_cls_label[postive_index,:] = label[cls_idx]
    if(DEBUG):print('the shape of target cls label is ', target_cls_label.shape)
    if(DEBUG):print('the target cls label', target_cls_label[postive_index])
    #for each negative samples fill all 0
    negative_index = np.where(target_labels == 0)[0]
    filled_value = np.zeros(num_of_cls)
    target_cls_label[negative_index,:] = filled_value
    return target_cls_label

def target_classification_label_generate(total_anchors, anchor_box, labels, _gta, gta, groundtruth, mc, DEBUG):
    postive_samples_index = np.where(labels == 1)[0]
    if(DEBUG):
        print('the postive sample index is', postive_samples_index)
        print('the postive anchor box coordinates is', anchor_box[postive_samples_index])
        print('the groundtruth coordinates and labels is', _gta)
    num_gta = len(_gta)
    num_postive_anchors = len(postive_samples_index)
    overlaps_table = np.zeros((num_postive_anchors, num_gta))
    valid_anchor_box = anchor_box[postive_samples_index]
    valid_anchor_box = coord2box(valid_anchor_box)
    for i in range(num_postive_anchors):
        for j in range(num_gta):
            overlaps_table[i,j] = utils.box_iou(valid_anchor_box[i], groundtruth[j])
    valid_argmax_overlaps = overlaps_table.argmax(axis = 1)
    if(DEBUG):
        print('display of overlaps_table is', overlaps_table)
        print('the valid argmax overlaps for each valid anchor box is ', valid_argmax_overlaps)
        print('shape of valid anchor box is', num_postive_anchors)
        print('shape of valid argmax overlaps', valid_argmax_overlaps.shape)
    target_anchor_box_labels = np.zeros((total_anchors, mc.CLASSES))
    target_anchor_box_labels.fill(-1)
    target_anchor_box_labels[postive_samples_index] = one_hot_encode(mc, _gta[valid_argmax_overlaps, 4])
    if(DEBUG):
        print('each valid anchor is', target_anchor_box_labels[postive_samples_index])
        print('verify value is', np.sum(target_anchor_box_labels[postive_samples_index]))
    negative_samples_index = np.where(labels == 0)[0]
    target_anchor_box_labels[negative_samples_index] = np.zeros((1, mc.CLASSES))
    if(DEBUG):
        print('anchor[negative_samples_index] is', target_anchor_box_labels[negative_samples_index])
        print('verify value is', np.sum(target_anchor_box_labels[negative_samples_index]))
        print('number of negative sample is', len(negative_samples_index))
    return target_anchor_box_labels

def one_hot_encode(mc, y):
    labels = []
    for i in range(len(y)):
        enc_label = np.zeros(mc.CLASSES)
        np.put(enc_label, y[i], 1)
        labels.append(enc_label)
    return np.array(labels)

def unmap2original(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def target_bbox(out_inside, anchor_box, gt_boxes, target_labels, mc):
    #create target bbox delta here
    target_delta = np.zeros((len(anchor_box), 4), dtype = np.float32)
    target_delta = utils.bbox_delta_convert(anchor_box, gt_boxes)
    target_delta[out_inside] = 0
    bbox_in_w = np.zeros((len(anchor_box), 4), dtype = np.float32)
    bbox_out_w = np.zeros((len(anchor_box), 4), dtype = np.float32)
    RPN_BBOX_INSIDE_WEIGHTS = mc.RPN_BBOX_INSIDE_WEIGHTS
    RPN_POSITIVE_WEIGHT = mc.RPN_POSITIVE_WEIGHT
    bbox_in_w[target_labels == 1] = np.array(RPN_BBOX_INSIDE_WEIGHTS)
    if RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(target_labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((RPN_POSITIVE_WEIGHT > 0) & (RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (RPN_POSITIVE_WEIGHT / np.sum(target_labels == 1))
        negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) / np.sum(target_labels == 0))
    bbox_out_w[target_labels == 1] = positive_weights
    bbox_out_w[target_labels == 0] = negative_weights
    return target_delta, bbox_in_w, bbox_out_w


def pick_samples(max_overlaps, gt_argmax_overlaps, mc):
    negative_threshould = mc.neg_max_overlaps
    postive_threshould = mc.pos_min_overlaps
    #initialize target labels here
    #like the original faster rcnn model, we set postive samples as 1, negative samples as 0, and -1 for no care
    target_labels = np.empty((len(max_overlaps), ), dtype=np.int32)
    #all target labels will fill -1 first
    target_labels.fill(-1)

    #negative samples, < negative_threshould
    target_labels[max_overlaps < negative_threshould] = 0

    #all gt argmax, the maximun overlaps of each groundtruth set as postive samples
    target_labels[gt_argmax_overlaps] = 1

    #postive samples, >= postive_threshould
    target_labels[max_overlaps >= postive_threshould] = 1
    return target_labels

def subsampling(target_labels, mc, sampling_methods = 'random'):
    """
    Random Sampling, Bootstracp and Mixture methods
    now, only support random sampling methods
    """
    fraction = mc.RPN_FRACTION
    batch_size = mc.RPN_BATCH_SIZE
    bal_num_of_pos = int(fraction * batch_size)
    fg = np.where(target_labels == 1)[0]
    if(len(fg) > bal_num_of_pos):
        #subsampling the postive samples
        disable_inds = npr.choice(fg, size=(len(fg) - bal_num_of_pos), replace=False)
        target_labels[disable_inds] = -1
    bal_num_of_neg = batch_size - np.sum(target_labels == 1)
    bg = np.where(target_labels == 0)[0]
    if(len(bg) > bal_num_of_neg):
        #subsampling the negative samples
        disable_inds = npr.choice(bg, size=(len(bg) - bal_num_of_neg), replace=False)
        target_labels[disable_inds] = -1
    return target_labels

def coord2box(bbox):
    boxes = []
    for i in range(len(bbox)):
        x = bbox[i,0]
        y = bbox[i,1]
        w = bbox[i,2]
        h = bbox[i,3]
        boxes.append(Box(x,y,w,h))
    return boxes

def bbox2cxcy(bb_boxes):
    gta = np.zeros((len(bb_boxes), 5))
    for i in range(len(bb_boxes)):
        gta[i,0] = bb_boxes[i,0] + (bb_boxes[i,2] - bb_boxes[i,0]) / 2
        gta[i,1] = bb_boxes[i,1] + (bb_boxes[i,3] - bb_boxes[i,1]) / 2
        gta[i,2] = (bb_boxes[i,2] - bb_boxes[i,0])
        gta[i,3] = (bb_boxes[i,3] - bb_boxes[i,1])
        gta[i,4] = bb_boxes[i,4]
    return gta

def bbox_transform(bb_boxes, is_df = True):
    """
    convert the x_center, y_center, w, h to xmin, ymin, xmax, ymax type
    """
    gta = np.zeros((len(bb_boxes), 5))
    #print(gta.shape)
    if(is_df):
        for i in range(len(bb_boxes)):
            """
            gta index:
            0: xmin -> x_center - (w / 2.)
            1: ymin -> y_center - (h / 2.)
            2: xmax -> x_center + (w / 2.)
            3: ymax -> y_center + (h / 2.)
            """
            gta[i,0] = bb_boxes.iloc[i]['x_center'] - (bb_boxes.iloc[i]['w'] / 2.)
            gta[i,1] = bb_boxes.iloc[i]['y_center'] - (bb_boxes.iloc[i]['h'] / 2.)
            gta[i,2] = bb_boxes.iloc[i]['x_center'] + (bb_boxes.iloc[i]['w'] / 2.)
            gta[i,3] = bb_boxes.iloc[i]['y_center'] + (bb_boxes.iloc[i]['h'] / 2.)
            gta[i,4] = bb_boxes.iloc[i]['label']
    else:
        for i in range(len(bb_boxes)):
            cx = bb_boxes[i,0]
            cy = bb_boxes[i,1]
            w = bb_boxes[i,2]
            h = bb_boxes[i,3]
            gta[i,0] = cx - (w / 2.)
            gta[i,1] = cy - (h / 2.)
            gta[i,2] = cx + (w / 2.)
            gta[i,3] = cy + (h / 2.)
            # gta[i,4] = bb_boxes[i,4]

    return gta#data
