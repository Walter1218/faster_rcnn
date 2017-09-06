import numpy as np
import tensorflow as tf
import utils
import config
import batch_generate

def roi_proposal(rpn_cls_prob_reshape, rpn_bbox_pred, H, W, ANCHOR_PER_GRID, ANCHOR_BOX, TOP_N_DETECTION, NMS_THRESH, IM_H, IM_W):
    """
    clip the predict results fron rpn output
    appply nms
    proposal topN results as final layer output, no backward operation need here
    """
    #print(rpn_cls_prob_reshape.shape)
    box_probs = tf.reshape(rpn_cls_prob_reshape, [-1,2])[:,1]#np.reshape(rpn_cls_prob_reshape,[-1,2])[:,1]
    box_delta = tf.reshape(rpn_bbox_pred,[-1,4])
    #print(box_delta.shape)
    #anchor_box = ANCHOR_BOX

    pred_box_xyxy = tf.py_func(utils.bbox_delta_convert_inv,[ANCHOR_BOX, box_delta],[tf.float32])#utils.bbox_delta_convert_inv(ANCHOR_BOX, box_delta)
    #box_nms, probs_nms, pick = utils.non_max_suppression_fast(pred_box_xyxy, box_probs, TOP_N_DETECTION, overlap_thresh=NMS_THRESH)
    #box_nms, probs_nms, pick = tf.py_func(utils.non_max_suppression_fast,[pred_box_xyxy, box_probs, TOP_N_DETECTION, NMS_THRESH],[tf.float32, tf.float32, tf.int32])
    pred_box_xyxy = pred_box_xyxy[0]#tf.expand_dims(pred_box_xyxy, axis = 0)
    selected_idx = tf.image.non_max_suppression(pred_box_xyxy, box_probs, max_output_size = TOP_N_DETECTION, iou_threshold = NMS_THRESH)
    box_nms = tf.gather(box_delta, selected_idx)
    probs_nms = tf.gather(box_probs, selected_idx)
    #box_nms = box_nms[probs_nms>0.90]
    box = box_nms
    #box = batch_generate.bbox2cxcy(box)
    #clip box
    proposal_region = box#tf.py_func(clip_box, [box, IM_H, IM_W], [tf.int32])
    #proposal_region = clip_box(box, IM_H, IM_W)
    #print('the shape of proposaled region is ',proposal_region.shape)
    #print('the proposaled region value is ', proposal_region)
    #batch_inds = np.zeros((proposal_region.shape[0], 1), dtype=np.float32)
    #blob = np.hstack((batch_inds, proposal_region.astype(np.float32, copy=False)))
    #blob, probs_nms = tf.py_func(blob_generate, [proposal_region], [tf.float32, tf.float32])
    return box, probs_nms, selected_idx#blob, probs_nms

def blob_generate(proposal_region):
    batch_inds = np.zeros((proposal_region.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposal_region.astype(np.float32, copy=False)))
    return blob, probs_nms

def clip_box(box, IM_H, IM_W):
    h = IM_H
    w = IM_W
    # x1 >= 0
    box[:, 0::4] = np.maximum(np.minimum(box[:, 0::4], w - 1), 0)
    # y1 >= 0
    box[:, 1::4] = np.maximum(np.minimum(box[:, 1::4], h - 1), 0)
    # x2 < im_shape[1]
    box[:, 2::4] = np.maximum(np.minimum(box[:, 2::4], w - 1), 0)
    # y2 < im_shape[0]
    box[:, 3::4] = np.maximum(np.minimum(box[:, 3::4], h - 1), 0)
    return box

def proposaled_target_label_generate(rois, rpn_scores, gt_boxes, num_cls, selected_idx, mc):
    #calculate iou
    negative_threshould = mc.neg_max_overlaps
    postive_threshould = mc.pos_min_overlaps
    #groundtruth = tf.py_func(batch_generate.coord2box, [gt_boxes[:,4]], [tf.float32])#batch_generate.coord2box(gt_boxes[:,:4])
    #convert delta to center x, center y , w and h structure
    ANCHOR_BOX = mc.ANCHOR_BOX
    anchor_box = tf.gather(ANCHOR_BOX, selected_idx)
    pred_box_xyxy = tf.py_func(utils.bbox_delta_convert_inv,[anchor_box, rois],[tf.float32])
    keep_rois = tf.py_func(clip_box, [pred_box_xyxy, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH], [tf.float32])
    #bbox_transform
    #keep_rois = keep_rois[0]
    pred_box_xywh, pred_box_xyxy = tf.py_func(bbox2cxcy, [keep_rois], [tf.float32, tf.float32])
    proposed_roi_iou_table, proposed_rois = tf.py_func(iou, [pred_box_xywh, gt_boxes],[tf.float32, tf.float32])
    #proposed_rois = tf.py_func(batch_generate.coord2box, [pred_box_xywh],[tf.float32])#batch_generate.coord2box(rois[:,:4])
    return proposed_roi_iou_table, pred_box_xyxy/16

def iou(proposed_rois, groundtruth):
    gta = batch_generate.coord2box(proposed_rois[0])
    rois = batch_generate.coord2box(groundtruth[:,0:4])
    num_gta = len(gta)
    num_rois = len(rois)
    overlaps_table = np.zeros((num_rois, num_gta), dtype = np.float32)
    for i in range(num_rois):
        for j in range(num_gta):
            overlaps_table[i,j] = utils.box_iou(rois[i], gta[j])
    gt_assignment = overlaps_table.argmax(axis=0)
    max_overlaps = overlaps_table.argmax(axis=1)
    labels = groundtruth[gt_assignment, 4]
    fg_inds = np.where(max_overlaps >= 0.5)[0]
    bg_inds = np.where(max_overlaps < 0.5)[0]
    keep_inds = np.append(fg_inds, bg_inds)
    labels = labels[keep_inds] + 1
    labels[bg_inds] = 0
    #target labels(fg/bg)
    one_hot_encode_vector = utils.one_hot_encode(labels, 20)
    #target delta generate(only for pos   )
    #keep_inds = [fg_inds, bg_inds]
    return one_hot_encode_vector, proposed_rois#rois[keep_inds,:]


def bbox2cxcy(bb_boxes):
    bb_boxes = np.reshape(bb_boxes, [-1,4])
    gta = np.zeros((len(bb_boxes), 4), dtype = np.float32)
    #for i in range(len(bb_boxes)):
    gta[:,0] = bb_boxes[:,0] + (bb_boxes[:,2] - bb_boxes[:,0]) / 2
    gta[:,1] = bb_boxes[:,1] + (bb_boxes[:,3] - bb_boxes[:,1]) / 2
    gta[:,2] = (bb_boxes[:,2] - bb_boxes[:,0])
    gta[:,3] = (bb_boxes[:,3] - bb_boxes[:,1])
    #gta[i,4] = bb_boxes[i,4]
    return gta, bb_boxes

def proposaled_target_layer(rois, rpn_scores, gt_boxes, target_cls_label, num_cls):
    all_rois = rois
    all_scores = rpn_scores
    _num_classes = num_cls
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
    all_scores = np.vstack((all_scores, zeros))
    num_images = 1
    rois_per_image = 256 / num_images
    fg_rois_per_image = np.round(0.5 * rois_per_image)
    labels, rois, bbox_targets, bbox_inside_weights = sampling_methods(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes)
    rois = rois.reshape(-1, 5)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes*4)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

def sampling_methods(mc, all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes):
    negative_threshould = mc.neg_max_overlaps
    postive_threshould = mc.pos_min_overlaps
    groundtruth = batch_generate.coord2box(gt_boxes[:,:4])
    proposed_rois = batch_generate.coord2box(all_rois[:,:4])
    num_of_groundtruth = len(groundtruth)
    num_of_rois = len(proposed_rois)
    overlaps_table = np.zeros((num_of_rois, num_of_groundtruth))
    for i in range(num_of_rois):
        for j in range(num_of_groundtruth):
            overlaps_table[i,j] = utils.box_iou(proposed_rois[i], groundtruth[j])
    gt_assignment = overlaps_table.argmax(axis=1)
    max_overlaps = overlaps_table.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    fg_inds = np.where(max_overlaps >= postive_threshould)[0]
    bg_inds = np.where(max_overlaps < negative_threshould)[0]
    #fg
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    #bg
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    bbox_targets, bbox_inside_weights = target_delta(bbox_target_data, _num_classes)
    return labels, rois, bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    targets = utils.bbox_delta_convert(ex_rois, gt_rois)

    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def target_delta(target_data, num_classes, mc):
    clss = target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = mc.RPN_BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
