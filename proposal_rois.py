import numpy as np
import tensorflow as tf
import pandas
import utils
import config
import batch_generate

def proposal_layer(rpn_bbox_cls_prob, rpn_bbox_pred, mc):
    anchors = mc.ANCHOR_BOX
    TOP_N_DETECTION = mc.TOP_N_DETECTION
    NMS_THRESH = mc.NMS_THRESH
    return tf.reshape(tf.py_func(proposal_rois_nms,[rpn_bbox_cls_prob, rpn_bbox_pred, anchors, TOP_N_DETECTION, NMS_THRESH], [tf.float32]), [-1, 5])
    #return tf.py_func(proposal_rois_nms,[rpn_bbox_cls_prob, rpn_bbox_pred, anchors, TOP_N_DETECTION, NMS_THRESH], [tf.int64])
def proposal_rois_nms(rpn_bbox_cls_prob, rpn_bbox_pred, anchors, TOP_N_DETECTION, NMS_THRESH):
    feat_stride = 16
    rpn_bbox_cls_prob = np.reshape(rpn_bbox_cls_prob, [-1,2])[:,1]
    rpn_bbox_pred = np.reshape(rpn_bbox_pred,[-1,4])
    #convert to xmin, ymin, xmax, ymax
    box_delta = np.reshape(rpn_bbox_pred,[21600,4])
    rpn_bbox_coords = utils.bbox_delta_convert_inv(anchors, box_delta)
    #clip to valid size
    bbox = clip_boxes(rpn_bbox_coords, [640, 960])
    #convert to cx, cy, w, h
    bbox_cxcy = batch_generate.bbox_transform(bbox, is_df = False)
    #nms
    box_nms, probs_nms, keep = utils.non_max_suppression_fast(bbox, rpn_bbox_cls_prob, TOP_N_DETECTION, overlap_thresh=NMS_THRESH)
    batch_inds = np.zeros((box_nms.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, box_nms.astype(np.float32, copy=False)))
    return blob

def proposal_target_layer(blob, gt_boxes, mc):
    num_cls = mc.CLASSES + 1
    argmax_overlaps = tf.py_func(proposaled_target_label_generate, [blob, gt_boxes, num_cls], [tf.int64])
    return argmax_overlaps

def proposaled_target_label_generate(proposal_rois, gt_boxes, num_cls):
    #convert to cx, cy, w, h
    proposal_rois = proposal_rois[:, 1:]
    #bbox = batch_generate.bbox_transform(proposal_rois, is_df = False)
    #get groundtruth
    gta = gt_boxes
    #prepare to calculate ious, initialize overlaps table
    bbox = batch_generate.coord2box(proposal_rois)
    gta = batch_generate.coord2box(gta[:,0:4])
    num_bbox = len(bbox)
    num_gta = len(gta)
    overlaps_table = np.zeros((num_bbox, num_gta))
    #calculate iou between proposaled rois and groundtruth
    for i in range(num_bbox):
        for j in range(num_gta):
            overlaps_table[i,j] = utils.box_iou(bbox[i], gta[j])
    #iou table
    argmax_overlaps = overlaps_table.argmax(axis = 1)
    #generate target classification label
    #take labels first
    labels = gt_boxes[argmax_overlaps, 4]
    #applied one hot vector
    
    #product target regression label(bbox delta)

    #return them back

    return np.asarray(labels, dtype = np.int64)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
