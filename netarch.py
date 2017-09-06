from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
import utils
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from graphical_model import graphical_model
import tensorflow.contrib.slim as slim
from roi_proposal import roi_proposal, proposaled_target_layer,proposaled_target_label_generate
import proposal_rois

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_device(name, shape, initializer, trainable=True):
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var

class ResNet50(graphical_model):
    def __init__(self, mc, gpu_id):
        with tf.device('/gpu:{}'.format(gpu_id)):
            graphical_model.__init__(self, mc)
            self.forward_graph()
            #self.logits_node()
            self.loss_func()
            self.opt_graph()

    def forward_graph(self):
        mc = self.mc
        if mc.LOAD_PRETRAINED_MODEL:
            self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

        conv1 = self.conv_bn_layer(self.image_input, 'conv1', 'bn_conv1', 'scale_conv1', filters=64,size=7, stride=2, freeze=True, conv_with_bias=True)
        pool1 = self.pooling_layer('pool1', conv1, size=3, stride=2, padding='VALID')

        with tf.variable_scope('conv2_x') as scope:
            with tf.variable_scope('res2a'):
                branch1 = self.conv_bn_layer(pool1, 'res2a_branch1', 'bn2a_branch1', 'scale2a_branch1',filters=256, size=1, stride=1, freeze=True, relu=False)
                branch2 = self.residual_branch(pool1, layer_name='2a', in_filters=64, out_filters=256,down_sample=False, freeze=True)
                res2a = tf.nn.relu(branch1+branch2, 'relu')
            with tf.variable_scope('res2b'):
                branch2 = self.residual_branch(res2a, layer_name='2b', in_filters=64, out_filters=256,down_sample=False, freeze=True)
                res2b = tf.nn.relu(res2a+branch2, 'relu')
            with tf.variable_scope('res2c'):
                branch2 = self.residual_branch(res2b, layer_name='2c', in_filters=64, out_filters=256,down_sample=False, freeze=True)
                res2c = tf.nn.relu(res2b+branch2, 'relu')

        with tf.variable_scope('conv3_x') as scope:
            with tf.variable_scope('res3a'):
                branch1 = self.conv_bn_layer(res2c, 'res3a_branch1', 'bn3a_branch1', 'scale3a_branch1',filters=512, size=1, stride=2, freeze=True, relu=False)
                branch2 = self.residual_branch(res2c, layer_name='3a', in_filters=128, out_filters=512,down_sample=True, freeze=True)
                res3a = tf.nn.relu(branch1+branch2, 'relu')
            with tf.variable_scope('res3b'):
                branch2 = self.residual_branch(res3a, layer_name='3b', in_filters=128, out_filters=512,down_sample=False, freeze=True)
                res3b = tf.nn.relu(res3a+branch2, 'relu')
            with tf.variable_scope('res3c'):
                branch2 = self.residual_branch(res3b, layer_name='3c', in_filters=128, out_filters=512,down_sample=False, freeze=True)
                res3c = tf.nn.relu(res3b+branch2, 'relu')
            with tf.variable_scope('res3d'):
                branch2 = self.residual_branch(res3c, layer_name='3d', in_filters=128, out_filters=512,down_sample=False, freeze=True)
                res3d = tf.nn.relu(res3c+branch2, 'relu')

        with tf.variable_scope('conv4_x') as scope:
            with tf.variable_scope('res4a'):
                branch1 = self.conv_bn_layer(res3d, 'res4a_branch1', 'bn4a_branch1', 'scale4a_branch1',filters=1024, size=1, stride=2, relu=False)
                branch2 = self.residual_branch(res3d, layer_name='4a', in_filters=256, out_filters=1024,down_sample=True)
                res4a = tf.nn.relu(branch1+branch2, 'relu')
            with tf.variable_scope('res4b'):
                branch2 = self.residual_branch(res4a, layer_name='4b', in_filters=256, out_filters=1024,down_sample=False)
                res4b = tf.nn.relu(res4a+branch2, 'relu')
            with tf.variable_scope('res4c'):
                branch2 = self.residual_branch(res4b, layer_name='4c', in_filters=256, out_filters=1024,down_sample=False)
                res4c = tf.nn.relu(res4b+branch2, 'relu')
            with tf.variable_scope('res4d'):
                branch2 = self.residual_branch(res4c, layer_name='4d', in_filters=256, out_filters=1024,down_sample=False)
                res4d = tf.nn.relu(res4c+branch2, 'relu')
            with tf.variable_scope('res4e'):
                branch2 = self.residual_branch(res4d, layer_name='4e', in_filters=256, out_filters=1024,down_sample=False)
                res4e = tf.nn.relu(res4d+branch2, 'relu')
            with tf.variable_scope('res4f'):
                branch2 = self.residual_branch(res4e, layer_name='4f', in_filters=256, out_filters=1024,down_sample=False)
                res4f = tf.nn.relu(res4e+branch2, 'relu')

        dropout4 = tf.nn.dropout(res4f, self.keep_prob, name='drop4')

        #RPN Layer
        #3*3 conv layer
        rpn = self.conv_layer('rpn', dropout4, filters=512, size=3, stride=1, padding='SAME', xavier=False, relu=False, mean = 0.0, stddev=0.001)
        rpn_cls_score = self.conv_layer('rpn_cls_score', rpn, filters = mc.ANCHOR_PER_GRID * 2, size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        rpn_bbox_pred = self.conv_layer('rpn_bbox_pred', rpn, filters = mc.ANCHOR_PER_GRID * 4, size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        #cls_pred = self.conv_layer('cls_pred', rpn, filters = 2* mc.ANCHOR_PER_GRID , size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        rpn_cls_score_reshape = self.spatial_reshape_layer(rpn_cls_score, 2, name = 'rpn_cls_score_reshape')
        rpn_cls_prob = self.spatial_softmax(rpn_cls_score_reshape, name='rpn_cls_prob')
        rpn_cls_prob_reshape = self.spatial_reshape_layer(rpn_cls_prob , mc.ANCHOR_PER_GRID * 2, name = 'rpn_cls_prob_reshape')

        #cls_score_reshape = self.spatial_reshape_layer(cls_pred, 2, name = 'cls_score_reshape')
        #cls_prob = self.spatial_softmax(cls_score_reshape, name='cls_prob')
        #cls_prob_reshape = self.spatial_reshape_layer(cls_prob , mc.ANCHOR_PER_GRID * 2 , name = 'cls_prob_reshape')

        #proposal_nms here
        blobs = proposal_rois.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, mc)
        #rois, rpn_scores, selected_idx = roi_proposal(rpn_cls_prob_reshape, rpn_bbox_pred, mc.H, mc.W, mc.ANCHOR_PER_GRID, mc.ANCHOR_BOX, mc.TOP_N_DETECTION, mc.NMS_THRESH, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH)
        #rois, rpn_scores = self.proposal_nms_layer(mc, rpn_cls_score_reshape, rpn_bbox_pred)
        #select postive/negative samples from proposaled nms bboxes
        #proposed_roi_table, proposed_rois = proposaled_target_label_generate(rois, rpn_scores, self.gt_boxes, 20,  selected_idx ,mc)
        #rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self.proposaled_target_layer(mc, rois, rpn_scores, self.gt_boxes, self.target_cls_label)
        argmax_overlaps = proposal_rois.proposal_target_layer(blobs, self.gt_boxes, mc)
        #self._anchor_targets['bbox_targets'] = bbox_targets
        #self._anchor_targets['labels'] = labels
        #self._anchor_targets['bbox_inside_weights'] = bbox_inside_weights
        #self._anchor_targets['bbox_outside_weights'] = bbox_outside_weights
        pool5 = self._crop_pool_layer(res4f, blobs, 'roi-pool')
        #batch_ids = tf.squeeze(tf.slice(proposed_rois, [0, 0], [-1, 1], name="batch_id"), [1])
        #fc6 = self.conv_layer('fc6_', pool5, filters=4096, size=1, stride=1, padding='SAME', xavier=False, relu=True, mean = 0.0, stddev=0.001)
        #fc7 = self.conv_layer('fc7_', fc6, filters=4096, size=1, stride=1, padding='SAME', xavier=False, relu=True, mean = 0.0, stddev=0.001)
        #cls_score = self.conv_layer('fc7', fc7, filters=mc.CLASSES, size=1, stride=1, padding='SAME', xavier=False, relu=True, mean = 0.0, stddev=0.001)
        #cls_prob = self._softmax_layer(cls_score, "cls_prob")
        #cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        #bbox_pred = self.conv_layer('bbox', fc7, filters = mc.CLASSES * 4, size=1, stride=1, padding='SAME', xavier=False, relu=False, mean = 0.0, stddev=0.001)

        #end2end network, classification tasks here
        #cls_layer = self.conv_layer('cls_layer', rpn, filters = 512, size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        #cls_score = self.conv_layer('cls_score', cls_layer, filters = mc.ANCHOR_PER_GRID * mc.CLASSES, size = 1, stride = 1, padding = 'VALID', xavier = False, relu = True, mean = 0.0, stddev = 0.001)
        #cls_score_reshape = self.spatial_reshape_layer(cls_score, mc.CLASSES, name = 'cls_score_reshape')
        #cls_prob = self.spatial_softmax(cls_score_reshape, name = 'cls_prob')
        #cls_prob_reshape = self.spatial_reshape_layer(cls_prob, mc.ANCHOR_PER_GRID * mc.CLASSES, name = 'cls_prob_reshape')

        #self._predictions["cls_score_reshape"] = cls_score_reshape

        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        #self.pred_rois = rois
        #self.pred_proposed_rois = proposed_rois
        #self._predictions["cls_score"] = cls_score
        #self._predictions["cls_pred"] = cls_pred
        #self._predictions["cls_prob"] = cls_prob
        #self._predictions["bbox_pred"] = bbox_pred
        #################################################
        ########DEBUG PRINT##############################
        #self._predictions["rpn_rois"] = rois
        #self._predictions["rpn_scores"] = rpn_scores
        #self._predictions["cls_pred"] = cls_score_reshape
        #self.det_cls = cls_prob_reshape
        self.det_probs = rpn_cls_prob_reshape
        self.det_boxes = rpn_bbox_pred
        #batch_ids = tf.squeeze(proposed_rois, [1])
        self.det_proposaled_scores = argmax_overlaps

    def roi_pool(self, featureMaps,rois,im_dims):
        '''
        Regions of Interest (ROIs) from the Region Proposal Network (RPN) are
        formatted as:
        (image_id, x1, y1, x2, y2)

        Note: Since mini-batches are sampled from a single image, image_id = 0s
        '''
        with tf.variable_scope('roi_pool'):
            # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
            box_ind = tf.cast(rois[:,0],dtype=tf.int32)

            # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
            boxes = rois[:,1:]
            normalization = tf.cast(tf.stack([im_dims[:,1],im_dims[:,0],im_dims[:,1],im_dims[:,0]],axis=1),dtype=tf.float32)
            boxes = tf.div(boxes,normalization)
            boxes = tf.stack([boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2]],axis=1)  # y1, x1, y2, x2

            # ROI pool output size
            crop_size = tf.constant([14,14])

            # ROI pool
            pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind, crop_size=crop_size)

            # Max pool to (7x7)
            pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return pooledFeatures

    def _crop_pool_layer(self, bottom, rois, name, RESNET_MAX_POOL = False):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(1/16)
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(1/16)
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
            #bboxes = tf.stop_gradient(tf.concat(rois, 1))
        if RESNET_MAX_POOL:
            pre_pool_size = 7 * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],name="crops")
            crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
        else:
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [7, 7],name="crops")
        return crops

    def proposaled_target_layer(self, mc, rois, rpn_scores, gt_boxes, target_cls_label, name = 'proposaled_target_layer'):
        with tf.variable_scope(name) as scope:
            num_cls = mc.CLASSES
            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposaled_target_layer, [rois, rpn_scores, gt_boxes, target_cls_label, num_cls],[tf.float32, tf.float32,tf.float32,tf.float32,tf.float32])
            rois = tf.reshape(rois, [-1, 5], name='rois') # goes to roi_pooling
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels') # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets') # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

            self._predictions['rois'] = rois

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def proposal_nms_layer(self, mc, rpn_cls_prob_reshape, rpn_bbox_pred, name = 'proposal_nms_layer'):
        with tf.variable_scope(name) as scope:
            #print('proposal_nms_layer process')
            self.H = mc.H
            self.W = mc.W
            self.ANCHOR_PER_GRID = mc.ANCHOR_PER_GRID
            self.ANCHOR_BOX = mc.ANCHORS
            self.TOP_N_DETECTION = mc.TOP_N_DETECTION
            self.IM_H = mc.IMAGE_HEIGHT
            self.IM_W = mc.IMAGE_WIDTH
            self.NMS_THRESH = mc.NMS_THRESH
            rois, rpn_scores = tf.py_func(roi_proposal,[rpn_cls_prob_reshape, rpn_bbox_pred, self.H, self.W, self.ANCHOR_PER_GRID,
                                          self.ANCHOR_BOX, self.TOP_N_DETECTION, self.NMS_THRESH, self.IM_H, self.IM_W],
                                          [tf.float32, tf.float32], name = 'roi_proposal')
            rois.set_shape([mc.TOP_N_DETECTION, 5])
            rpn_scores.set_shape([mc.TOP_N_DETECTION, 1])
        return rois, rpn_scores

    def residual_branch(self, inputs, layer_name, in_filters, out_filters, down_sample=False, freeze=False):
        with tf.variable_scope('res'+layer_name+'_branch2'):
            stride = 2 if down_sample else 1
            output = self.conv_bn_layer(inputs,conv_param_name='res'+layer_name+'_branch2a',
                                         bn_param_name='bn'+layer_name+'_branch2a',scale_param_name='scale'+layer_name+'_branch2a',
                                         filters=in_filters, size=1, stride=stride, freeze=freeze)
            output = self.conv_bn_layer(output,conv_param_name='res'+layer_name+'_branch2b',
                                         bn_param_name='bn'+layer_name+'_branch2b',scale_param_name='scale'+layer_name+'_branch2b',
                                         filters=in_filters, size=3, stride=1, freeze=freeze)
            output = self.conv_bn_layer(output,conv_param_name='res'+layer_name+'_branch2c',
                                         bn_param_name='bn'+layer_name+'_branch2c',scale_param_name='scale'+layer_name+'_branch2c',
                                         filters=out_filters, size=1, stride=1, freeze=freeze, relu=False)
        return output

    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
        return fc

    def conv_bn_layer(self, inputs, conv_param_name, bn_param_name, scale_param_name, filters, size, stride, padding='SAME',
                      freeze=False, relu=True, conv_with_bias=False, stddev=0.001):
        mc = self.mc
        with tf.variable_scope(conv_param_name) as scope:
            channels = inputs.get_shape()[3]
            if mc.LOAD_PRETRAINED_MODEL:
                cw = self.caffemodel_weight
                #because the weights parameters stored in caffe model is different with tensorflow, so we need transpose to tf structure
                kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
                if conv_with_bias:
                    bias_val = cw[conv_param_name][1]
                mean_val   = cw[bn_param_name][0]
                var_val    = cw[bn_param_name][1]
                gamma_val  = cw[scale_param_name][0]
                beta_val   = cw[scale_param_name][1]
            else:
                kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
                if conv_with_bias:
                    bias_val = tf.constant_initializer(0.0)
                mean_val   = tf.constant_initializer(0.0)
                var_val    = tf.constant_initializer(1.0)
                gamma_val  = tf.constant_initializer(1.0)
                beta_val   = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
            self.model_params += [kernel]
            if conv_with_bias:
                biases = _variable_on_device('biases', [filters], bias_val,trainable=(not freeze))
                self.model_params += [biases]
            gamma = _variable_on_device('gamma', [filters], gamma_val,trainable=(not freeze))
            beta  = _variable_on_device('beta', [filters], beta_val,trainable=(not freeze))
            mean  = _variable_on_device('mean', [filters], mean_val, trainable=False)
            var   = _variable_on_device('var', [filters], var_val, trainable=False)
            self.model_params += [gamma, beta, mean, var]

            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding,name='convolution')
            if conv_with_bias:
                conv = tf.nn.bias_add(conv, biases, name='bias_add')

            conv = tf.nn.batch_normalization(conv, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')

            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def pooling_layer(self, layer_name, inputs, size, stride, padding='SAME'):
        with tf.variable_scope(layer_name) as scope:
            out =  tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],padding=padding)
        return out

    def conv_layer(self, layer_name, inputs, filters, size, stride, padding='SAME',freeze=False, xavier=False, relu=True, mean = 0.0, stddev=0.001):
        mc = self.mc
        use_pretrained_param = False
        if mc.LOAD_PRETRAINED_MODEL:
            cw = self.caffemodel_weight
            if layer_name in cw:
                kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
                bias_val = cw[layer_name][1]
                # check the shape
                if (kernel_val.shape == (size, size, inputs.get_shape().as_list()[-1], filters)) and (bias_val.shape == (filters, )):
                    use_pretrained_param = True
                else:
                    print ('Shape of the pretrained parameter of {} does not match, use randomly initialized parameter'.format(layer_name))
            else:
                print ('Cannot find {} in the pretrained model. Use randomly initialized parameters'.format(layer_name))

        if mc.DEBUG_MODE:
            print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]
            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            if use_pretrained_param:
                if mc.DEBUG_MODE:
                    print ('Using pretrained model for {}'.format(layer_name))
                kernel_init = tf.constant(kernel_val , dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(mean = 0.0, stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

            biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
            self.model_params += [kernel, biases]

            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding,name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

        return out
