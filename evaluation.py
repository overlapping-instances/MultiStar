import numpy as np
import pickle
from experiments.dev import sampling
from scipy.optimize import linear_sum_assignment
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def predict_segmentations(dataset, model, device, iou_threshold, min_objprob, num_proposals):
    """Predict the segmentation for all images in the dataset.

    Parameters:
    dataset -- dataset.Dataset instance
    model -- multitaskmodel.Multitaskmodel instance
    device -- cuda device
    iou_threshold -- intersection over union threshold
    min_objprob -- minimum required object probability to sample
    num_proposals -- number of proposals to generate

    Returns: list of arrays of shapes (num_pred_cells, height, width) with predicted segmentations
    """

    model = model.to(device)

    pred_segmentation_stack = []
    labels_stack = []
    num_images = dataset.images.shape[0]

    for i in tqdm(range(num_images)):
        # get and normalize image, convert to tensor
        img = dataset.images[i]
        img = (img - dataset.min_max_value[0]) / (dataset.min_max_value[1] - dataset.min_max_value[0])
        img = torch.from_numpy(img).to(device)

        # predict features
        pred_overlap, pred_stardist, pred_objprob = model(img.unsqueeze(0))

        pred_overlap = torch.sigmoid(pred_overlap).cpu().detach().numpy()
        pred_objprob = torch.sigmoid(pred_objprob).cpu().detach().numpy()
        pred_stardist = pred_stardist.cpu().detach().numpy()

        # find segmentation with non-maximum suppression
        pred_segmentation, _ = sampling.nms(pred_overlap[0, 0], pred_stardist[0], pred_objprob[0, 0], num_proposals, iou_threshold, min_objprob)
        
        pred_segmentation_stack.append(pred_segmentation)
        labels_stack.append(dataset.labels[i])

    return pred_segmentation_stack, labels_stack

def optimal_assignment_dice(pred_segmentation, labels):
    """Find the optimal assignment of predictions and labels wrt dice coefficient with the Hungarian algorithm.

    Parameters:
    pred_segmentation -- array of shape (num_pred_cells, H, W) with predicted segmentation masks
    labels -- array of shape (num_gt_cells, H, W) with gt segmentation masks

    Returns: list with detected gt cell indies, list with matched predicted cell indices,
    list with corresponding dice coefficients
    """
    
    assert pred_segmentation.shape[1:] == labels.shape[1:]

    num_cells_pred = pred_segmentation.shape[0]
    num_cells_gt = labels.shape[0]

    # table for the dice coefficients of all predicted - gt combinations
    coefficients = np.zeros((num_cells_gt, num_cells_pred))
    
    # iterate over ground truth objects
    for i in range(num_cells_gt):
        # make as many copies of ground truth label i as there are predicted cells
        gt_cell = np.repeat(labels[i][np.newaxis, :, :], num_cells_pred, axis=0)

        # compute the dice coefficient between every predicted cell and the ground truth cell i
        dc = (
            2 * np.count_nonzero(np.logical_and(gt_cell, pred_segmentation), axis=(1,2)) /
            (np.count_nonzero(gt_cell, axis=(1, 2)) + np.count_nonzero(pred_segmentation, axis=(1,2)))
            )

        # set i-th row of coefficient table to the determined coefficients
        coefficients[i, :] = dc

    # find the optimal assignment between gt cells and predicted cells to maximize the sum of dice coefficients
    gt_order, pred_order = linear_sum_assignment(coefficients, maximize=False)
    
    # pick the dice coefficients of the optimal assignment
    dice = coefficients[gt_order, pred_order]
    
    return gt_order, pred_order, dice

def optimal_assignment_iou(pred_segmentation, labels):
    """Find the optimal assignment of predictions and labels wrt IoU with the Hungarian algorithm.

    Parameters:
    pred_segmentation -- array of shape (num_pred_cells, H, W) with predicted segmentation masks
    labels -- array of shape (num_gt_cells, H, W) with gt segmentation masks

    Returns: list with detected gt cell indies, list with matched predicted cell indices,
    list with corresponding iou scores
    """
    
    assert pred_segmentation.shape[1:] == labels.shape[1:]

    num_cells_pred = pred_segmentation.shape[0]
    num_cells_gt = labels.shape[0]

    # table for the iou scores of all predicted - gt combinations
    scores = np.zeros((num_cells_gt, num_cells_pred))
    
    # iterate over ground truth objects
    for i in range(num_cells_gt):
        # make as many copies of ground truth label i as there are predicted cells
        gt_cell = np.repeat(labels[i][np.newaxis, :, :], num_cells_pred, axis=0)

        # compute the iou score between every predicted cell and the ground truth cell i
        iou = (
            np.count_nonzero(np.logical_and(gt_cell, pred_segmentation), axis=(1,2)) /
            np.count_nonzero(gt_cell + pred_segmentation, axis=(1,2))
            )

        # set i-th row of scores table to the determined iou scores
        scores[i, :] = iou

    # find the optimal assignment between gt cells and predicted cells to maximize the sum of dice coefficients
    gt_order, pred_order = linear_sum_assignment(scores, maximize=True)
    
    # pick the dice coefficients of the optimal assignment
    iou_scores = scores[gt_order, pred_order]
    
    return gt_order, pred_order, iou_scores
         
def hungry_assignment_dice(pred_segmentation, labels):
    """Compute the "hungry" assignment of predicted objects to gt objects using dice coefficient.
    
    This means that in the order of gt objects, every object gets its best match (irrespective of
    whether this prediction better matches an object that comes later)
    Parameters:
    pred_segmentation -- array of shape (num_pred_cells, H, W) with predicted segmentation masks
    labels -- array of shape (num_gt_cells, H, W) with gt segmentation masks

    Returns: list with detected gt cell indies, list with matched predicted cell indices,
    list with corresponding dice coefficients
    """
    
    assert pred_segmentation.shape[1:] == labels.shape[1:]

    num_gt_objects = labels.shape[0]
    num_pred_objects = pred_segmentation.shape[0]

    gt_order = []
    pred_order = []
    dice_coefficients = []

    # iterate over all ground truth objects
    for i in range(num_gt_objects):
        max_dc = 0
        max_index = -1
        # iterate over all predicted objects
        for j in range(num_pred_objects):
            # skip this object if already matched to gt object
            if j in pred_order:
                continue
            # compute dice coefficient
            dc = (
            2 * np.count_nonzero(np.logical_and(labels[i], pred_segmentation[j])) /
            (np.count_nonzero(labels[i]) + np.count_nonzero(pred_segmentation[j]))
            )
            # check if best matching so far
            if dc > max_dc:
                max_dc = dc
                max_index = j
        # if a predicted object could be matched, add to list
        if max_dc != 0:
            gt_order.append(i)
            pred_order.append(max_index)
            dice_coefficients.append(max_dc)

    return np.array(gt_order), np.array(pred_order), np.array(dice_coefficients)

def hungry_assignment_iou(pred_segmentation, labels):
    """Compute the "hungry" assignment of predicted objects to gt objects using IoU.
    Parameters:
    pred_segmentation -- array of shape (num_pred_cells, H, W) with predicted segmentation masks
    labels -- array of shape (num_gt_cells, H, W) with gt segmentation masks

    Returns: list with detected gt cell indies, list with matched predicted cell indices,
    list with corresponding IoU scores
    """

    assert pred_segmentation.shape[1:] == labels.shape[1:]

    num_gt_objects = labels.shape[0]
    num_pred_objects = pred_segmentation.shape[0]

    gt_order = []
    pred_order = []
    iou_scores = []

    # iterate over all ground truth objects
    for i in range(num_gt_objects):
        max_iou = 0
        max_index = -1
        # iterate over all predicted objects
        for j in range(num_pred_objects):
            # skip this object if already matched to gt object
            if j in pred_order:
                continue
            # compute iou score
            iou = (
                np.count_nonzero(np.logical_and(labels[i], pred_segmentation[j])) /
                np.count_nonzero(np.logical_or(labels[i], pred_segmentation[j]))
                )
            # check if best matching so far
            if iou > max_iou:
                max_iou = iou
                max_index = j
        # if a predicted object could be matched
        if max_iou != 0:
            gt_order.append(i)
            pred_order.append(max_index)
            iou_scores.append(max_iou)

    return np.array(gt_order), np.array(pred_order), np.array(iou_scores)

def prediction_grid(dataset, model, device, nms_thresholds, min_objprobs, num_proposals):
    """Generate predictions on the complete dataset for different parameters.
    
    Parameters:
    dataset -- dataset.Dataset instance
    model -- multitaskmodel.MultitaskModel instance
    device -- cuda device
    nms_thresholds -- list with IoU thresholds for non-maximum suppression
    min_objprobs -- list with minimum object probabilities for sampling proposals
    num_proposals -- number of proposals to generate per image
    
    Returns: 2d list with complete dataset segmentations for different values nms_thresholds and
    min_objprobs, labels
    """
    
    predictions = []
    
    for i, nms_threshold in enumerate(tqdm(nms_thresholds)):
        predictions.append([])
        for j, min_objprob in enumerate(tqdm(min_objprobs)):
            prediction, labels = predict_segmentations(dataset, model, device, nms_threshold, min_objprob, num_proposals)
            predictions[-1].append(prediction)
            
    return predictions, labels

def save_prediction(file, predictions, labels, nms_thresholds, min_objprobs, num_proposals):
    """Save prediction and parameters with pickle."""
    data = {"predictions":predictions, "labels":labels, "nms_thresholds":nms_thresholds, "min_objprobs":min_objprobs, "num_proposals":num_proposals}
    pickle.dump(data, open(file, 'wb'))

def get_precisions(pred_segmentation_stack, labels_stack, matching_thresholds):
    """Compute the average precision of detections on a number of images and average, defined as tp / (tp + fp + fn).

    tp: number of matches (gt object with predicted object such that IoU > threshold)
    fp: number of unmatched predicted objects
    fn: number of unmatched gt objects

    Parameters:
    pred_segmentation_stack -- list of arrays, with each array having shape (num_pred_cells, height, width)
    labels_stack -- list of arrays, with each array having shape (num_gt_cells, height, width)
    matching_thresholds -- list of thresholds above which a two objects are considered a match

    Returns: average precisions, standard deviation
    """

    assert len(pred_segmentation_stack) == len(labels_stack)
    
    num_images = len(labels_stack)
    num_thresholds = len(matching_thresholds)
    precisions = np.zeros((num_images, num_thresholds))

    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)

    for i in range(num_images):
        gt_order, pred_order, iou_scores = optimal_assignment_iou(pred_segmentation_stack[i], labels_stack[i])
        
        gt_order = np.array(gt_order)
        pred_order = np.array(pred_order)
        iou_scores = np.array(iou_scores)
        
        for j, threshold in enumerate(matching_thresholds):
            tp[j] = (iou_scores > threshold).sum()
            fp[j] = len(pred_segmentation_stack[i]) - (iou_scores > threshold).sum()
            fn[j] = len(labels_stack[i]) - (iou_scores > threshold).sum()

        precisions[i,:] = tp / (tp + fp + fn)

    return precisions.mean(axis=0), precisions.std(axis=0)

def get_isbi_metrics(pred_segmentation_stack, labels_stack):
    """Compute the metrics from the ISBI challenge:
    - average dice coefficient of all matchings with IoU > 0.7
    - object-based false negative rate (missed gt cells and matchings with IoU < 0.7)
    - average pixel-based true positive rate of all matchings with IoU > 0.7
    - average pixel-based false positive rate of all matchings with IoU > 0.7 

    Parameters:
    pred_segmentation_stack -- list of arrays, with each array having shape (num_pred_cells, height, width)
    labels_stack -- list of arrays, with each array having shape (num_gt_cells, height, width)
    
    Returns: dc, dc_std, fnr fnr_std, tpr, tpr_std, fpr, fpr_std
    """

    assert len(pred_segmentation_stack) == len(labels_stack)

    num_images = len(labels_stack)

    qualified_dc_list = []
    qualified_tpr_list = []
    qualified_fpr_list = []

    num_gt_objects = 0
    fnr_images = []

    for i in range(num_images):
        gt_order, pred_order, _ = hungry_assignment_iou(pred_segmentation_stack[i], labels_stack[i])
        num_matchings = gt_order.shape[0]
        num_gt_objects += labels_stack[i].shape[0]
        num_gt_objects_i = labels_stack[i].shape[0]
        num_qualified_i = 0

        for j in range(num_matchings):
            prediction = pred_segmentation_stack[i][pred_order][j].astype('bool')
            gt = labels_stack[i][gt_order][j].astype('bool')

            dc = 2 * np.count_nonzero(np.logical_and(prediction, gt)) / (np.count_nonzero(prediction) + np.count_nonzero(gt))
            if dc < 0.7:
                continue

            diff = np.logical_xor(prediction, gt)

            # pixel-based metris
            tp = np.logical_and(prediction, gt).sum()
            tn = np.logical_and(np.invert(gt), np.invert(prediction)).sum()
            fp = np.logical_and(diff, prediction).sum()
            fn = np.logical_and(diff, gt).sum()

            qualified_dc_list.append(dc)
            qualified_tpr_list.append(tp / (tp + fn))
            qualified_fpr_list.append(fp / (fp + tn))

            num_qualified_i += 1

        fnr_images.append((num_gt_objects_i - num_qualified_i) / num_gt_objects_i)
            
    dc_average = np.array(qualified_dc_list).mean()
    dc_std = np.array(qualified_dc_list).std()

    fnr = (num_gt_objects - len(qualified_dc_list)) / num_gt_objects
    fnr_std = np.array(fnr_images).std()

    tpr_average = np.array(qualified_tpr_list).mean()
    tpr_std = np.array(qualified_tpr_list).std()

    fpr_average = np.array(qualified_fpr_list).mean()
    fpr_std = np.array(qualified_fpr_list).std()

    return round(dc_average, 5), round(dc_std, 5), round(fnr, 5), round(fnr_std, 5), round(tpr_average, 5), round(tpr_std, 5), round(fpr_average, 5), round(fpr_std, 5)

def scores_prediction_grid(predictions, labels):
    """Compute the metrics from the ISBI15 challenge on a number of predictions for different parameters, as returned by prediction_grid().

    Parameters:
    predictions -- 2d list of predictions on the complete dataset
    labels -- list with arrays of labels for every image

    Returns: arrays dc, dc_std, fnr, fnr_std, tpr, tpr_std, fpr, fpr_std
    """

    dc = np.empty((len(predictions), len(predictions[0])))
    dc_std = np.empty((len(predictions), len(predictions[0])))
    fnr = np.empty((len(predictions), len(predictions[0])))
    fnr_std = np.empty((len(predictions), len(predictions[0])))
    tpr = np.empty((len(predictions), len(predictions[0])))
    tpr_std = np.empty((len(predictions), len(predictions[0])))
    fpr = np.empty((len(predictions), len(predictions[0])))
    fpr_std = np.empty((len(predictions), len(predictions[0])))
    
    for i in trange(len(predictions)):
        for j in trange(len(predictions[0])):
            dc[i, j], dc_std[i, j], fnr[i, j], fnr_std[i, j], tpr[i, j], tpr_std[i, j], fpr[i, j], fpr_std[i, j] = get_isbi_metrics(predictions[i][j], labels)
            
    return dc, dc_std, fnr, fnr_std, tpr, tpr_std, fpr, fpr_std

def precision_prediction_grid(predictions, labels, matching_thresholds):
    """Compute the average precision on a number of predictions for different parameters, as returned by prediction_grid().

    Parameters:
    predictions -- 2d list of predictions on the complete dataset
    labels -- list with arrays of labels for every image

    Returns: array with average precisions, array with std's
    """
    num_nms_thresholds = len(predictions)
    num_min_objprobs = len(predictions[0])
    num_matching_thresholds = len(matching_thresholds)

    precisions = np.zeros((num_nms_thresholds, num_min_objprobs, num_matching_thresholds))
    precisions_std = np.zeros((num_nms_thresholds, num_min_objprobs, num_matching_thresholds))
    for i in trange(num_nms_thresholds):
        for j in trange(num_min_objprobs):
                precisions[i,j,:], precisions_std[i,j,:] = get_precisions(predictions[i][j], labels, matching_thresholds)

    return precisions, precisions_std

def save_isbi_metrics(file, dc, dc_std, fnr, fnr_std, tpr, tpr_std, fpr, fpr_std, nms_thresholds, min_objprobs, num_proposals):
    """Save isbi metrics and parameters with pickle."""
    data = {"dc":dc, "dc_std":dc_std, "fnr":fnr, "fnr_std":fnr_std, "tpr":tpr, "tpr_std":tpr_std, "fpr":fpr, "fpr_std":fpr_std, "nms_thresholds":nms_thresholds, "min_objprobs":min_objprobs, "num_proposals":num_proposals}
    pickle.dump(data, open(file, 'wb'))

def save_precisions(file, precisions, precisions_std, nms_thresholds, min_objprobs, num_proposals, matching_thresholds):
    """Save precisions and parameters with pickle"""
    data = {"precision":precisions, "precision_std":precisions_std, "nms_thresholds":nms_thresholds, "min_objprobs":min_objprobs, "num_proposals":num_proposals, "matching_thresholds":matching_thresholds}
    pickle.dump(data, open(file, 'wb'))
