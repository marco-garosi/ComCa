import torch
from torchvision.ops import box_iou
import numpy as np


def evaluate_attribute_detection(
    predictions,
    attributes,
    metadata,
    output_dir,
    metrics=None,
    return_gt_pred_pair=False,
    cls_depend=False,
    min_iou_match=0.5,
    include_missing_boxes=True,
    conditional=False,
):
    """
    Evaluate attribute detection metrics.
    """
    # 1. match gt bounding box with best predicted bounding box
    # 2. get attributes for every bounding box matched
    gt_overlaps = []
    gt_pred_match = []
    no_gt = 0
    no_pred = 0
    no_corrs = 0
    gt_vec = []
    pred_vec = []
    att_key = "cond_att_scores" if conditional else "att_scores"
    imgId2attBoxes = {sample["image_id"]: sample for sample in attributes}

    for img_prediction in predictions:
        img_id = img_prediction["image_id"]
        prediction = img_prediction["instances"]

        # Discard images without predictions
        if len(prediction) == 0:
            no_pred += 1
            continue

        # Get ground truth labels
        ground_truth = imgId2attBoxes[img_id]

        # Discard images without labels
        if len(ground_truth["id_instances"]) == 0:
            no_gt += 1
            continue

        # Build Boxes with ground truth and predictions
        gt_boxes = torch.tensor(ground_truth['gt_boxes'])
        pd_boxes = torch.tensor([pred['bbox'] for pred in prediction])

        # Calculate the IoU between ground truth and predictions
        overlaps = box_iou(pd_boxes, gt_boxes)
        # Consider IoUs bigger than min_iou_match (0.5)
        overlaps[overlaps < min_iou_match] = -1

        # get best matching proposals - gt boxes
        _gt_overlaps = torch.zeros(len(gt_boxes))
        gt_prop_corresp = []
        for j in range(min(len(pd_boxes), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            if gt_ovr < 0:
                continue
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            gt_prop_corresp.append((gt_ind, box_ind, _gt_overlaps[j]))
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        if len(gt_prop_corresp) != len(gt_boxes):
            no_corrs += len(gt_boxes) - len(gt_prop_corresp)
            if include_missing_boxes:
                # Find missing gt indexes
                matching_gt_ind = set(
                    [match_triple[0].item() for match_triple in gt_prop_corresp]
                )
                missing_gt_id = set(range(len(gt_boxes))).difference(matching_gt_ind)
                # Include missing indexes to the predictions as if model had not predicted anything -> 0 val prediction
                for gt_ind in missing_gt_id:
                    gt_att_vec = ground_truth["gt_att_vec"][gt_ind]
                    att_scores = [0.0] * len(gt_att_vec)
                    gt_vec.append(gt_att_vec)
                    pred_vec.append(att_scores)

        for gt_ind, box_ind, iou_score in gt_prop_corresp:
            gt_ind = gt_ind.item()
            box_ind = box_ind.item()

            obj_pair = {"image_id": ground_truth["image_id"]}
            for key, val in ground_truth.items():
                if isinstance(val, list):# or isinstance(val, torch.Tensor):
                    obj_pair[key] = val[gt_ind]
            box_pred = pd_boxes[box_ind : box_ind + 1]
            obj_pred = prediction[box_ind]
            for key in obj_pred.keys():
                if key not in obj_pair.keys():
                    obj_pair[key] = obj_pred[key]
            obj_pair["box_pred"] = box_pred
            obj_pair["iou_score"] = iou_score
            _ = obj_pair.pop("bbox")
            gt_pred_match.append(obj_pair)

            gt_vec.append(obj_pair["gt_att_vec"])
            pred_vec.append(np.asarray(obj_pair[att_key]))

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)

    if len(gt_overlaps) == 0:
        return {}

    if return_gt_pred_pair:
        return gt_pred_match
    else:
        file_path = (
            output_dir + ("_cond" if conditional else "_uncond") + "_gt_pred_match.pth"
        )

    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    pred_vec = np.vstack(pred_vec)
    gt_vec = np.vstack(gt_vec)

    gt_vec[gt_vec == -1] = 2

    attribute_list = metadata.attribute_classes
    assert len(attribute_list) == pred_vec.shape[1]
    assert len(attribute_list) == gt_vec.shape[1]

    return gt_vec, pred_vec, gt_pred_match