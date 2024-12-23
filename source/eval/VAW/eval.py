from typing import Optional

import argparse
import os
import json
import numpy as np

from .evaluator import Evaluator


def eval(
        pred: np.array,
        gt_label: np.array,
        evaluator: Optional[Evaluator] = None,
        output_dir: Optional[str] = None,
        ann_path: Optional[str] = None,
        print_detailed_results: bool = False
    ):

    if evaluator is None:
        if ann_path is None:
            raise ValueError('`ann_path` must not be None if `evaluator` is None')
        
        evaluator = Evaluator(
            os.path.join(ann_path, 'attr2idx.json'),
            os.path.join(ann_path, 'attribute_types.json'),
            os.path.join(ann_path, 'attribute_parent_types.json'),
            os.path.join(ann_path, 'head_tail.json')
        )

    # Compute scores.
    scores_overall, scores_per_class = evaluator.evaluate(pred, gt_label)
    scores_overall_topk, scores_per_class_topk = evaluator.evaluate(pred, gt_label, threshold_type='topk')
    
    CATEGORIES = ['all', 'head', 'medium', 'tail']
    if print_detailed_results:
        CATEGORIES += list(evaluator.attribute_parent_type.keys())

    for category in CATEGORIES:
        print(f"----------{category.upper()}----------")
        print(f"mAP: {100. * scores_per_class[category]['ap']:.2f}")
        
        if print_detailed_results:
            print(f"Per-class (threshold {evaluator.threshold}):")       
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_per_class[category]:
                    print(f"- {metric}: {scores_per_class[category][metric]:.4f}")
        
        if print_detailed_results:
            print("Per-class (top 15):")
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_per_class_topk[category]:
                    print(f"- {metric}: {scores_per_class_topk[category][metric]:.4f}")
    
        if print_detailed_results:
            print(f"Overall (threshold {evaluator.threshold}):")
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_overall[category]:
                    print(f"- {metric}: {scores_overall[category][metric]:.4f}")

        if print_detailed_results:
            print("Overall (top 15):")
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_overall_topk[category]:
                    print(f"- {metric}: {scores_overall_topk[category][metric]:.4f}")

    if output_dir is None:
        return {
            'scores_overall': scores_overall,
            'scores_per_class': scores_per_class,
            'scores_overall_topk': scores_overall_topk,
            'scores_per_class_topk': scores_per_class_topk,
        }
    
    # Store raw results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'scores_overall': scores_overall,
            'scores_per_class': scores_per_class,
            'scores_overall_topk': scores_overall_topk,
            'scores_per_class_topk': scores_per_class_topk
        }, f)

    # Store results as table
    with open(os.path.join(output_dir, 'detailed_results.txt'), 'w') as f:
        f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
        f.write('-----------------------------------------------------------------------------------------------------\n')
        for i_class in range(evaluator.n_class):
            att = evaluator.idx2attr[i_class]
            f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
                att,
                evaluator.get_score_class(i_class).ap,
                evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
                evaluator.get_score_class(i_class).get_bacc(),
                evaluator.get_score_class(i_class).n_pos,
                evaluator.get_score_class(i_class).n_neg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath_pred', type=str, required=True,
                        help='path to model prediction numpy array file')
    parser.add_argument('--fpath_label', type=str, required=True,
                        help='path to groundtruth label file')
    
    parser.add_argument('--fpath_attribute_index', type=str,
                        default='../data/attribute_index.json')
    parser.add_argument('--fpath_attribute_types', type=str,
                        default='../data/attribute_types.json')
    parser.add_argument('--fpath_attribute_parent_types', type=str,
                        default='../data/attribute_parent_types.json')
    parser.add_argument('--fpath_head_tail', type=str,
                        default='../data/head_tail.json')

    parser.add_argument('--output', type=str, default='output_detailed.txt')
    
    args = parser.parse_args()

    pred = np.load(args.fpath_pred)
    gt_label = np.load(args.fpath_label)
    print(f'# of instances: {pred.shape[0]}')

    evaluator = Evaluator(
        args.fpath_attribute_index, args.fpath_attribute_types,
        args.fpath_attribute_parent_types, args.fpath_head_tail)

    # Compute scores.
    scores_overall, scores_per_class = evaluator.evaluate(pred, gt_label)
    scores_overall_topk, scores_per_class_topk = evaluator.evaluate(
        pred, gt_label, threshold_type='topk')
    
    CATEGORIES = ['all', 'head', 'medium', 'tail'] + \
        list(evaluator.attribute_parent_type.keys())

    for category in CATEGORIES:
        print(f"----------{category.upper()}----------")
        print(f"mAP: {scores_per_class[category]['ap']:.4f}")
        
        print("Per-class (threshold 0.5):")
        for metric in ['recall', 'precision', 'f1', 'bacc']:
            if metric in scores_per_class[category]:
                print(f"- {metric}: {scores_per_class[category][metric]:.4f}")
        print("Per-class (top 15):")
        for metric in ['recall', 'precision', 'f1']:
            if metric in scores_per_class_topk[category]:
                print(f"- {metric}: {scores_per_class_topk[category][metric]:.4f}")
    
        print("Overall (threshold 0.5):")
        for metric in ['recall', 'precision', 'f1', 'bacc']:
            if metric in scores_overall[category]:
                print(f"- {metric}: {scores_overall[category][metric]:.4f}")
        print("Overall (top 15):")
        for metric in ['recall', 'precision', 'f1']:
            if metric in scores_overall_topk[category]:
                print(f"- {metric}: {scores_overall_topk[category][metric]:.4f}")

    with open(args.output, 'w') as f:
        f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
        f.write('-----------------------------------------------------------------------------------------------------\n')
        for i_class in range(evaluator.n_class):
            att = evaluator.idx2attr[i_class]
            f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
                att,
                evaluator.get_score_class(i_class).ap,
                evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
                evaluator.get_score_class(i_class).get_bacc(),
                evaluator.get_score_class(i_class).n_pos,
                evaluator.get_score_class(i_class).n_neg))
