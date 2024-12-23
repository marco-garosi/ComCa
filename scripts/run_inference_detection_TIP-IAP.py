import sys
sys.path.append('./')

import warnings
warnings.filterwarnings('ignore')

import os
import glob

import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import transforms

from ovad.misc import ovad_validate

from source.utils.encode_clip import *
from source.utils.reproducibility import *
from source.utils import (
    get_device,
    load_model,
    load_templates_cache_generation,
)
from source.data import get_ovad_dataloader, get_dataloader
from source.tip_adapter import load_cache
from source.model import (
    Predictor, PredictorWithCache, Cache,
    embed_prompts
)
from source.eval import evaluate_attribute_detection


CACHE_TEMPLATE = '{prefix}_{t}__{model}__{shots}_shots{suffix}.pt'


def get_arguments():
    parser = argparse.ArgumentParser(description='open_clip evaluation')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ovad2000',
        help='dataset name',
    )
    parser.add_argument(
        '--ann_file',
        type=str,
        default='ovad/ovad2000.json',
        help='annotation file with images and objects for attribute annotation',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out/OVAD_Benchmark2000/multimodal_ovac/open_clip_ours/',
        help='dir where models are',
    )
    parser.add_argument(
        '--dir_data',
        type=str,
        default='datasets/ovad_box_instances/2000_img',
        help='image data dir',
    )
    parser.add_argument(
        '--model_from',
        type=str,
        default='OpenCLIP',  # 'OpenCLIP' # 'HF'
        help='where to load the model from',
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='ViT-B/16',  # 'RN50' #'RN101' #'RN50x4' #'ViT-B/32' #'ViT-B/16'
        help='architecture name',
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default='laion400m_e31',  # 'cc12m' #'yfcc15m' #'openai' #'laion400m_e31' #'laion400m_e32' #'laion2b_e16'
        help='dataset trained name',
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='all',
        help='prompt',
    )
    parser.add_argument(
        '--average_syn',
        action='store_true',
    )
    parser.add_argument(
        '--object_word',
        action='store_true',
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.17,
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='sd'
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--scale_logits',
        type=str,
        default='max',
    )
    parser.add_argument(
        '--augmented_cache',
        type=str,
        default=None
    )
    parser.add_argument(
        '--softmax',
        type=str,
        default='none'
    )
    parser.add_argument(
        '--soft_labels',
        type=str,
        default=''
    )
    parser.add_argument(
        '--scale_base_logits',
        type=float,
        default=1.
    )
    parser.add_argument(
        '--cache_by_category',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--cache_by_sample',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--path_for_image_features',
        type=str
    )
    parser.add_argument(
        '--path_to_cache',
        type=str,
        default=''
    )
    parser.add_argument(
        '--evaluate_metrics',
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        '--store_predictions',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--use_cache',
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        '--default_is_has',
        type=str,
        default='is'
    )
    parser.add_argument(
        '--template_file',
        type=str,
        default='OvarNet.json'
    )
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        default=None
    )

    parser.add_argument(
        '--detection_model',
        type=str,
        default='yolov11n.pt'
    )

    return parser.parse_args()


def main(args):
    object_word = 'object' if args.object_word else ''
    use_prompts = ['a', 'the', 'none']
    if args.prompt in use_prompts or args.prompt == 'photo':
        use_prompts = [args.prompt]

    # Get the folder for output and prepare for loading cache
    output_dir, att_emb_path, experiment_id = get_new_folder(args, object_word, max_id=20_000)
    print(f'Experiment ID: {experiment_id}')  # output now, so in case of crash we know where to look

    suffix = get_suffix(args)
    model_name = get_model_slug(args)

    # Store configuration in JSON file for reproducibility
    path_to_cache = os.path.join(args.path_to_cache, CACHE_TEMPLATE)
    store_config(args, path_to_cache, output_dir)
    save_args(args, script_name=os.path.basename(__file__), base_dir=output_dir)

    # Load annotations
    annotations = json.load(open(args.ann_file, 'r'))
    device = get_device(allow_accelerator=True)

    # Build Pandas DataFrame with categories
    categories = pd.DataFrame.from_dict(annotations['categories'])

    # Load TIP-Adapter Cache
    cache_keys, cache_values = load_cache(model_name, args.prefix, suffix, args.shots, None, path_to_cache, cache_by_category=args.cache_by_category, soft_labels=args.soft_labels, device=device)
    cache = Cache(args.alpha, args.beta, cache_keys, cache_values, scale_logits='none', softmax='overall')

    oa_distribution = torch.load(path_to_cache.format(prefix=args.prefix, t='attribute_dist', model=model_name, shots=args.shots, suffix=suffix), map_location=torch.device(device)).float().to(device)
    oa_distribution /= oa_distribution.sum(dim=-1, keepdim=True)  # normalize for each category, so that attributes within category sum to 1

    # Load backbone model
    model, processor, tokenizer = load_model(args.model_from, args.model_arch, pretrained=args.pretrained, device=device, cache_dir=args.model_cache_dir)

    # Load detection model
    detector = None
    if 'yolo' in args.detection_model.lower():
        from ultralytics import YOLO
        detector = YOLO(args.detection_model).to(device)

    assert detector is not None, 'Detector not provided'

    # Data
    data_loader = get_dataloader(
        'OVAD_DET', annotations_path=args.ann_file, images_path=args.dir_data, split='test',
        batch_size=args.batch_size,
        num_workers=os.cpu_count()
    )
    dataset = data_loader.dataset

    # Generate and embed (or load embeddings of) prompts
    templates = load_templates_cache_generation(args.template_file)['class_only']
    if args.prompt == 'all':
        templates = list(itertools.chain.from_iterable([list(x) for x in templates.values()]))
    else:
        templates = templates[args.prompt]
    categories_queries = []
    for category in categories['name'].tolist():
        # Generate prompts
        prompts = [
            template.format(noun=category)
            for template in templates
        ]

        category_queries = clip_encode_text(prompts, model, tokenizer, device=device, model_from=args.model_from, normalize=True)
        category_queries = category_queries.mean(dim=0)
        category_queries /= category_queries.norm(dim=-1)
        
        categories_queries.append(category_queries)
    text_features = torch.stack(categories_queries)

    len_synonyms = [1] * len(text_features)

    predictions = []
    ground_truth = []
    pred_vectors = []
    label_vectors = []
    indx_max_syn = []

    # Construct model for inference
    predictor = PredictorWithCache(
        model,
        args.model_from,
        processor,
        device,
        text_features,
        len_synonyms,
        cache=cache,
        average_syn=args.average_syn,
        batch_size=args.batch_size,
        sigmoid='siglip' in args.model_arch.lower(),
        model_arch=args.model_arch.lower(),
    )

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader)):
            ground_truth_bounding_boxes = [x['bounding_boxes'] for x in labels]
            att_label = [x['attribute_vectors'] for x in labels]
            label_vectors.append(torch.cat(att_label, dim=0).cpu().numpy())

            for sample_gt in labels:
                ground_truth.append({
                    'image_id': sample_gt['image_id'],
                    'num_instances': len(sample_gt['raw']),
                    'id_instances': [x['id'] for x in sample_gt['raw']],
                    'gt_boxes': sample_gt['bounding_boxes'],
                    'gt_classes': [x['category_id'] for x in sample_gt['raw']],
                    'gt_att_vec': sample_gt['attribute_vectors'].cpu().tolist(),
                })

            # Predict
            detections = detector(images, verbose=False)
            
            # Extract objects, and prepare data for prediction model
            predicted_bounding_boxes = []
            cropped_objects = []

            for prediction, image in zip(detections, images):
                preds = prediction.boxes.xyxy.cpu().numpy()
                predicted_bounding_boxes.append(preds)
                
                if preds is None:
                    cropped_objects.append([])
                    continue
                
                for pred in preds:
                    cropped_objects.append(image.crop(pred))

            # If cache is by sample (i.e. retrieved/generated by x_i), then
            # update the cache with the specific cache for current sample x_i
            if args.cache_by_sample:
                cache.keys = cache_keys[i].to(device)
                cache.values = cache_values[i].to(device)

            output = predictor(cropped_objects, scale_base_logits=args.scale_base_logits)#, image_features=image_features)
            image_features = output['image_features']
            x_attrs = output['scores']
            idx_attrs = output['idx_attrs']

            # IAP
            result = torch.zeros((len(x_attrs), 117)).to(device)
            for sample_x in range(len(x_attrs)):  # sample index
                for category in range(len(oa_distribution)):  # would be sum_{k=1}^{K} in paper [26]
                    result[sample_x] += oa_distribution[category] * x_attrs[sample_x][category]  # (117) * scalar

            x_attrs = result
            # End IAP

            group_by_sample = [len(x) for x in predicted_bounding_boxes]
            attribute_predictions_by_image = x_attrs.split(group_by_sample)

            for bbox_predictions_for_sample, attribute_predictions, image, label in zip(predicted_bounding_boxes, attribute_predictions_by_image, images, labels):
                data = {
                    'image_id': label['image_id'],
                    'instances': [],
                    'width': image.size[0],
                    'height': image.size[1],
                }
                
                for bbox_prediction, attr_prediction in zip(bbox_predictions_for_sample, attribute_predictions):
                    data['instances'].append({
                        'image_id': label['image_id'],
                        'category_id': 0,
                        'bbox': bbox_prediction.tolist(),
                        'score': 1.0,
                        'att_scores': attr_prediction.cpu().tolist(),
                    })

                predictions.append(data)

            # Store to computer metrics
            pred_vectors.append(x_attrs.cpu().numpy())
            indx_max_syn.append(idx_attrs.cpu().numpy())

    pred_vectors = np.concatenate(pred_vectors, axis=0)
    label_vectors = np.concatenate(label_vectors, axis=0)
    indx_max_syn = np.concatenate(indx_max_syn, axis=0)

    pred_vectors = np.nan_to_num(pred_vectors, nan=0)

    ###############
    #   Output    #
    ###############

    class Metadata:
        name: str = 'OVAD'
        attribute_classes: list = dataset.annotations['attributes']
    
        def __init__(self):
            with open(os.path.join('./res/ovad/attr2idx.json'), 'r') as f:
                self.att2idx = json.load(f)

    metadata = Metadata()

    if args.evaluate_metrics:
        evaluator = None
        gt_vec, pred_vec, gt_pred_match = evaluate_attribute_detection(
            predictions,
            ground_truth,
            metadata,
            output_dir,
            return_gt_pred_pair=False
        )

        ovad_validate(
            annotations['attributes'],
            pred_vec,
            gt_vec,
            output_dir,
            args.dataset_name,
            save_output=False,
            print_table=False,
            evaluator=evaluator,
            evaluating_box_free=True,
            gt_pred_match=gt_pred_match,

        )
    
    if args.store_predictions:
        np.save(os.path.join(output_dir, 'pred_vectors.npy'), pred_vectors)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
