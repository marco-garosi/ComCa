import os
import glob
import json


def save_args(args, script_name=None, base_dir=''):
    """Store the configuration (i.e. all the command-line arguments)
        to a file in JSON format for reproducibility
    """

    if 'out_folder' in args:
        folder = os.path.join(base_dir, 'out', 'metadata', args.out_folder)
    else:
        folder = base_dir
    os.makedirs(folder, exist_ok=True)
    
    args = args.__dict__
    if script_name is not None:
        args['script'] = script_name
    
    with open(os.path.join(folder, f'arguments.json'), 'w') as f:
        json.dump(args, f)


def get_suffix(args):
    """Builds the suffix for the cache
    """

    suffix = f'_seed_{args.seed}'

    if args.augmented_cache is not None:
        suffix += '_with_transforms__augment_' + str(args.augmented_cache)

    return suffix


def get_model_slug(args):
    """Constructs a "slug" for the model given the command-line arguments
    """

    return args.model_from.lower() + '__' + args.model_arch.replace('-', '_').replace('/', '_') + '__' + args.pretrained


def store_config(args, path_to_cache, base_dir):
    """Store the configuration (i.e. all the relevant command-line arguments)
        to a file in JSON format for reproducibility
    """

    path = os.path.join(base_dir, 'config.json')

    data = {
        'path_to_cache': path_to_cache,
        'prefix': args.prefix,
        'suffix': get_suffix(args),
        'shots': args.shots,
        'alpha': args.alpha,
        'beta': args.beta,
        'seed': args.seed,
        'augmented_cache': args.augmented_cache,
        'softmax': args.softmax,
        'soft_labels': args.soft_labels,
        'cache_by_category': args.cache_by_category,

        'model_from': args.model_from,
        'model_arch': args.model_arch,
        'pretrained': args.pretrained,
        'prompt': args.prompt,
        'average_syn': args.average_syn,
        'object_word': args.object_word,
        'scale_logits': args.scale_logits,
        'scale_base_logits': args.scale_base_logits,
    }

    with open(path, 'w') as f:
        json.dump(data, f)


def get_new_folder(args, object_word, max_id=10_000):
    """Generate a folder ID, and create it, to store the configuration and results
        of an experiment in.
        The ID is unique and does not take into account the other parts of the folder's
        name, so there cannot be two folders with the same ID
    """

    unavailable_ids = glob.glob(os.path.join(args.output_dir, '*'))
    unavailable_ids = {int(os.path.basename(id).split('__')[0]) for id in unavailable_ids if os.path.isdir(id)}
    
    available_ids = set(list(range(max_id))).difference(unavailable_ids)

    if len(available_ids) == 0:
        print(f'There are no more available ids. Please increase `max_id` (currently set to {max_id})')
        exit()

    # Sorting the list ensures we get the first available id in ascending order.
    experiment_id = sorted(list(available_ids))[0]

    try:
        output_dir = args.output_dir
        # make dir to save results and embeddings
        output_dir += (
            f'{experiment_id}_'
            + "_{}".format(args.model_from.lower())
            + "_{}".format(args.model_arch.replace("/", ""))
            + "_{}".format(args.pretrained.replace("_", ""))
            + ("_avsyn" if args.average_syn else "")
            + "_prompt-{}".format(args.prompt)
            + "{}".format(object_word)
        )
        os.makedirs(output_dir, exist_ok=False)
        att_emb_path = os.path.join(
            args.output_dir, f'att_{get_model_slug(args)}__{args.prompt}_{"_avsyn_" if args.average_syn else ""}_{object_word}__{args.template_file}.pkl'
        )
    except:
        print('An error occurred while creating the output folder')
        exit()

    return output_dir, att_emb_path, experiment_id
