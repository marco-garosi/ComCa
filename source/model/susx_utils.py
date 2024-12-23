import torch
import torch.nn as nn
from tqdm import tqdm


def scale_(x, target):
    
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y


def compute_image_text_distributions(temp, train_images_features_agg, test_features, val_features, vanilla_zeroshot_weights):
    train_image_class_distribution = train_images_features_agg.T @ vanilla_zeroshot_weights
    train_image_class_distribution = nn.Softmax(dim=-1)(train_image_class_distribution/temp)

    test_image_class_distribution = test_features @ vanilla_zeroshot_weights
    test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution/temp)

    val_image_class_distribution = val_features @ vanilla_zeroshot_weights
    val_image_class_distribution = nn.Softmax(dim=-1)(val_image_class_distribution/temp)

    return train_image_class_distribution, test_image_class_distribution, val_image_class_distribution


def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution, bs: int = 100):
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0]//bs)):
        curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)    
        q = train_image_class_distribution
        q_repeated = torch.cat([q]*bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl  

    return kl_divs_sim


def get_kl_div_sims(temperature, test_features, val_features, train_features, clip_weights, bs: int = 100):

    train_image_class_distribution, test_image_class_distribution, val_image_class_distribution = compute_image_text_distributions(temperature, train_features, test_features, val_features, clip_weights)

    # train_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, train_image_class_distribution)
    test_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution, bs=bs)
    # val_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, val_image_class_distribution)
    train_kl_divs_sim, val_kl_divs_sim = None, None

    return train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim
