import torch
import tqdm
import glob
import os
import numpy as np
from torchreid.scripts.default_config import get_default_config, display_config_diff
from torchreid.tools.feature_extractor import FeatureExtractor

def extract_part_based_features(extractor, image_list, batch_size=400):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_embeddings = []
    all_visibility_scores = []
    all_masks = []

    images_chunks = chunks(image_list, batch_size)
    for chunk in tqdm.tqdm(images_chunks):
        embeddings, visibility_scores, masks = extractor(chunk)

        embeddings = embeddings.cpu().detach()
        visibility_scores = visibility_scores.cpu().detach()
        masks = masks.cpu().detach()

        all_embeddings.append(embeddings)
        all_visibility_scores.append(visibility_scores)
        all_masks.append(masks)

    all_embeddings = torch.cat(all_embeddings, 0).numpy()
    all_visibility_scores = torch.cat(all_visibility_scores, 0).numpy()
    all_masks = torch.cat(all_masks, 0).numpy()

    return {
        "parts_embeddings": all_embeddings,
        "parts_visibility_scores": all_visibility_scores,
        "parts_masks": all_masks,
    }

def extract_part_based_features_tuple(extractor, image_list, batch_size=400):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_embeddings = []
    all_visibility_scores = []
    all_masks = []

    images_chunks = chunks(image_list, batch_size)
    for chunk in tqdm.tqdm(images_chunks):
        outputs = extractor(chunk)
        embeddings, visibility_scores, masks = outputs[0], outputs[1], outputs[5]

        for k, v in embeddings.items():
            if k == 'bn_foreg':
                v = v if len(v.shape) == 3 else v.unsqueeze(1)
                all_embeddings.append(v.cpu().detach())
            if k == "parts":
                v = v if len(v.shape) == 3 else v.unsqueeze(1)
                all_embeddings.append(v.cpu().detach())
        for k, v in visibility_scores.items():
            if k == "foreg":
                v = v if len(v.shape) == 2 else v.unsqueeze(1)
                all_visibility_scores.append(v.cpu().detach())
            if k == "parts":
                v = v if len(v.shape) == 2 else v.unsqueeze(1)
                all_visibility_scores.append(v.cpu().detach())
        for k, v in masks.items():
            if k == "foreg":
                v = v  if len(v.shape) == 4 else v.unsqueeze(1)
                all_masks.append(v.cpu().detach())
            if k == "parts":
                v = v if len(v.shape) == 4 else v.unsqueeze(1)
                all_masks.append(v.cpu().detach())

        # embeddings, visibility_scores, masks = extractor(chunk)
        # all_embeddings.append(embeddings)
        # all_visibility_scores.append(visibility_scores)
        # all_masks.append(masks)

    all_embeddings = torch.cat(all_embeddings, 1)
    all_visibility_scores = torch.cat(all_visibility_scores, 1)
    all_masks = torch.cat(all_masks, 1)

    return {
        "parts_embeddings": all_embeddings,
        "parts_visibility_scores": all_visibility_scores,
        "parts_masks": all_masks,
    }

def extract_det_idx(img_path):
    return int(os.path.basename(img_path).split('_')[0])


def extract_det_endless(img_path):
    return int(os.path.basename(img_path).split('.')[0])


def extract_reid_features(cfg, base_folder, out_path, model=None, model_path=None, num_classes=None):
    extractor = FeatureExtractor(
        cfg,
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=num_classes,
        model=model
    )

    print("Looking for video folders with images crops in {}".format(base_folder))
    folder_list = glob.glob(base_folder + '/*')
    for folder in folder_list:
        image_list = glob.glob(os.path.join(folder, "*.png"))
        image_list.sort(key=extract_det_idx)
        print("{} images to process for folder {}".format(len(image_list), folder))
        results = extract_part_based_features(extractor, image_list, batch_size=50)

        # dump to disk
        video_name = os.path.splitext(os.path.basename(folder))[0]
        parts_embeddings_filename = os.path.join(out_path, "embeddings_" + video_name + ".npy")
        parts_visibility_scores_filanme = os.path.join(out_path, "visibility_scores_" + video_name + ".npy")
        parts_masks_filename = os.path.join(out_path, "masks_" + video_name + ".npy")

        os.makedirs(os.path.dirname(parts_embeddings_filename), exist_ok=True)
        os.makedirs(os.path.dirname(parts_visibility_scores_filanme), exist_ok=True)
        os.makedirs(os.path.dirname(parts_masks_filename), exist_ok=True)

        np.save(parts_embeddings_filename, results['parts_embeddings'])
        np.save(parts_visibility_scores_filanme, results['parts_visibility_scores'])
        np.save(parts_masks_filename, results['parts_masks'])

        print("features saved to {}".format(out_path))

def extract_our_reid_features(cfg, base_folder, out_path, model=None, model_path=None, num_classes=None):
    extractor = FeatureExtractor(
        cfg,
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=num_classes,
        model=model
    )

    print("Looking for video folders with images crops in {}".format(base_folder))
    image_list = sorted([f for f in os.listdir(base_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print("{} images to process for folder {}".format(len(image_list), base_folder))
    result_list = list()
    for f in image_list:
        current_batch = list()
        current_batch.append(os.path.join(base_folder, f))
        result = extract_part_based_features_tuple(extractor, current_batch, batch_size=1)

        # dump to disk
        file_name = os.path.splitext(os.path.basename(f))[0]
        parts_embeddings_filename = os.path.join(out_path, "embeddings_" + file_name + ".npy")
        parts_visibility_scores_filanme = os.path.join(out_path, "visibility_scores_" + file_name + ".npy")
        parts_masks_filename = os.path.join(out_path, "masks_" + file_name + ".npy")

        os.makedirs(os.path.dirname(parts_embeddings_filename), exist_ok=True)
        os.makedirs(os.path.dirname(parts_visibility_scores_filanme), exist_ok=True)
        os.makedirs(os.path.dirname(parts_masks_filename), exist_ok=True)

        np.save(parts_embeddings_filename, result['parts_embeddings'].numpy())
        np.save(parts_visibility_scores_filanme, result['parts_visibility_scores'].numpy())
        np.save(parts_masks_filename, result['parts_masks'].numpy())

        print("features saved to {}".format(out_path))
        result_list.append(result)
    return image_list, result

