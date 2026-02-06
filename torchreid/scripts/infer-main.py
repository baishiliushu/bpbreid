from __future__ import division, print_function, absolute_import
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ========================
# 一键运行配置（只需修改这里！）
# ========================
# 图片路径（直接修改这两行即可切换图片）
# /home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/compare_images/special-rois
IMG_ENDLESS = "jpg"
BASE_PATH = "/home/leon/mount_point_d/test-result-moved/reid_datas/npy_todo/" #"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/compare_images"
IMG1_PATH = f"{BASE_PATH}/src/squence_based_time/d17/1.png"   #e1-1.{IMG_ENDLESS} special-rois/id4-g.png Screenshot from 2025-11-06 14-13-41.png
IMG2_PATH = f"{BASE_PATH}/src/squence_based_time/d17/"   # special-rois/id2-e2 ; 1105-display/Screenshot from 2025-11-06 14-29-45.png Screenshot from 2025-11-06 14-15-20.png

# 模型和配置文件路径（固定后无需修改）
# MODEL_PATH = "/home/leon/mount_point_c/githubs_new/sota-reid-tcks/reid-files/onnx-files/bpbreid_market1501_hrnet32_10642.pth" # bpbreid_market1501_hrnet32_10642.pth # bpbreid_dukemtmcreid_hrnet32_10669 bpbreid_market1501_hrnet32_10642
MODEL_PATH = "/home/leon/mount_point_c/githubs_new/sota-reid-tcks/reid-files/onnx-files/bpbreid_models/market1-120_model.state_no_pkl.pth"

CONFIG_FILE = "/home/leon/mount_point_c/githubs_new/sota-reid-tcks/bpbreid/configs/bpbreid/bpbreid_market1501_test.yaml"

# 是否使用GPU（默认开启）
USE_GPU = False
VISUAL_BOOL = True
VISUAL_MIN = 0.55
DO_EXPORT = False

# ========================
# 仅导入必需模块（完全来自你的代码）
# ========================
from torchreid.utils import load_pretrained_weights
from torchreid.models import build_model
from torchreid.scripts.default_config import get_default_config
from torchreid.data.masks_transforms import compute_parts_num_and_names
from torchreid.utils.constants import *

from torchreid.metrics.distance import (
    compute_distance_matrix_using_bp_features,
    masked_mean,
    replace_values
)


# 新增导入函数部分，import类


# ========================
# 复刻part_based_engine.py的核心方法
# ========================
def extract_test_embeddings(model_output, test_embeddings):
    """完全复刻ImagePartBasedEngine.extract_test_embeddings的逻辑"""
    embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
    embeddings_list = []
    visibility_scores_list = []
    embeddings_masks_list = []
    # TODO: network structure mode = 2
    for test_emb in test_embeddings:
        embds = embeddings[test_emb]
        embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
        if test_emb in bn_correspondants:
            test_emb = bn_correspondants[test_emb]
        vis_scores = visibility_scores[test_emb]
        visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
        if parts_masks is not None:
            pt_masks = parts_masks[test_emb]
            embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

    assert len(embeddings) != 0
    embeddings = torch.cat(embeddings_list, dim=1)  # [N, P+F, D]
    visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+F]

    embeddings_masks = None
    if len(embeddings_masks_list) > 0:
        embeddings_masks = torch.cat(embeddings_masks_list, dim=1)  # [N, P+F, Hf, Wf]
    return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores


def extract_test_embeddings_postprocess(model_output, parts_num=5, infer_mode=3):
    foreground_masks = None
    parts_masks = None

    if infer_mode == 1 or infer_mode == 2:
        bn_foreground_embeddings, parts_embeddings, foreground_visibility, parts_visibility, foreground_masks, parts_masks = model_output
    if infer_mode == 3:
        bn_foreground_embeddings, parts_embeddings, pixels_parts_probabilities, foreground_masks, parts_masks = model_output

        pixels_parts_predictions_org = pixels_parts_probabilities.argmax(dim=1)  # [N, Hf, Wf]

        # pixels_parts_predictions = F.one_hot(pixels_parts_predictions_org, self.parts_num + 1) #
        # pixels_parts_predictions_one_hot = pixels_parts_predictions.permute(0, 3, 1, 2)  # [N, K+1, Hf, Wf]
        # parts_visibility = pixels_parts_predictions_one_hot.amax(dim=(2, 3))  # [N, K+1]
        # parts_visibility = parts_visibility.to(torch.bool)

        one_hot = torch.zeros(*pixels_parts_predictions_org.shape, parts_num + 1,
                              dtype=pixels_parts_predictions_org.dtype)
        one_hot.scatter_(3, pixels_parts_predictions_org.unsqueeze(3), 1)
        one_hot.to(torch.int32)
        one_hot = one_hot.permute(0, 3, 1, 2)
        one_hot = one_hot.amax(dim=(2, 3))
        all_visibility = one_hot.to(torch.bool)
        foreground_visibility = all_visibility.amax(dim=1)  # [N]
        parts_visibility = all_visibility[:, 1:]  # [N, K]
    if infer_mode == 7:
        bn_foreground_embeddings, parts_embeddings, foreground_visibility, parts_visibility = model_output

    N = 10
    #values = bn_foreground_embeddings[0, :N].detach().cpu().tolist()  # 转为Python列表
    values = bn_foreground_embeddings.reshape(-1)[:N].detach().cpu().tolist()
    print(" ".join(f"{v:.4f}" for v in values))

    for i in range(parts_embeddings.size(1)):  # 遍历5个部位
        vals = parts_embeddings[0, i, :N].detach().cpu().tolist()
        print(f"Part {i} (前{N}维): " + " ".join(f"{v:.4f}" for v in vals))

    embeddings = {
        GLOBAL: None,  # [N, D]
        BACKGROUND: None,  # [N, D]
        FOREGROUND: None,  # [N, D]
        CONCAT_PARTS: None,  # [N, K*D]
        PARTS: parts_embeddings,  # [N, K, D]
        BN_GLOBAL: None,  # [N, D]
        BN_BACKGROUND: None,  # [N, D]
        BN_FOREGROUND: bn_foreground_embeddings,  # [N, D]
        BN_CONCAT_PARTS: None,  # [N, K*D]
        BN_PARTS: None,  # [N, K, D]
    }

    visibility_scores = {
        GLOBAL: None,  # [N]
        BACKGROUND: None,  # [N]
        FOREGROUND: foreground_visibility,  # [N]
        CONCAT_PARTS: None,  # [N]
        PARTS: parts_visibility,  # [N, K]
    }

    masks = {
        GLOBAL: None,  # [N, Hf, Wf]
        BACKGROUND: None,  # [N, Hf, Wf]
        FOREGROUND: foreground_masks,  # [N, Hf, Wf]
        CONCAT_PARTS: foreground_masks,  # [N, Hf, Wf]
        PARTS: parts_masks,  # [N, K, Hf, Wf]
    }
    if foreground_masks is None or parts_masks is None:
        masks = None

    return embeddings, visibility_scores, None, None, None, masks


def normalize_features(features):
    """完全复刻ImagePartBasedEngine.normalize的逻辑（L2归一化）"""
    return nn.functional.normalize(features, p=2, dim=-1)


# ========================
# 加载配置（无需命令行参数）
# ========================
def build_config():
    cfg = get_default_config()
    print("[Debug]D0 -> cfg.test_embeddings = {}, cfg.mask.parts_num = {}".format(cfg.model.bpbreid.test_embeddings,
                                                                                  cfg.model.bpbreid.masks.parts_num))
    cfg.merge_from_file(CONFIG_FILE)
    cfg.model.name = 'bpbreid'
    cfg.use_gpu = USE_GPU and torch.cuda.is_available()
    compute_parts_num_and_names(cfg)
    print("[Debug]D1 -> cfg.test_embeddings = {}, cfg.mask.parts_num = {}".format(cfg.model.bpbreid.test_embeddings,
                                                                                  cfg.model.bpbreid.masks.parts_num))
    cfg.test_embeddings = cfg.model.bpbreid.test_embeddings
    return cfg


# ========================
# 图片预处理
# ========================
def preprocess_image(img_path, cfg):
    transform = transforms.Compose([
        transforms.Resize((384, 128)),  # HRNet输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 384, 128]
    if cfg.use_gpu:
        img_tensor = img_tensor.cuda()

    return img_tensor


# ========================
# 构建模型
#ckpt = torch.load("market1-120_model.pth.tar", map_location="cpu")
#torch.save(ckpt["state_dict"], "market1-120_state_dict.pth")
# ========================
def build_model_and_load_weights(cfg, struct_mode=0):
    model = build_model(
        name=cfg.model.name,
        num_classes=1000,
        loss='part_based',
        pretrained=False,
        use_gpu=cfg.use_gpu,
        config=cfg,
        struct_mode=struct_mode
    )
    load_pretrained_weights(model, MODEL_PATH, weights_only=False)
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    model.eval()
    return model


# ========================
# export_onnx(model, img, external_parts_masks=None)
# ========================

def file_size(path):
    from pathlib import Path
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


def export_onnx(model, im, pth_path, external_parts_masks=None, opset=11, train=False, dynamic=False, simplify=False):
    # ONNX export
    # model.to(str('cpu'))
    print("export {} with opset-{}, dynamic {}, simplify {}".format(pth_path, opset, dynamic, simplify))
    try:
        import onnx
        from onnx import shape_inference
        f = "{}.onnx".format(pth_path)
        print(f'\nStarting export with onnx {onnx.__version__}...')

        torch.onnx.export(
            model,  #  if dynamic else model --dynamic only compatible with cpu
            im, #if dynamic else im
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],  # 'images'
            output_names=['ff', 'fp', 'vf', 'vp'], # 'ff', 'fp', 'vf', 'vp', 'mf', 'mp'
            dynamic_axes={
                'images': {
                    0: '1',  # 'batch',
                },  # shape(x,3,256,128)
                # 'output': {
                #     0: '1',  # 'batch',
                # }  # shape(x,2048)
                # 'fp': {
                #     0: '1',
                # },
                # 'vf':{0: '1'},
                # 'vp': {
                #     0: '1',
                # }
            } if dynamic else None
        )
        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        model_onnx = shape_inference.infer_shapes(model_onnx)

        onnx.save(model_onnx, f)
        f = "{}.simonnx.onnx".format(pth_path)
        # Simplify
        if simplify:

            try:
                cuda = torch.cuda.is_available()
                import onnxsim

                print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'simplifier failure: {e}')
        print(f'export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f"run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'export failure: {e}')
    return f


# TODO: masks_output_visualize(img, outputs)


# ========================
# 核心相似度计算
# ========================
def visual_pair_score_map(vi_q, vi_g, min_t=VISUAL_MIN, bool_map=VISUAL_BOOL):
    mapped_value = min(vi_q, vi_g)

    if mapped_value > min_t:
        mapped_value = (vi_q + vi_g) * 0.5
        if bool_map:
            mapped_value = 1
    else:
        mapped_value = 0
    return mapped_value


def compute_similarity(img1, img2, model, cfg, img1_path="", img2_path="", infer_mode=0):
    with torch.no_grad():
        # 模型推理（明确参数名）
        output1 = model(images=img1, external_parts_masks=None)
        output2 = model(images=img2, external_parts_masks=None)
        feat1, vis1, mask1, pixel_cls1 = None, None, None, None
        # 提取特征和可见性分数
        if infer_mode > 0:
            output1 = extract_test_embeddings_postprocess(output1, infer_mode=infer_mode)
            output2 = extract_test_embeddings_postprocess(output2, infer_mode=infer_mode)

        feat1, vis1, mask1, pixel_cls1 = extract_test_embeddings(output1, cfg.test_embeddings)
        feat2, vis2, mask2, pixel_cls2 = extract_test_embeddings(output2, cfg.test_embeddings)
        print("[Debug]SHAPES -> features:{}, visuals:{}".format(feat1.shape, vis1.shape))
        print("{} \n {}".format(img1_path, vis1))
        print("{} \n {}".format(img2_path, vis2))
        # 特征归一化
        feat1 = normalize_features(feat1)
        feat2 = normalize_features(feat2)


        '''    
             # GPU张量转CPU再转numpy
             vis1 = vis1.cpu().numpy()[0]
             vis2 = vis2.cpu().numpy()[0]
        
         # 按论文公式9计算总距离：可见性加权欧式距离
         total_dist = 0.0
         total_weight = 0.0
         for i in range(feat1.shape[1]):  # 遍历前景+6个部位
             weight = vis1[i] * vis2[i]
             if weight < 1e-6:
                 continue
        
             # 欧式距离（GPU张量先转CPU）
             f1 = feat1[0, i, :].cpu().numpy()
             f2 = feat2[0, i, :].cpu().numpy()
             dist = np.linalg.norm(f1 - f2)
        
             total_dist += weight * dist
             total_weight += weight
        
         # 距离转相似度
         if total_weight < 1e-6:
             total_dist = np.inf
             similarity = 0.0
         else:
             total_dist /= total_weight
             similarity = 1 / (1 + total_dist)
        
         return total_dist, similarity
        '''
        total_dist_mat, body_part_dist_mat = compute_distance_matrix_using_bp_features(
            qf=feat1,
            gf=feat2,
            qf_parts_visibility=vis1,
            gf_parts_visibility=vis2,
            dist_combine_strat='mean',  # 对应论文公式9的加权平均
            batch_size_pairwise_dist_matrix=5000,
            use_gpu=cfg.use_gpu,
            metric='euclidean'  # 官方欧式距离
            )

        # 2. 提取总距离（压缩维度为标量）
        total_dist = total_dist_mat.squeeze().cpu().numpy()
        # 处理无有效部位的情况（官方函数未完全处理，补充论文逻辑）
        # if total_dist == float('inf') or np.isnan(total_dist):
        #     total_dist = np.inf
        #     similarity = 0.0
        # else:
        #     # 距离转相似度（保持原有映射逻辑）
        #     similarity = 1 / (1 + total_dist)
        similarity, body_part_dist_mat = compute_distance_matrix_using_bp_features(
            qf=feat1,
            gf=feat2,
            qf_parts_visibility=vis1,
            gf_parts_visibility=vis2,
            dist_combine_strat='mean',  # 对应论文公式9的加权平均
            batch_size_pairwise_dist_matrix=5000,
            use_gpu=cfg.use_gpu,
            metric='cosine'  # 官方cosine
        )
        cos_sim_total = nn.functional.cosine_similarity(feat1, feat2, dim=2)
        ol_dist_total = torch.sqrt(torch.sum((feat1.squeeze(0) - feat2.squeeze(0)) ** 2, dim=1)).unsqueeze(0)


        cos_sim_foreground = nn.functional.cosine_similarity(feat1[:, 0:1, :], feat2[:, 0:1, :], dim=2)
        feat1_f = feat1[:, 0:1, :]
        feat2_f = feat2[:, 0:1, :]
        ol_dist_foreground = torch.sqrt(torch.sum((feat1_f.squeeze(0) - feat2_f.squeeze(0)) ** 2, dim=1)).unsqueeze(0)

        mean_cos = 0.0
        mean_dist = 3.0

        print("[Debug]cos_all : \n{}\ndist_all : {}\ncos_fore : {}\ndist_fore : {}".format(cos_sim_total, ol_dist_total,
                                                                                  cos_sim_foreground, ol_dist_foreground))

        #fused = sum(weights[i] * scores[i]) / sum(weights)
        fuse_part_cos = -1.0
        fuse_part_foreground = -1.0

        vis1 = vis1.squeeze().tolist()
        vis2 = vis2.squeeze().tolist()
        cos_sim_total = cos_sim_total.squeeze().tolist()
        v_weights = list()
        if len(vis1) == len(vis2) and len(vis1) > 1:
            if cos_sim_total[0] < 0.5:
                print("[INFO]Almost should believe they're different person BASED the low foreground cosine.")
            for i in range(1, len(vis1)):
                v_weights.append(visual_pair_score_map(vis1[i], vis2[i]))

            parts_cos_seen = list()
            for i in range(0, len(v_weights)):
                cos_weight = cos_sim_total[i + 1] * v_weights[i]
                parts_cos_seen.append(cos_weight)

            fuse_part_cos = sum(parts_cos_seen) / max(sum(v_weights), 0.0001)
            print("[Debug]cos_fused : {:.4f}".format(fuse_part_cos))
            v_foreground = min(vis1[0], vis2[0])
            if sum(v_weights) < 4:
                print("[INFO]Maybe should believe part value INSTEAD-OF foreground value.")

            if v_foreground > VISUAL_MIN:
                v_foreground = visual_pair_score_map(vis1[0], vis2[0])
                v_weights.append(v_foreground)
                parts_cos_seen.append(cos_sim_total[0] * v_foreground)
                fuse_part_foreground = sum(parts_cos_seen) / max(sum(v_weights), 0.0001)
                print("[Debug]cos_fused_foreground : {:.4f}".format(fuse_part_foreground))
        print("visual w: {}".format(v_weights))

        return total_dist, similarity.squeeze().cpu().numpy(), fuse_part_cos, fuse_part_foreground


# ========================
# 主函数（一键运行，无需参数）
# ========================
def main():
    # 1. 加载配置
    cfg = build_config()
    print(f"✅ 配置加载完成：{CONFIG_FILE}")

    # 2. 构建模型并加载权重
    INFER_MODE = 7   # 1 -1 2 3
    print(f"加载预训练模型：{MODEL_PATH}")

    # 3. 预处理张图片
    print(f"加载图片：\n{IMG1_PATH}")
    img1 = preprocess_image(IMG1_PATH, cfg)

    # x. EXPORT
    if DO_EXPORT:
        model_4_export = build_model_and_load_weights(cfg, struct_mode=-1)
        export_onnx(model_4_export, img1, MODEL_PATH,
                    simplify=True)

    model = build_model_and_load_weights(cfg, struct_mode=INFER_MODE)
    print("✅ 模型加载完成，已切换至推理模式 (MODE {})".format(INFER_MODE))

    gallery_image_paths = list()
    if os.path.isdir(IMG2_PATH):
        # TODO: pick up only in ['.jpg', '.png']
        # for i in  os.listdir(IMG2_PATH):
        #     if i.endswith(".jpg")
        img_files = os.listdir(IMG2_PATH)
        img_files = sorted(img_files, key=lambda x: int(x.split('.')[0]))
        gallery_image_paths = [os.path.join(IMG2_PATH, i) for i in img_files]

    else:
        gallery_image_paths.append(IMG2_PATH)

    q_file_name = os.path.basename(IMG1_PATH)
    result_infos = list()
    for gpath in  gallery_image_paths:
        g_file_name = os.path.basename(gpath)
        print("\n" + "=" * 60)
        print(f"BPBreID 两张图片相似度计算结果(infer mode = {INFER_MODE})")
        print(f"预训练模型：{os.path.basename(MODEL_PATH)}")
        print(f"图片1：{q_file_name}")
        print(f"图片2：{g_file_name}")
        print("=" * 60)

        # 4. 计算相似度
        img2 = preprocess_image(gpath, cfg)
        total_dist, similarity, cos_by_parts, cos_by_all = compute_similarity(img1, img2, model, cfg, IMG1_PATH, gpath, infer_mode=INFER_MODE)
        # 5. 输出结果
        print(f"euclidean距离（越小越相似）：{total_dist:.4f}")
        print(f"cos_official相似度（越小越相似）：{similarity:.4f}")
        print("-" * 60)
        rst = {"g_file": g_file_name, "cos_all":float("{:.5f}".format(cos_by_all)), "cos_parts": float("{:.5f}".format(cos_by_parts)), "eucl": f"{total_dist:.3f}", "cos_official": f"{similarity:.3f}"}
        result_infos.append(rst)

    results_sort_key = 'cos_parts'
    result_infos = sorted(result_infos, key=lambda x: x.get(results_sort_key, float('-inf')), reverse=True)
    for r in result_infos:
        print("{}".format(r))
    print(f"图片query：{q_file_name}, SORT-KEY {results_sort_key}")
    print(f"BPBreID 两张图片相似度计算结果(infer mode = {INFER_MODE})")
    print(f"预训练模型：{os.path.basename(MODEL_PATH)}")


if __name__ == '__main__':
    main()

