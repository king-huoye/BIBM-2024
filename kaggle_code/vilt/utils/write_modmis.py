# -*- coding = utf-8 -*-
# @Time：2024/3/7 13:21
# @Author：Bin
# @File：write_nii.py
# @Software：PyCharm
import json

import pyarrow as pa
import random
import os

from tqdm import tqdm
import nibabel as nib
import pandas as pd
import torch
import numpy as np
from kaggle_code.vilt.transforms.utils import pixelbert_np, norm255_np
import pickle
import zlib
from torchvision import transforms


def make_image_id_json(
    image_paths: dict, label_paths: dict, id_json_path: str
):
    assert len(image_paths) * len(label_paths) > 0
    exist_id_list = list(set(image_paths.keys()) & set(label_paths.keys()))
    id_path_dict = {id_str: [image_paths[id_str], label_paths[id_str]] for id_str in exist_id_list}
    os.makedirs(os.path.dirname(id_json_path), exist_ok=True)
    with open(id_json_path, 'w') as write_f:
        json.dump(id_path_dict, write_f, indent=4, ensure_ascii=False)

    print("both have:", len(exist_id_list), "\nimage only:", len(image_paths), "\nlabel only:", len(label_paths))
    print('Write json file done.')


def make_field_id_json(
    csv_path: str, id_json_path: str
):
    id_list = pd.read_csv(csv_path, encoding='gbk').values[:, 0].astype(np.int32).tolist()
    id_list = set(map(str, id_list))
    id_dict = {id_str: None for id_str in id_list}
    os.makedirs(os.path.dirname(id_json_path), exist_ok=True)
    with open(id_json_path, 'w') as write_f:
        json.dump(id_dict, write_f, indent=4, ensure_ascii=False)
    print(len(id_dict), " samples in ", csv_path)
    print('Write json file done.')


def make_split_by_id_json(
    id_json_path_list: list, split_json_path: str, allow_miss: bool,
    max_count: int, train_rate: float, val_rate: float, test_rate: float, reshuffle=True
):
    id_set = None
    for path in id_json_path_list:
        with open(path, "r") as fp:
            json_data = json.load(fp)
            if allow_miss:
                id_set = set(json_data.keys()) if id_set is None else id_set | set(json_data.keys())
            else:
                id_set = set(json_data.keys()) if id_set is None else id_set & set(json_data.keys())

    id_list = list(id_set)
    print("Total sample count:", len(id_set))
    max_count = min(max_count, len(id_list))
    if reshuffle:
        random.shuffle(id_list)

    total_rate = train_rate + val_rate + test_rate
    tr_count = int(max_count * train_rate / total_rate)
    tr_dv_count = tr_count + int(max_count * val_rate / total_rate)
    save_dict = {
        'train': id_list[:tr_count],
        'val': id_list[tr_count:tr_dv_count],
        'test': id_list[tr_dv_count:max_count]
    }
    os.makedirs(os.path.dirname(split_json_path), exist_ok=True)
    with open(split_json_path, 'w') as write_f:
        json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
    print('Write json file done.')


def make_image_pickle(split_json_path: str, image_id_json_path: str, dataset_root: str, name: str, image_size=384):
    with open(split_json_path, "r") as fp:
        split_sets = json.load(fp)
    with open(image_id_json_path, "r") as fp:
        image_id_path = json.load(fp)
    id_set = set(image_id_path.keys())

    for split, sample_id_list in split_sets.items():
        data_dict = {}
        for sid in tqdm(sample_id_list):
            if sid in id_set:
                data_dict[sid] = (
                    zlib.compress(  # 只做归一化和放缩
                        pickle.dumps(pixelbert_np(
                        norm255_np(np.array(nib.load(image_id_path[sid][0]).dataobj, dtype=np.int16)),
                        image_size, normalize=False)[0])
                    ),
                    zlib.compress(  # 只做归一化和放缩
                        pickle.dumps(pixelbert_np(
                        norm255_np(np.array(nib.load(image_id_path[sid][1]).dataobj, dtype=np.uint8)),
                        image_size, dtype=torch.uint8, normalize=False)[0])
                    ),
                )
            else:
                data_dict[sid] = (
                    zlib.compress(pickle.dumps(torch.zeros((image_size, image_size), dtype=torch.float32))),
                    zlib.compress(pickle.dumps(torch.zeros((image_size, image_size), dtype=torch.uint8))),
                )
        path = f"{dataset_root}/modmis_{name}_{split}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_field_pickle(split_json_path: str, csv_path: str, dataset_root: str, name: str):
    with open(split_json_path, "r") as fp:
        split_sets = json.load(fp)

    csv_data = pd.read_csv(csv_path, encoding='gbk')
    id_list = list(map(str, csv_data.iloc[:, 0]))
    id_map_dict = {k: v for v, k in enumerate(id_list)}
    id_set = set(id_list)
    field_count = len(csv_data.values[0, 1:])
    for split, sample_id_list in split_sets.items():
        data_dict = {}
        for sid in tqdm(sample_id_list):
            if sid in id_set:
                data_dict[sid] = torch.from_numpy(csv_data.values[id_map_dict[sid], 1:]).type(torch.float32)
            else:
                data_dict[sid] = torch.zeros(field_count, dtype=torch.float32)
        path = f"{dataset_root}/modmis_{name}_{split}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_modmis_image_pickles(
    raw_image_data_dir: str, raw_csv_data_dir: str, output_dir: str,
    allow_miss=False, max_count=100, train_rate=0.7, val_rate=0.15,
    test_rate=0.15, reshuffle=False, image_size=384
):
    tal_dir = f"{raw_image_data_dir}/1044tumorAndLymph"
    ot_dir = f"{raw_image_data_dir}/1352Onlytumor"
    csv_dir = raw_csv_data_dir

    for image_modal in ("T1", "T1C", "T2"):
        id_json_path = f"{output_dir}/jsons/{image_modal}_image.json"

        image_modal = f"{image_modal}_".lower()
        image_paths, label_paths = {}, {}
        for sid in os.listdir(tal_dir):
            for name in os.listdir(f"{tal_dir}/{sid}"):
                name = name.lower()
                if image_modal in name:
                    if "image" in name:
                        image_paths[sid] = f"{tal_dir}/{sid}/{name}"
                    # # label用tumorAndLymph的
                    # elif "label1_2" in name:
                    #     label_paths[sid] = f"{tal_dir}/{sid}/{name}"
        # tal_id = os.listdir(tal_dir)
        for sid in os.listdir(ot_dir):
            # # label用tumorAndLymph的
            # if sid in tal_id:
            #     continue
            for name in os.listdir(f"{ot_dir}/{sid}"):
                name = name.lower()
                if image_modal in name:
                    if "image" in name:
                        image_paths[sid] = f"{ot_dir}/{sid}/{name}"
                    elif "label1" in name:
                        label_paths[sid] = f"{ot_dir}/{sid}/{name}"

        make_image_id_json(image_paths, label_paths, id_json_path)

    for name in os.listdir(csv_dir):
        make_field_id_json(f"{csv_dir}/{name}", f"{output_dir}/jsons/fields_{name}.json")

    make_split_by_id_json(
        [f"{output_dir}/jsons/{name}" for name in os.listdir(f"{output_dir}/jsons")],
        f"{output_dir}/datasets/splits.json",
        allow_miss, max_count, train_rate, val_rate, test_rate, reshuffle,
    )

    for image_modal in ("T1", "T1C", "T2"):
        make_image_pickle(f"{output_dir}/datasets/splits.json",
                          f"{output_dir}/jsons/{image_modal}_image.json",
                          f"{output_dir}/datasets", f"{image_modal}_image",
                          image_size=image_size)
    for name in os.listdir(csv_dir):
         make_field_pickle(f"{output_dir}/datasets/splits.json", f"{csv_dir}/{name}",
                           f"{output_dir}/datasets", f"f_{name[1:-4]}")


# 计算指定模态图像的沿通道维度的平均值和标准差
def compute_mean_std(
    image_id_json_path_list: list,
    output_path: str,
    image_size=384,
):
    def compute_stats(id_list, image_id_path: dict):
        # 初始化分组存储器
        group_data = {}
        # 遍历所有图像ID并分组累积
        for image_id in id_list:
            image = pixelbert_np(
                norm255_np(np.array(nib.load(image_id_path[image_id][0]).dataobj, dtype=np.int16)),
                image_size, normalize=False)[0]
            c = image.shape[0]  # 获取当前图像的通道数
            if c not in group_data:
                group_data[c] = []
            group_data[c].append(image)

        # 计算每个分组的平均值和标准差
        all_stats = {}
        for c, images in group_data.items():
            # 将列表中的图像堆叠成一个大的张量，以便一次性计算平均值和标准差
            stacked_images = torch.stack(images)
            mean = torch.mean(stacked_images, dim=(0, 2, 3))  # 沿着通道和空间维度计算平均值
            std = torch.std(stacked_images, dim=(0, 2, 3))      # 沿着通道和空间维度计算标准差
            
            all_stats[c] = {"mean": [mean,], "std": [std,], "ct": len(images)}
        
        return all_stats

    batch_size=128
    modal_mean_std_dict = {}
    for image_id_json_path in image_id_json_path_list:
        with open(image_id_json_path, "r") as fp:
            image_id_path = json.load(fp)
        id_list = list(image_id_path.keys())
        print(len(id_list), image_id_json_path)
        
        normalizers = {}
        all_means = {}
        all_stds = {}
        cn_count = {}
        
        # 对id_list进行分批次处理
        for i in tqdm(range(0, len(id_list), batch_size)):
            batch_ids = id_list[i:i+batch_size]
            batch_stats = compute_stats(batch_ids, image_id_path)  # 假设compute_stats函数已定义并能正确工作
            
            for k, v in batch_stats.items():
                if k in all_means:
                    all_means[k] += v["mean"]
                    all_stds[k] += v["std"]
                    cn_count[k] += v["ct"]
                else:
                    all_means[k] = v["mean"]
                    all_stds[k] = v["std"]
                    cn_count[k] = v["ct"]

        mean_list = []
        std_list= []
        count_list = []
        for ck in all_means.keys():
            mean_v = torch.stack(all_means[ck])
            std_v = torch.stack(all_stds[ck])
            all_means[ck] = (mean_v.mean(), mean_v.mean(dim=0), mean_v)
            all_stds[ck] = (std_v.mean(), std_v.mean(dim=0), std_v)
            normalizers[ck] = transforms.Normalize(mean=all_means[ck][1], std=all_stds[ck][1])
            mean_list.append(all_means[ck][0])
            std_list.append(all_stds[ck][0])
            count_list.append(cn_count[ck])
        mean_arr = torch.stack(mean_list)
        std_arr = torch.stack(std_list)
        ct_arr = torch.tensor(count_list).float()
        modal_mean_std_dict[os.path.splitext(os.path.basename(image_id_json_path))[0]] = {
            "normalizer": normalizers,
            "mean": all_means,
            "std": all_stds,
            "count": cn_count,
            "w_mean_std": (torch.dot(mean_arr, ct_arr) / ct_arr.sum(), torch.dot(std_arr, ct_arr) / ct_arr.sum())
        }

    with open(output_path, 'wb') as handle:
        pickle.dump(modal_mean_std_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
