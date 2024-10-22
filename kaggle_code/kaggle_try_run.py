import json
import pyarrow as pa
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import torch


def move_all_files(src_base_dir="G:/datasets/本院鼻咽癌nii最终版1000+病例谢辉整理2024-01-12",
               tar_base_dir="F:/鼻咽癌数据集"):
    import os
    import shutil
    from tqdm.auto import tqdm

    src_dir = f"{src_base_dir}/1044tumorAndLymph"
    tar_dir = f"{tar_base_dir}/1044tumorAndLymph"
    os.makedirs(tar_dir, exist_ok=True)

    for sub_dir in tqdm(os.listdir(src_dir), desc="复制数据集"):
        os.makedirs(f"{tar_dir}/{sub_dir}", exist_ok=True)
        for name in os.listdir(f"{src_dir}/{sub_dir}"):
            shutil.copy(f"{src_dir}/{sub_dir}/{name}", f"{tar_dir}/{sub_dir}/{name}")

    tal_id = os.listdir(tar_dir)
    src_dir = f"{src_base_dir}/1352Onlytumor"
    tar_dir = f"{tar_base_dir}/1352Onlytumor"
    os.makedirs(tar_dir, exist_ok=True)

    for sub_dir in tqdm(os.listdir(src_dir), desc="复制数据集"):
        os.makedirs(f"{tar_dir}/{sub_dir}", exist_ok=True)
        if sub_dir in tal_id:  # 只复制标签 原图像在1044里面有
            for name in os.listdir(f"{src_dir}/{sub_dir}"):
                if "image" not in name:
                    shutil.copy(f"{src_dir}/{sub_dir}/{name}", f"{tar_dir}/{sub_dir}/{name}")
        else:  # 复制所有
            for name in os.listdir(f"{src_dir}/{sub_dir}"):
                shutil.copy(f"{src_dir}/{sub_dir}/{name}", f"{tar_dir}/{sub_dir}/{name}")


def move_files():
    import os
    import shutil
    from tqdm.auto import tqdm

    src_dir = "G:/datasets/本院鼻咽癌nii最终版1000+病例谢辉整理2024-01-12/1044tumorAndLymph"
    tar_dir = "F:/鼻咽癌数据集/1044tumorAndLymph"
    os.makedirs(tar_dir, exist_ok=True)

    for sub_dir in tqdm(os.listdir(src_dir), desc="复制数据集"):
        os.makedirs(f"{tar_dir}/{sub_dir}", exist_ok=True)
        for name in os.listdir(f"{src_dir}/{sub_dir}"):
            if not name.startswith("flip_T2_ax_image"):
                shutil.copy(f"{src_dir}/{sub_dir}/{name}", f"{tar_dir}/{sub_dir}/{name}")


def show_roc(pkl_path: str):
    import pickle
    from vilt.gadgets.my_metrics import ROC_Drawer
    with open(pkl_path, "rb") as f:
        fprs, tprs, aucs = pickle.load(f)
    drawer = ROC_Drawer()
    for fpr, tpr, auc in zip(fprs, tprs, aucs):
        drawer.plot_label_roc(fpr, tpr, auc)


def show_roc2(dir: str):
    import pickle
    from vilt.gadgets.my_metrics import ROC_Drawer
    name_list = os.listdir(dir)
    name_list.sort()
    for id, name in enumerate(name_list):
        id = id % 5 - 1
        with open(f"{dir}/{name}", "rb") as f:
            epoch, auc, fprs, tprs, aucs = pickle.load(f)
        print(id, epoch, auc)
        if id < 0:
            continue
        drawer = ROC_Drawer()
        drawer.plot_label_roc(fprs[id], tprs[id], auc)


def show_roc3(dir: str):
    import pickle
    from vilt.gadgets.my_metrics import ROC_Drawer
    name_list = os.listdir(dir)
    name_list.sort()
    for id, name in enumerate(name_list):
        with open(f"{dir}/{name}", "rb") as f:
            epoch, auc, fprs, tprs, aucs = pickle.load(f)
        print(epoch, auc)
        drawer = ROC_Drawer()
        drawer.plot_label_roc(fprs[id], tprs[id], aucs[id])


def check_torch_n_cuda():
    import torch
    import sys
    print(sys.version)
    print("Pytorch version：", torch.__version__)
    print("CUDA Version: ", torch.version.cuda)
    print("CUDA Name: ", torch.cuda.get_device_name(0))
    print("cuDNN version is :", torch.backends.cudnn.version())


def compare_compress():
    import pickle
    import zlib
    import time

    with open("F:/鼻咽癌数据集/compressed/datasets/modmis_T1_image_train.pkl", 'rb') as handle:
        sample_dict = pickle.load(handle)

    test_count = 10
    stt = time.time()
    for t in range(test_count):
        for sid, image_label in sample_dict.items():
            image = pickle.loads(zlib.decompress(image_label[0]))
            label = pickle.loads(zlib.decompress(image_label[1]))
            # print(torch.sum(torch.abs(image.type(torch.float32))))
    print(len(sample_dict))
    print((time.time() - stt) / test_count)

    # with open("F:/鼻咽癌数据集/datasets/modmis_T1_image_train.pkl", 'rb') as handle:
    #     sample_dict = pickle.load(handle)
    # test_count = 10
    # stt = time.time()
    # for t in range(test_count):
    #     for sid, image_label in sample_dict.items():
    #         image = image_label[0]
    #         label = image_label[1]
    # print((time.time() - stt) / test_count)
# compare_compress()


def read_norm_img(data_dir="F:/鼻咽癌数据集/compressed2"):
    import pickle
    import nibabel as nib
    import zlib
    from vilt.transforms.utils import pixelbert_np, norm255_np
    import matplotlib.pyplot as plt

    image_modal_list = ["T1_image", "T1C_image", "T2_image"]
    for img_modal in image_modal_list:
        with open(f"{data_dir}/jsons/{img_modal}.json", "r") as fp:
            image_id_path = json.load(fp)
        with open(f"{data_dir}/datasets/modmis_{img_modal}_train.pkl", 'rb') as handle:
            img_dict = pickle.load(handle)
        for sid, value in img_dict.items():
            ori_image = np.array(nib.load(image_id_path[sid][0]).dataobj, dtype=np.int16)
            processed_image = pickle.loads(zlib.decompress(value[0])).numpy()
            print(sid)
            print(np.sum(ori_image < 0), np.sum(ori_image > 0),
                  np.sum(ori_image), np.min(ori_image), np.max(ori_image))
            print(np.sum(processed_image < 0), np.sum(processed_image > 0),
                  np.sum(processed_image), np.min(processed_image), np.max(processed_image))
            for i in range(ori_image.shape[-1]):
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(ori_image[:, :, i], cmap='gray')
                axs[0].set_title('origin image')
                axs[0].axis('off')
                axs[1].imshow(processed_image[i, :, :], cmap='gray')
                axs[1].set_title('processed image')
                axs[1].axis('off')
                plt.tight_layout()
                plt.show()
# read_norm_img()

def read_img_label(data_dir="../datasets/modmis/pickles/datasets2"):
    import pickle
    # import nibabel as nib
    import zlib
    import matplotlib.pyplot as plt
    image_modal_list = ["T1_image", "T1C_image", "T2_image"]
    for img_modal in image_modal_list:
        with open(f"{data_dir}/modmis_{img_modal}_train.pkl", 'rb') as handle:
            img_dict = pickle.load(handle)
        for sid, value in img_dict.items():
            pro_image = pickle.loads(zlib.decompress(value[0])).numpy()
            pro_mask = pickle.loads(zlib.decompress(value[1])).numpy()
            for i in range(pro_image.shape[0]):
                if pro_mask[i, :, :].sum() == 0:
                    continue
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                pro_image_lay = pro_image[i, :, :]
                pro_mask_lay = pro_mask[i, :, :]
                for ind, _img in enumerate((pro_image_lay,)):
                    axs[0].imshow(_img, cmap='gray')
                    axs[0].set_title(f'image {ind}')
                    axs[0].axis('off')
                for ind, _img in enumerate((pro_mask_lay,)):
                    axs[1].imshow(_img, cmap='gray')
                    axs[1].set_title(f'mask {ind}')
                    axs[1].axis('off')
                for ind, (_img, _mask) in enumerate(((pro_image_lay, pro_mask_lay),)):
                    axs[2].imshow(_img, cmap='gray', interpolation='none')

                    mask = np.zeros((_img.shape[0], _img.shape[1], 4))
                    mask[_img > 0.2] = [0, 1, 0, 0.3]  # Red color with 50% transparency
                    mask[_mask == 1] = [0, 1, 0, 0.3]  # Red color with 50% transparency
                    axs[2].imshow(mask, interpolation='none')

                    mask = np.zeros((_img.shape[0], _img.shape[1], 4))
                    mask[_mask == 1] = [1, 0, 0, 0.3]  # Red color with 50% transparency
                    axs[2].imshow(mask, interpolation='none')

                    axs[2].set_title(f'labeled image {ind}')
                    axs[2].axis('off')
                plt.tight_layout()
                plt.show()
# read_img_label()

def mean_std_data_compute():
    from vilt.utils.write_modmis import compute_mean_std
    compute_mean_std(
        [
            "F:/鼻咽癌数据集/compressed/jsons/T1_image.json",
            "F:/鼻咽癌数据集/compressed/jsons/T1C_image.json",
            "F:/鼻咽癌数据集/compressed/jsons/T2_image.json",
        ],
        "F:/鼻咽癌数据集/compressed/mean_std.pkl"
    )
  
def read_mean_std_data():
    import pickle
    with open("F:/鼻咽癌数据集/compressed/mean_std.pkl", 'rb') as handle:
        mean_std_dict = pickle.load(handle)
    with open("F:/鼻咽癌数据集/compressed/mean_std2.pkl", 'wb') as handle:
        pickle.dump(mean_std_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(mean_std_dict.keys())
    for image_key in ("T1_image", "T1C_image", "T2_image"):
        print(mean_std_dict[image_key].keys())
        print(mean_std_dict[image_key]["normalizer"].keys())
        print(mean_std_dict[image_key]["w_mean_std"])

def get_field_name_dict(merge_list=[], csv_dir="../raw_data/modmis/fields"):
    file_name_list = ['2base', '3conventional', '4special', '5blood', '6complication']
    field_name_list = [f"f_{file_name[1:]}" for file_name in file_name_list]
    field_name_dict = {}
    for field_name, file_name in zip(field_name_list, file_name_list):
        csv_data = pd.read_csv(f"{csv_dir}/{file_name}.csv", encoding='gbk')
        field_name_dict[field_name] = csv_data.columns.values.tolist()[1:]
    field_name_dict["f_ebv"] = field_name_dict["f_base"][6:10]  # 分离ebv
    field_name_dict["f_base"] = field_name_dict["f_base"][:6] + field_name_dict["f_base"][10:-3]  # 去除治疗
    field_name_dict["f_normal"] = []
    for field_name in field_name_list:
        field_name_dict["f_normal"] += field_name_dict[field_name]
    for merge_name, merge_fields in merge_list:
        field_name_dict[merge_name] = []
        for field_name in merge_fields:
            field_name_dict[merge_name] += field_name_dict[field_name]
    return field_name_dict

def get_field_ind(field_name_dict: dict, key_name: str, q_names: list):
    return [field_name_dict[key_name].index(q_name) for q_name in q_names]

def run_lgb(data_dir="../datasets/modmis/pickles/datasets2"):
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    import pickle
    import torchmetrics.functional.classification as TFC

    used_labels = (0,)
    field_column_name_list = ['f_base', 'f_conventional', 'f_special', 'f_blood', 'f_complication']
    file_kstr = ["f_ending", "f_base", "f_conventional", "f_special", "f_blood", "f_complication"]
    # field_must = ["T分期", "Ttwo", "N分期", "Ntwo", "test2f", "test2ebv", "test3ebv", "EBVDNA拷贝数103copyml.治疗前", "年龄", "性别"]
    field_must = ["T分期", "Ttwo", "N分期", "Ntwo", "EBVDNA拷贝数103copyml.治疗前", "年龄", "性别"]
    # field_must = []
    consider_count = 20
    filter_arr = np.array(list(range(194)))
    # filter_arr = np.array([0, 1, 6, 7, 8, 9, 6, 141, 142, 143, 144, 150, 154, 159, 161, 163, 190, 191, 192, 193])
    # filter_arr = np.array([0, 1, 2, 3, 6, 7, 8, 9, 96, 141, 142, 143, 144, 146, 148, 150, 157, 159, 161])

    # 6个表格+3种图像的模态
    with open(f"{data_dir}/splits.json", "r") as fp:
        phase_used_id = json.load(fp)
    phase_data_dict = {}
    for split in ("train", "val", "test"):
        # 读取文件
        phase_data_dict[split] = {}
        for fname in os.listdir(data_dir):
            if split in fname:
                for kstr in file_kstr:
                    if kstr in fname:
                        with open(f"{data_dir}/{fname}", 'rb') as handle:
                            phase_data_dict[split][kstr] = pickle.load(handle)
        # 分离出EBV 去除治疗
        phase_data_dict[split]["f_ebv"] = {}
        for sid, v in phase_data_dict[split]['f_base'].items():
            phase_data_dict[split]["f_ebv"][sid] = v[6:10]
            phase_data_dict[split]['f_base'][sid] = torch.cat((v[:6], v[10:-3])).nan_to_num(nan=-1)
        # 合并字段信息 f_normal 为所需要的
        phase_data_dict[split]["f_normal"] = {}
        for sid in phase_used_id[split]:
            phase_data_dict[split]["f_normal"][sid] = torch.cat([
                phase_data_dict[split][kstr][sid]
                for kstr in field_column_name_list
            ]).nan_to_num(nan=-1)
        # for kstr in field_column_name_list:
        #     phase_data_dict.pop(kstr, None)
        # 调整ending字段
        for sid in phase_used_id[split]:
            phase_data_dict[split]["f_ending"][sid] = phase_data_dict[split]["f_ending"][sid][range(0, 8, 2)].int()

    tmp_items = [
        (
            torch.cat([
                torch.stack([phase_data_dict[split]["f_normal"][sid] for sid in phase_used_id[split]]),
                torch.stack([phase_data_dict[split]["f_ebv"][sid] for sid in phase_used_id[split]]),
            ], dim=1)[:, filter_arr].numpy(),
            torch.stack([phase_data_dict[split]["f_ending"][sid] for sid in phase_used_id[split]])[:, used_labels].numpy()
        )
        for split in ("train", "val", "test")
    ]
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = tmp_items
    # print(train_x.shape, train_y.shape)
    # print(val_x.shape, val_y.shape)
    # print(test_x.shape, test_y.shape)
    params = {
        'tr_kw_params': {
            'params': {
                'num_leaves': 32,  # 31
                'min_data_in_leaf': 2,  # 20
                'objective': 'binary',  # 定义的目标函数
                'max_depth': -1,  # -1
                'learning_rate': 0.001,  # 0.001
                # "min_sum_hessian_in_leaf": 6,
                "boosting": "dart",  # gbdt, rf, dart, goss
                "feature_fraction": 0.9,  # 0.9 提取的特征比率 X
                "bagging_freq": 1,  # 1 X
                "bagging_fraction": 0.8,  # 0.8 X
                "bagging_seed": 11,  # 11 X
                "lambda_l1": 0.01,  # 0.1 l1正则
                # 'lambda_l2': 0.001,		# 0.001 l2正则 X
                "verbosity": -1,
                "nthread": -1,  # 线程数量，-1表示全部线程，线程越多，运行的速度越快
                'metric': {'binary_logloss', 'auc'},  ##评价函数选择
                "random_state": 0,  # 随机数种子，可以防止每次运行的结果不一致
                # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
            },
            'num_boost_round': 500,
        },
        'stop_round': 300,
    }

    mg_train_x, mg_train_y = np.concatenate((train_x, val_x)), np.concatenate((train_y, val_y))
    all_test_probs = np.zeros_like(test_y, dtype=np.float32)
    all_importance = []
    all_importance_score = []
    for label_id in range(len(used_labels)):
        print("=" * 58, f" {label_id} ", "=" * 58)
        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
            print("fold {}".format(fold_ + 1))

            train_data = lgb.Dataset(mg_train_x[trn_idx], label=mg_train_y[trn_idx, label_id])
            val_data = lgb.Dataset(mg_train_x[val_idx], label=mg_train_y[val_idx, label_id])
            clf = lgb.train(
                train_set=train_data, valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=params['stop_round'])],
                **params['tr_kw_params'],
            )
            probs = clf.predict(test_x)
            all_importance.append(clf.feature_importance())
            all_importance_score.append(TFC.binary_auroc(torch.from_numpy(probs[:, None]), torch.from_numpy(test_y)))
            all_test_probs[:, label_id] += probs / folds.n_splits
        print("=" * 120)
    print(all_importance_score)
    print(TFC.binary_auroc(torch.from_numpy(all_test_probs), torch.from_numpy(test_y)))

    all_importance = np.stack(all_importance)
    min_v = all_importance.min(axis=1, keepdims=True)
    all_importance = (all_importance - min_v) / (all_importance.max(axis=1, keepdims=True) - min_v)
    all_importance = (all_importance - all_importance.mean(axis=1, keepdims=True)) / all_importance.std(axis=1, keepdims=True)
    all_importance_score = np.stack(all_importance_score)
    feat_importance = (all_importance_score / all_importance_score.sum()) @ all_importance
    min_v = feat_importance.min()
    feat_importance = (feat_importance - min_v) / (feat_importance.max() - min_v)

    field_name_dict = get_field_name_dict([("used", ("f_normal", "f_ebv"))])
    used_field_names = np.array(field_name_dict["used"])

    sort_ind = np.argsort(feat_importance)[::-1].astype(np.int32)[: consider_count]
    field_must_ind = np.array(get_field_ind(field_name_dict, "used", field_must))
    sort_ind = sort_ind[~np.in1d(filter_arr[sort_ind], field_must_ind)][:consider_count-len(field_must_ind)]
    # print(sort_ind)
    # print(feat_importance[sort_ind])
    # print(used_field_names[sort_ind])

    sort_ind = np.sort(sort_ind)
    # print(sort_ind)
    # print(feat_importance[sort_ind])
    # print(used_field_names[filter_arr][sort_ind])
    consider_ind = np.array(get_field_ind(field_name_dict, "used", field_must + used_field_names[filter_arr][sort_ind].tolist()))
    consider_ind = np.sort(consider_ind)
    print(consider_ind)
    print(used_field_names[consider_ind])
# run_lgb()  # 8381 7859

def run_xgb(data_dir="../datasets/modmis/pickles/datasets"):
    from sklearn.model_selection import KFold
    from xgboost import XGBClassifier
    import pickle
    import torchmetrics.functional.classification as TFC

    used_labels = (0,)
    field_column_name_list = ['f_base', 'f_conventional', 'f_special', 'f_blood', 'f_complication']
    file_kstr = ["f_ending", "f_base", "f_conventional", "f_special", "f_blood", "f_complication"]
    field_must = ["T分期", "Ttwo", "N分期", "Ntwo", "test2f", "test2ebv", "test3ebv", "EBVDNA拷贝数103copyml.治疗前", "年龄", "性别"]
    # field_must = []
    consider_count = 20
    filter_arr = np.array(list(range(194)))
    # filter_arr = np.array([0, 1, 6, 7, 8, 9, 6, 141, 142, 143, 144, 150, 154, 159, 161, 163, 190, 191, 192, 193])

    # 6个表格+3种图像的模态
    with open(f"{data_dir}/splits.json", "r") as fp:
        phase_used_id = json.load(fp)
    phase_data_dict = {}
    for split in ("train", "val", "test"):
        # 读取文件
        phase_data_dict[split] = {}
        for fname in os.listdir(data_dir):
            if split in fname:
                for kstr in file_kstr:
                    if kstr in fname:
                        with open(f"{data_dir}/{fname}", 'rb') as handle:
                            phase_data_dict[split][kstr] = pickle.load(handle)
        # 分离出EBV 去除治疗
        phase_data_dict[split]["f_ebv"] = {}
        for sid, v in phase_data_dict[split]['f_base'].items():
            phase_data_dict[split]["f_ebv"][sid] = v[6:10]
            phase_data_dict[split]['f_base'][sid] = torch.cat((v[:6], v[10:-3])).nan_to_num(nan=-1)
        # 合并字段信息 f_normal 为所需要的
        phase_data_dict[split]["f_normal"] = {}
        for sid in phase_used_id[split]:
            phase_data_dict[split]["f_normal"][sid] = torch.cat([
                phase_data_dict[split][kstr][sid]
                for kstr in field_column_name_list
            ]).nan_to_num(nan=-1)
        # for kstr in field_column_name_list:
        #     phase_data_dict.pop(kstr, None)
        # 调整ending字段
        for sid in phase_used_id[split]:
            phase_data_dict[split]["f_ending"][sid] = phase_data_dict[split]["f_ending"][sid][range(0, 8, 2)].int()

    tmp_items = [
        (
            torch.cat([
                torch.stack([phase_data_dict[split]["f_normal"][sid] for sid in phase_used_id[split]]),
                torch.stack([phase_data_dict[split]["f_ebv"][sid] for sid in phase_used_id[split]]),
            ], dim=1)[:, filter_arr].numpy(),
            torch.stack([phase_data_dict[split]["f_ending"][sid] for sid in phase_used_id[split]])[:, used_labels].numpy()
        )
        for split in ("train", "val", "test")
    ]
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = tmp_items
    # print(train_x.shape, train_y.shape)
    # print(val_x.shape, val_y.shape)
    # print(test_x.shape, test_y.shape)
    params = dict(
        init_kw_params=dict(
            max_depth=16,  # 8
            learning_rate=0.29717311927223616,  # 0.1
            n_estimators=10000,  # 1000
            verbosity=0,
            silent=None,
            objective='binary:logistic',
            booster='gbtree',
            n_jobs=-1,
            nthread=None,
            gamma=0.17386321871484525,  # 0
            min_child_weight=1.5429408838717562,  # 1
            max_delta_step=0,
            subsample=0.7138127581965952,  # 0.7
            colsample_bytree=0.7398641574747308,  # 1
            colsample_bylevel=0.7938104883852553,  # 1
            colsample_bynode=0.6768131790017465,  # 1
            reg_alpha=0.8124509194020697,  # 0
            reg_lambda=1.1739670203617365,  # 1
            scale_pos_weight=1.3485386082654622,  # 1
            base_score=0.3900530204588142,  # 0.5
            random_state=0,
            seed=None,
        ),
        tr_kw_params=dict(
            early_stopping_rounds=177,  # 40 500
            verbose=0,  # 0, 1e9
        ),
    )

    mg_train_x, mg_train_y = np.concatenate((train_x, val_x)), np.concatenate((train_y, val_y))
    all_test_probs = np.zeros_like(test_y, dtype=np.float32)
    all_importance = []
    all_importance_score = []
    for label_id in range(len(used_labels)):
        print("=" * 58, f" {label_id} ", "=" * 58)
        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
            print("fold {}".format(fold_ + 1))

            clf = XGBClassifier(**params['init_kw_params'])
            clf.fit(
                train_x, train_y, eval_set=[(mg_train_x[trn_idx], mg_train_y[trn_idx, label_id]),
                                            (mg_train_x[val_idx], mg_train_y[val_idx, label_id])],
                **params['tr_kw_params'],
            )

            probs = clf.predict_proba(test_x)[:, 1]
            all_importance.append(clf.feature_importances_)
            all_importance_score.append(TFC.binary_auroc(torch.from_numpy(probs[:, None]), torch.from_numpy(test_y)))
            all_test_probs[:, label_id] += probs / folds.n_splits
        print("=" * 120)
    print(all_importance_score)
    print(TFC.binary_auroc(torch.from_numpy(all_test_probs), torch.from_numpy(test_y)))

    all_importance = np.stack(all_importance)
    min_v = all_importance.min(axis=1, keepdims=True)
    all_importance = (all_importance - min_v) / (all_importance.max(axis=1, keepdims=True) - min_v)
    all_importance = (all_importance - all_importance.mean(axis=1, keepdims=True)) / all_importance.std(axis=1, keepdims=True)
    all_importance_score = np.stack(all_importance_score)
    feat_importance = (all_importance_score / all_importance_score.sum()) @ all_importance
    min_v = feat_importance.min()
    feat_importance = (feat_importance - min_v) / (feat_importance.max() - min_v)

    field_name_dict = get_field_name_dict([("used", ("f_normal", "f_ebv"))])
    used_field_names = np.array(field_name_dict["used"])

    sort_ind = np.argsort(feat_importance)[::-1].astype(np.int32)[: consider_count]
    field_must_ind = np.array(get_field_ind(field_name_dict, "used", field_must))
    sort_ind = sort_ind[~np.in1d(filter_arr[sort_ind], field_must_ind)][:consider_count-len(field_must_ind)]
    # print(sort_ind)
    # print(feat_importance[sort_ind])
    # print(used_field_names[sort_ind])

    sort_ind = np.sort(sort_ind)
    # print(sort_ind)
    # print(feat_importance[sort_ind])
    # print(used_field_names[filter_arr][sort_ind])
    consider_ind = np.array(get_field_ind(field_name_dict, "used", field_must + used_field_names[filter_arr][sort_ind].tolist()))
    consider_ind = np.sort(consider_ind)
    print(consider_ind)
    print(used_field_names[consider_ind])
# run_xgb()  # 7254 7143


# from vilt.utils.write_modmis import make_modmis_image_pickles
# make_modmis_image_pickles("F:/鼻咽癌数据集",
#                         "E:/PyTest/ModalMissing/raw_data/modmis/fields",
#                         "F:/鼻咽癌数据集/compressed3",
#                         False, 9999, image_size=224).
# show_roc("../val_roc_modmis_bin_213.pkl")
# show_roc("../train_roc_modmis_bin_1813.pkl")
# show_roc2("../kaggle_results/mm-vilt_0/version_36/roc_records")
# show_roc3("../kaggle_results/roc_records/v1")
# check_torch_n_cuda()

a = torch.randint(0, 2, [10,])
b = a.nonzero()[:, 0].tolist()
# print(a)
# print(a.shape)
# print(b)
a = torch.full([10,], 2)
print(a)
