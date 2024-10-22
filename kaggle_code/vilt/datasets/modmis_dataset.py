import random

import torch
import os
import pickle
import json
import zlib


class MODMISDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_keys: list,
            image_size: int,
            field_column_name_list: list,
            draw_false_image=0,
            draw_false_field=0,
            image_only=False,
            split="",
            missing_info={},
            used_labels=(0, 1, 2, 3)
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["modmis_train"]
        elif split == "val":
            names = ["modmis_val"]
        else:
            names = ["modmis_test"]

        assert len(transform_keys) >= 1
        super().__init__()

        self.image_size = image_size
        self.field_column_name_list = field_column_name_list
        self.names = names
        self.draw_false_image = draw_false_image
        self.draw_false_field = draw_false_field
        self.image_only = image_only
    
        self.data_dir = data_dir
        self.used_id = []
        self.phase_data_dict = []

        # 读取文件
        with open(f"/tmp/pycharm_project_499/datasets/modmis/pickles/datasets/splits.json", "r") as fp:
            used_id = json.load(fp)[split]
            # used_id = torch.tensor(used_id, device='cuda').long()

        file_kstr = ["f_ending", "f_base", "f_conventional", "f_special", "f_blood", "f_complication",
                     "T1_image", "T1C_image", "T2_image"]
        #6个表格+3种图像的模态
        phase_data_dict = {}
        for fname in os.listdir(data_dir):
            if split in fname:
                for kstr in file_kstr:
                    if kstr in fname:
                        with open(f"{data_dir}/{fname}", 'rb') as handle:
                            phase_data_dict[kstr] = pickle.load(handle)
        with open(f"{data_dir}/mean_std.pkl", 'rb') as handle:
            mean_std_dict = pickle.load(handle)
        # # 去除图像的label 节省空间（当前尚未用到）
        # for sid in used_id:
        #     for kstr in ("T1_image", "T1C_image", "T2_image"):
        #         phase_data_dict[kstr][sid] = (phase_data_dict[kstr][sid][0], None)
        # 分离出EBV 去除治疗
        phase_data_dict["f_ebv"] = {}
        for sid, v in phase_data_dict['f_base'].items():
            # print('sid',sid)
            phase_data_dict["f_ebv"][sid] = v[6:10]
            phase_data_dict['f_base'][sid] = torch.cat((v[:6], v[10:-3])).nan_to_num(nan=-1)
        # 合并字段信息 f_normal 为所需要的 然后再挑选 20 个筛选过的特征（后续将使用特征选择模块代替）
        # selected_feature_ind = torch.tensor([0, 1, 6, 7, 8, 9, 6, 141, 142, 143, 144, 150, 154, 159, 161, 163]).int()
        selected_feature_ind = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 96, 141, 142, 143, 144, 146, 148, 150, 157, 159, 161]).long()
        phase_data_dict["f_normal"] = {}
        for sid in used_id:
            # print('sid',sid)
            phase_data_dict["f_normal"][sid] = torch.cat([
                phase_data_dict[kstr][sid]
                for kstr in field_column_name_list
            ]).nan_to_num(nan=-1)[selected_feature_ind]
        # for kstr in field_column_name_list:
        #     phase_data_dict.pop(kstr, None)
        # 调整ending字段
        for sid in used_id:
            phase_data_dict["f_ending"][sid] = phase_data_dict["f_ending"][sid][range(0, 8, 2)][used_labels].int()

        self.used_id = used_id
        self.phase_data_dict = phase_data_dict
        self.mean_std_dict = mean_std_dict

        # 设置缺失比率
        self.simulate_missing = missing_info['simulate_missing']  # False
        missing_ratio = missing_info['ratio'][split]  # 0.7
        mratio = str(missing_ratio).replace('.', '')  # 0.7 -> 07
        missing_type = missing_info['type'][split]  # both
        both_ratio = missing_info['both_ratio']  # 0.5
        restrict_modal_count = missing_info['restrict_modal_count']
        self.restrict_modal_count = restrict_modal_count
        missing_type_code = missing_info['missing_type_code']
        self.missing_type_code = missing_type_code
        mix_ratios = missing_info['mix_ratios'][split]
        missing_table_root = missing_info['missing_table_root']
        missing_table_name, missing_table_path = "", ""
        if missing_table_root is not None:
            missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}.pt'
            # "./datasets/missing_tables/modmis_missing_both_07.pt"
            missing_table_path = os.path.join(missing_table_root, missing_table_name)

        # 通过缺失的比率设置 missing_table
        # missing_table是缺失状态tensor 0代表无缺失 1代表缺字段 2代表缺图像
        total_num = len(self.used_id)

        if missing_table_root is not None and os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros((total_num, len(missing_type_code) + 1), dtype=torch.int8)
            if missing_type == 'mix':
                sample_hash = torch.full([total_num,], restrict_modal_count)
                for missing_type, missing_ratio in mix_ratios.items():
                    missing_index = random.sample(sample_hash.nonzero()[:, 0].tolist(), int(total_num * missing_ratio))
                    missing_table[missing_index, missing_type_code[missing_type]] = 1
                    sample_hash[missing_index] -= 1
            elif missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num * missing_ratio))
                if missing_type != 'both':
                    missing_table[missing_index, missing_type_code[missing_type]] = 1
                else:
                    missing_table[missing_index, 1] = 1
                    missing_index_both = random.sample(missing_index, int(len(missing_index) * both_ratio))
                    missing_table[missing_index_both, 1] = 0
                    missing_table[missing_index_both, 2] = 1

            if missing_table_root is not None:
                    cache_root = missing_info['cache_root']
                    missing_table_path = os.path.join(cache_root, missing_table_name)
                    os.makedirs(cache_root, exist_ok=True)
                    torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table

    def get_info(self):
        data_info = {
            "fields_keys": ["f_normal", "f_ebv"],  # 两种表格信息模态
            "image_keys": ["T1_image", "T1C_image", "T2_image"],  # 3种图像模态
        }
        data_info["fields_len"] = {
            k: len(self.phase_data_dict[k][self.used_id[0]]) for k in data_info["fields_keys"]
        }
        data_info["modal_count"] = len(data_info["fields_keys"]) + len(data_info["image_keys"])
        return data_info

    def __len__(self):
        return len(self.used_id)

    def get_raw_image_label(self, index, image_key="T1_image"):
        return self.phase_data_dict[image_key][self.used_id[index]]

    def get_image(self, index, image_key="T1_image"):
        return {
            image_key: self.get_raw_image_label(index, image_key)[0],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="T1_image"):
        return {
            f"false_{image_key}_{rep}": self.get_raw_image_label(random.randint(0, len(self) - 1), image_key)[0]
        }

    def get_raw_field(self, index, field_key="f_normal"):
        return self.phase_data_dict[field_key][self.used_id[index]]

    def get_field(self, index, field_key="f_normal"):
        return {
            field_key: self.phase_data_dict[field_key][self.used_id[index]],
            "raw_index": index,
        }

    def get_false_field(self, rep, field_key="f_normal"):
        return {f"false_{field_key}_{rep}": self.phase_data_dict[field_key][self.used_id[random.randint(0, len(self) - 1)]]}

    def get_suite(self, index):
        result = False
        while result is not True:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    for field_key in self.field_column_name_list:
                        ret.update(self.get_field(index, field_key))

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_field):
                    for field_key in self.field_column_name_list:
                        ret.update(self.get_false_field(i, field_key))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.table) - 1)
        return ret

    # 模拟源程序中的行为，主要获取input_ids和attention_mask
    # input_ids使用
    # https://blog.csdn.net/weixin_44219178/article/details/121991171
    def collate(self, batch):
        batch_size = len(batch)
        dict_batch = {k: [dic[k] for dic in batch] for k in batch[0].keys()}  # 按样本行排改成按关键字列排

        # ===== image处理部分
        # 含有'image'字符子串的所有关键字（原本只有'image'一个），原意为取出所有image样本
        img_keys = [k for k in dict_batch.keys() if "image" in k]
        for img_key in img_keys:
            images = [items[0] for items in dict_batch[img_key]]
            image_labels = None if dict_batch[img_key][0][1] is None else [items[1] for items in dict_batch[img_key]]
            new_images = torch.zeros(batch_size, *[max([img.shape[i] for img in images]) for i in range(3)])  # 改为全1
            new_image_labels = None if image_labels is None else torch.zeros(new_images.shape, dtype=torch.uint8)
            for bi in range(batch_size):
                orig = images[bi]
                new_images[bi, :orig.shape[0], :orig.shape[1], :orig.shape[2]] = orig
                if new_image_labels is not None:
                    orig = image_labels[bi]
                    new_image_labels[bi, :orig.shape[0], :orig.shape[1], :orig.shape[2]] = orig

            dict_batch[img_key] = (new_images, new_image_labels,
                                   self.mean_std_dict[img_key]["normalizer"][new_images.shape[1]])

        # ===== field处理部分
        field_keys = [k for k in dict_batch.keys() if "f_" in k]
        # encodings to mlms ... (need to be implemented)
        # fields = [[d[0] for d in dict_batch[f_key]] for f_key in field_keys]
        # encodings = [[d[1] for d in dict_batch[f_key]] for f_key in field_keys]

        for f_key in field_keys:
            mlm_ids, mlm_lables = (None, None)  # need to be implemented according to situation
            fields = torch.stack([d[0] for d in dict_batch[f_key]])
            attention_mask = torch.stack([d[1] for d in dict_batch[f_key]])

            dict_batch[f_key] = fields
            dict_batch[f"{f_key}_labels"] = torch.full_like(fields, -100)
            dict_batch[f"{f_key}_masks"] = attention_mask
            dict_batch[f"{f_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{f_key}_labels_mlm"] = mlm_lables

        return dict_batch

    def __getitem__(self, index):
        # print('modmis_dataset.__getitem__ index: ', index)
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        # 当前是模拟缺失的情况，且当前的index样本设置的无缺失，则模拟缺失类型随机挑选
        simulate_missing_type = torch.zeros(len(self.missing_type_code) + 1, dtype=torch.int8)
        if self.split == 'train' and self.simulate_missing and sum(self.missing_table[index]) == 0:
            simulate_missing_type[random.sample(range(1, len(self.missing_type_code)+1), random.randint(0, self.restrict_modal_count))] = 1

        missing_type_arr = self.missing_table[index] + simulate_missing_type
        ret_dict = {
            "label": self.get_raw_field(index, "f_ending"),
            "missing_type_arr": missing_type_arr,
        }
        for modal_name, missing_code in self.missing_type_code.items():
            if modal_name.startswith("f_"):
                field = self.get_raw_field(index, modal_name)
                if missing_type_arr[missing_code] == 1:
                    ret_dict[modal_name] = (torch.zeros_like(field), torch.zeros_like(field))
                else:
                    ret_dict[modal_name] = (field, torch.ones_like(field))
            else:
                image_label = self.get_raw_image_label(index, modal_name)
                image_label = (pickle.loads(zlib.decompress(image_label[0])), image_label[1])
                if missing_type_arr[missing_code] == 1:
                    ret_dict[modal_name] = (
                        torch.zeros_like(image_label[0]),
                        None if image_label[1] is None else torch.zeros_like(image_label[0]),
                    )
                else:
                    ret_dict[modal_name] = (
                        image_label[0],
                        None if image_label[1] is None else pickle.loads(zlib.decompress(image_label[1]))
                    )

        return ret_dict
