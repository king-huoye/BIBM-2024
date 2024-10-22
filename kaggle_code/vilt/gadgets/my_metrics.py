import os

import torch
from torchmetrics.functional import f1_score, auroc
import torchmetrics.functional.classification as TFC
# from pytorch_lightning.torchmetrics import Metric
# import torchmetrics
from torchmetrics import Metric
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


class ROC_Drawer:
    def __init__(self) -> None:
        pass

    def plot_label_roc(self, fpr, tpr, auc_score):
        fpr, tpr = fpr.cpu(), tpr.cpu()
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'(AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def get_weighted(self, input_vector, size):
        if input_vector.dtype != torch.int64:
            input_vector = input_vector.type(torch.int64)
        frequency_vector = torch.zeros(size, dtype=torch.int64, device=input_vector.device)
        frequency_vector.scatter_add_(0, input_vector, torch.ones_like(input_vector))
        return frequency_vector / frequency_vector.sum()

    def draw_mlb(self, probs, targets, label_count, show=True, verbose=True):
        if label_count == 1:
            auc_scores = TFC.binary_auroc(probs, targets).unsqueeze(0)
            fprs, tprs, _ = TFC.binary_roc(probs, targets)
            fprs, tprs = fprs.unsqueeze(0), tprs.unsqueeze(0)
        else:
            auc_scores = TFC.multilabel_auroc(probs, targets, num_labels=label_count, average=None)
            fprs, tprs, _ = TFC.multilabel_roc(probs, targets, num_labels=label_count)
        if show:
            for fpr, tpr, auc_score in zip(fprs, tprs, auc_scores):
                self.plot_label_roc(fpr, tpr, auc_score)
        if verbose:
            for prob, tg, fpr, tpr, auc_score in zip(probs.transpose(), targets.transpose(), fprs, tprs, auc_scores):
                print("prob: ", prob)
                print("targets: ", tg)
                print("weighted auc: ", auc_score)
                print("fpr: ", fpr, "tpr: ", tpr)
        return fprs, tprs, auc_scores

    def draw_mlb_mcls(self, logits, targets, label_class_count, show=True, verbose=False):
        roc_curves_data = []
        for lg, tg, lcc in zip(logits, targets, label_class_count):
            auc_score = TFC.multiclass_auroc(lg, tg, num_classes=lcc, average="weighted")
            fpr, tpr, _ = TFC.multiclass_roc(lg, tg, num_classes=lcc, average=None)
            weight_arr = self.get_weighted(tg, lcc)
            max_length = max(len(seq) for seq in fpr)
            fpr = weight_arr @ torch.stack([torch.cat((item, item[-1].repeat(max_length - len(item)))) for item in fpr])
            tpr = weight_arr @ torch.stack([torch.cat((item, item[-1].repeat(max_length - len(item)))) for item in tpr])
            roc_curves_data.append((fpr, tpr, auc_score))
            if show:
                self.plot_label_roc(fpr, tpr, auc_score)
            if verbose:
                print("logits: ", lg)
                print("targets: ", tg)
                print("weighted auc: ", auc_score)
                print("fpr: ", fpr, "tpr: ", tpr)
        return roc_curves_data


class Reg2Cls_Base(Metric):
    def __init__(self, label_ranges: list, label_intervals: list, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.label_ranges = label_ranges
        self.label_intervals = label_intervals

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.label_ranges[0].device != self.correct.device:
            self.label_ranges = [item.to(self.correct.device) for item in self.label_ranges]
        preds, targets = (
            preds.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.preds.append(preds)
        self.targets.append(targets)

    # 思路：
    # 将连续值分区，除了第1, 3, 5, 7, 9列是二分类使用[0, 1]两个区域外，其它按间隔3进行分区，每个分区即一个类
    # 分完区后，先对target（结果）以类似四舍五入的方式划入各区域，比如7.2划入区域6且对应类别2(6/3==2)，8.1划入区域9且对应类别3(9/3==3)
    # 然后将计算预测值与各分区的距离的倒数作为逻辑值（logits），特别地，到target值对应的区的距离以预测值和target值的距离代替
    # 最后将逻辑值归一化
    # 由于torchmetrics包只有多类别ROC计算（它多标签ROC中每个标签只能是二分类）
    # 于是这里对多个独立的标签分别用多类别ROC计算（至于后续是否要合并多个标签的ROC曲线以及AUC的值作为整体评估，待定，但不难实现）
    # 参数说明：
    # ori_preds: 模型的预测输出，(B, L)
    # ori_targets: 预测目标值，(B, L)
    # label_ranges: 标签的划分区域，[range1, range2, ...L个...]
    # label_intervals: 标签划分区域时采用的间隔，[interval1, interval2, ...L个...]
    def deal_reg(self, ori_preds: torch.Tensor, ori_targets: torch.Tensor, verbose=False):
        if ori_targets != ori_preds.dtype:  # float16够用  要改 float32 需要到 vilt_utils 和 modmis_datasets 里改
            ori_targets = ori_targets.type(ori_preds.dtype)
        batch_size, label_count = ori_preds.shape
        label_class_count = [len(lr) for lr in self.label_ranges]
        targets = [torch.round(ori_targets[:, i] / self.label_intervals[i]).type(torch.int32) for i in range(label_count)]
        logits = [1 / (torch.abs(self.label_ranges[i] - ori_preds[:, i].unsqueeze(-1).expand(batch_size, label_class_count[i])) + 1e-5)
                for i in range(label_count)]
        for i, (lg, tg) in enumerate(zip(logits, targets)):
            lg[range(batch_size), tg] = 1 / (torch.abs(ori_targets[:, i] - ori_preds[:, i]) + 1e-5)
        logits = [lg / torch.sum(lg, dim=-1, keepdim=True) for lg in logits]
        if verbose:
            print("logit shapes: ", [lg.shape for lg in logits])
            print("target shapes: ", [tg.shape for tg in targets])
            print("label class count: ", label_class_count)
        return logits, targets, label_class_count

    def compute(self):
        raise NotImplementedError("return compute results.")


class MM_Composite_Reg(Reg2Cls_Base):
    def __init__(self, label_ranges: list, label_intervals: list, dist_sync_on_step=False):
        super().__init__(label_ranges, label_intervals, dist_sync_on_step)
        self.used_metrics_fun = {
            "Accuracy": TFC.multiclass_accuracy,
            "Auc": TFC.multiclass_auroc,
            "F1_score": TFC.multiclass_f1_score,
            "Precision": TFC.multiclass_precision,
            "Recall": TFC.multiclass_recall,
        }
        self.roc_drawer = ROC_Drawer()

    def draw_roc(self, show=True, verbose=False):
        return self.roc_drawer.draw_mlb_mcls(
            *self.deal_reg(torch.cat(self.preds), torch.cat(self.targets), verbose), show, verbose)

    def compute(self):
        verbose = False
        deal_reg_ret = self.deal_reg(torch.cat(self.preds), torch.cat(self.targets), verbose=verbose)

        ret = {}
        for metric_name, metric_fun in self.used_metrics_fun.items():
            metric_values = []
            for lg, tg, cc in zip(*deal_reg_ret):
                metric_values.append(metric_fun(lg, tg, num_classes=cc, average="weighted"))
                if verbose:
                    print("logits: ", lg)
                    print("targets: ", tg)
                    print(f"weighted {metric_name}: ", metric_values[-1])
            ret[metric_name] = torch.mean(torch.stack(metric_values))
        return ret


class MM_Composite_Cls(Metric):
    def __init__(self, label_class_count: list, label_intervals: list, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.lcc = label_class_count
        self.label_intervals = label_intervals
        self.used_metrics_fun = {
            "Accuracy": TFC.multiclass_accuracy,
            "Auc": TFC.multiclass_auroc,
            "F1_score": TFC.multiclass_f1_score,
            "Precision": TFC.multiclass_precision,
            "Recall": TFC.multiclass_recall,
        }
        self.roc_drawer = ROC_Drawer()

    def update(self, logit_list, targets):
        self.logits.append([logit.detach().to(self.correct.device) for logit in logit_list])
        self.targets.append(targets.detach().to(self.correct.device))

    def draw_roc(self, show=True, verbose=False):
        all_logits = [torch.cat([logit[i] for logit in self.logits]) for i in range(len(self.logits[0]))]
        all_targets = [torch.round(self.targets[:, i] / self.label_intervals[i]).type(torch.int32)
                       for i in range(len(self.lcc))]
        return self.roc_drawer.draw_mlb_mcls(all_logits, all_targets, self.lcc, show, verbose)

    def compute(self):
        verbose = False
        all_logits = [torch.cat([logit[i] for logit in self.logits]) for i in range(len(self.logits[0]))]
        all_targets = [torch.round(self.targets[:, i] / self.label_intervals[i]).type(torch.int32)
                       for i in range(len(self.lcc))]

        ret = {}
        for metric_name, metric_fun in self.used_metrics_fun.items():
            metric_values = []
            for lg, tg, cc in zip((all_logits, all_targets, self.lcc)):
                metric_values.append(metric_fun(lg, tg, num_classes=cc, average="weighted"))
                if verbose:
                    print("logits: ", lg)
                    print("targets: ", tg)
                    print(f"weighted {metric_name}: ", metric_values[-1])
            ret[metric_name] = torch.mean(torch.stack(metric_values))
        return ret


class MM_Composite_Bin(Metric):
    def __init__(self, label_count: int, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.label_count = label_count
        if label_count == 1:
            self.used_metrics_fun = {
                "Accuracy": TFC.binary_accuracy,
                "Auc": TFC.binary_auroc,
                "F1_score": TFC.binary_f1_score,
                "Precision": TFC.binary_precision,
                "Recall": TFC.binary_recall,
            }
        else:
            self.used_metrics_fun = {
                "Accuracy": TFC.multilabel_accuracy,
                "Auc": TFC.multilabel_auroc,
                "F1_score": TFC.multilabel_f1_score,
                "Precision": TFC.multilabel_precision,
                "Recall": TFC.multilabel_recall,
            }
        self.roc_drawer = ROC_Drawer()

    def update(self, logits, targets):
        self.logits.append(logits.detach().to(self.correct.device))
        self.targets.append(targets.detach().to(self.correct.device))

    def draw_roc(self, show=True, verbose=False):
        return self.roc_drawer.draw_mlb(torch.sigmoid(torch.cat(self.logits)),
                                        torch.cat(self.targets), self.label_count, show, verbose)

    def compute(self):
        all_probs = torch.sigmoid(torch.cat(self.logits))
        all_targets = torch.cat(self.targets)

        ret = {}
        if self.label_count == 1:
            for metric_name, metric_fun in self.used_metrics_fun.items():
                ret[metric_name] = metric_fun(all_probs, all_targets)
        else:
            for metric_name, metric_fun in self.used_metrics_fun.items():
                ret[metric_name] = metric_fun(all_probs, all_targets, num_labels=self.label_count, average="weighted")
        return ret


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1) > 1:
            preds = logits.argmax(dim=-1)  # 多分类
        else:
            preds = (torch.sigmoid(logits) > 0.5).long()  # 二分类

        preds = preds[target != -100]  # 将-100作为缺省值（可能） 参照dataset模块里的dict_batch[f"{f_key}_labels"]取值
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class AUROC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()

        if all_logits.size(-1) > 1:
            all_logits = torch.softmax(all_logits, dim=1)
            AUROC = auroc(all_logits, all_targets, num_classes=2)
        else:
            all_logits = torch.sigmoid(all_logits)
            AUROC = auroc(all_logits, all_targets)

        return AUROC


class F1_Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits)
        F1_Micro = f1_score(all_logits, all_targets, average='micro')
        F1_Macro = f1_score(all_logits, all_targets, average='macro', num_classes=23)
        F1_Samples = f1_score(all_logits, all_targets, average='samples')
        F1_Weighted = f1_score(all_logits, all_targets, average='weighted', num_classes=23)
        return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)


class check(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits).long()
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits.long()
            all_targets = self.targets.long()

        mislead = all_logits ^ all_targets
        accuracy = mislead.sum(dim=0)
        return accuracy


# 计算平均损失 该损失是优化网络的目标损失 譬如是交叉熵或均方误差之类
class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class Scalar2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, num):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)

        self.scalar += scalar
        self.total += num

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


# class FeatsShower:
#     color = ['#e3847f', '#debc63', '#a5db52', '#70e177', '#68debe', '#82bfe4', '#736de0', '#be68de', '#e37fbc',
#              '#F0BDBA', '#EEDBAC', '#D0ECA6', '#B9F1BD', '#B7EFE0', '#B7DAEF', '#BFBDF1', '#E0B7EF', '#F5D3E8', ]
#     markers = {0: "o", 1: "o", 2: "o", 3: "o", 4: "o", 5: "o", 6: "o", 7: "o",
#                8: "o", 9: ".", 10: ".", 11: ".", 12: ".", 13: ".", 14: ".", 15: ".", 16: ".", 17: "."}
#
#     def __init__(self, save_dir: str) -> None:
#         super().__init__()
#         self.save_dir = f"{save_dir}/T-SNE"
#         self.tsne = TSNE(n_components=2, init='pca', random_state=0)
#         self.epoch_feats = []
#         self.epoch_labels = []
#
#     def update(self, feats: torch.Tensor, labels: torch.Tensor):
#         self.epoch_feats.append(feats)
#         self.epoch_labels.append(labels)
#
#     def reset(self):
#         self.epoch_feats = []
#
#     def plot_tsne(self, phase_epoch: str) -> None:
#         '''
#         features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
#         label:(N) 有N个标签
#         '''
#
#         features = torch.cat(self.epoch_feats).cpu().numpy()
#         labels = torch.cat(self.epoch_labels)[:, 0].cpu().numpy()
#         tsne_features = self.tsne.fit_transform(features)  # 将特征使用PCA降维至2维
#
#         fig, axes = plt.subplots(1, 1, figsize=(16, 12))
#
#         df = pd.DataFrame(columns=["y", "comp-1", "comp-2"])
#         df["y"] = labels.astype(int)
#         df["comp-1"] = tsne_features[:, 0]
#         df["comp-2"] = tsne_features[:, 1]
#         sns.set_style("white")
#         sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                         style='y',
#                         palette=self.color,
#                         markers=self.markers,
#                         data=df, alpha=1, legend=False)
#         # sns.get_legend().remove()
#
#         os.makedirs(self.save_dir, exist_ok=True)
#         fig.savefig(f"{self.save_dir}/{phase_epoch}.png", dpi=300)

# class FeatsShower:
#     color = ['#e3847f', '#debc63', '#a5db52', '#70e177', '#68debe', '#82bfe4', '#736de0', '#be68de', '#e37fbc',
#              '#F0BDBA', '#EEDBAC', '#D0ECA6', '#B9F1BD', '#B7EFE0', '#B7DAEF', '#BFBDF1', '#E0B7EF', '#F5D3E8', ]
#     markers = {0: "o", 1: "o", 2: "o", 3: "o", 4: "o", 5: "o", 6: "o", 7: "o",
#                8: "o", 9: ".", 10: ".", 11: ".", 12: ".", 13: ".", 14: ".", 15: ".", 16: ".", 17: "."}
#
#     def __init__(self, save_dir: str) -> None:
#         super().__init__()
#         self.save_dir = f"{save_dir}/T-SNE"
#         self.tsne = TSNE(n_components=3, init='pca', random_state=0)  # 设置 n_components 为 3
#         self.epoch_feats = []
#         self.epoch_labels = []
#
#     def update(self, feats: torch.Tensor, labels: torch.Tensor):
#         self.epoch_feats.append(feats)
#         self.epoch_labels.append(labels)
#
#     def reset(self):
#         self.epoch_feats = []
#
#     def plot_tsne(self, phase_epoch: str) -> None:
#         '''
#         features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
#         label:(N) 有N个标签
#         '''
#
#         features = torch.cat(self.epoch_feats).cpu().numpy()
#         labels = torch.cat(self.epoch_labels)[:, 0].cpu().numpy()
#         tsne_features = self.tsne.fit_transform(features)  # 将特征使用PCA降维至3维
#
#         fig = plt.figure(figsize=(16, 12))
#         ax = fig.add_subplot(111, projection='3d')
#
#         df = pd.DataFrame(columns=["y", "comp-1", "comp-2", "comp-3"])
#         df["y"] = labels.astype(int)
#         df["comp-1"] = tsne_features[:, 0]
#         df["comp-2"] = tsne_features[:, 1]
#         df["comp-3"] = tsne_features[:, 2]
#
#         colors = [self.color[label] for label in df["y"]]
#         markers = [self.markers[label] for label in df["y"]]
#
#         for color, marker, (x, y, z) in zip(colors, markers, zip(df["comp-1"], df["comp-2"], df["comp-3"])):
#             ax.scatter(x, y, z, c=color, marker=marker)
#
#         ax.set_xlabel('Component 1')
#         ax.set_ylabel('Component 2')
#         ax.set_zlabel('Component 3')
#
#         os.makedirs(self.save_dir, exist_ok=True)
#         plt.savefig(f"{self.save_dir}/{phase_epoch}.png", dpi=300)
#         plt.show()

class FeatsShower:
    color = ['#e3847f', '#debc63', '#a5db52', '#70e177', '#68debe', '#82bfe4', '#736de0', '#be68de', '#e37fbc',
             '#F0BDBA', '#EEDBAC', '#D0ECA6', '#B9F1BD', '#B7EFE0', '#B7DAEF', '#BFBDF1', '#E0B7EF', '#F5D3E8', ]
    markers = {0: "o", 1: "o", 2: "o", 3: "o", 4: "o", 5: "o", 6: "o", 7: "o",
               8: "o", 9: ".", 10: ".", 11: ".", 12: ".", 13: ".", 14: ".", 15: ".", 16: ".", 17: "."}

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.tsne = TSNE(n_components=2, init='pca', random_state=0)  # 设置 n_components 为 2
        self.epoch_feats = []
        self.epoch_labels = []
        self.field_tokens = []

    def update(self, feats: torch.Tensor, labels: torch.Tensor, field_tokens: torch.Tensor):
        self.epoch_feats.append(feats)
        self.epoch_labels.append(labels)
        self.field_tokens.append(field_tokens)

    def reset(self):
        self.epoch_feats = []
        self.epoch_labels = []
        self.field_tokens = []

    # def plot_tsne(self, phase_epoch: str) -> None:
    #     '''
    #     features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    #     label:(N) 有N个标签
    #     '''
    #
    #     features = torch.cat(self.epoch_feats).cpu().numpy()
    #     labels = torch.cat(self.epoch_labels)[:, 0].cpu().numpy()
    #     tsne_features = self.tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    #
    #     fig, ax = plt.subplots(figsize=(16, 12))
    #
    #     df = pd.DataFrame(columns=["y", "comp-1", "comp-2"])
    #     df["y"] = labels.astype(int)
    #     df["comp-1"] = tsne_features[:, 0]
    #     df["comp-2"] = tsne_features[:, 1]
    #
    #     colors = [self.color[label] for label in df["y"]]
    #     markers = [self.markers[label] for label in df["y"]]
    #
    #     for color, marker, (x, y) in zip(colors, markers, zip(df["comp-1"], df["comp-2"])):
    #         ax.scatter(x, y, c=color, marker=marker)
    #
    #     ax.set_xlabel('Component 1')
    #     ax.set_ylabel('Component 2')
    #
    #     ax.axis('off')
    #     os.makedirs(self.save_dir, exist_ok=True)
    #     plt.savefig(f"{self.save_dir}/{phase_epoch}.png", dpi=300)
    #     plt.show()
    def plot_tsne(self, phase_epoch: str) -> None:
        features = torch.cat(self.epoch_feats).cpu().numpy()
        labels = torch.cat(self.epoch_labels)[:, 0].cpu().numpy()
        tsne_features = self.tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(16, 12))

        df = pd.DataFrame(columns=["y", "comp-1", "comp-2"])
        df["y"] = labels.astype(int)
        df["comp-1"] = tsne_features[:, 0]
        df["comp-2"] = tsne_features[:, 1]

        # 使用红色和绿色
        colors = ['red' if label == 1 else 'green' for label in df["y"]]

        ax.scatter(df["comp-1"], df["comp-2"], c=colors)

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        ax.axis('off')
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(f"{self.save_dir}/T-SNE/{phase_epoch}.png", dpi=300)
        plt.close()  # 关闭图形，避免在非交互环境中显示

    def plot_field_rela_heatmap(self, phase_epoch: str) -> None:
        batch_hm = []  # (N,B,20,20)
        for ft in self.field_tokens:
            ft = ft.cpu().numpy()
            batch_hm.append(ft @ ft.transpose(1,2))
        hm = np.stack(batch_hm)
        print(111, hm.shape)
        hm = np.mean(hm, axis=0)
        hm = np.mean(hm, axis=0)
        print(222, hm.shape)

        # 绘制热力图
        plt.figure(figsize=(20, 20))  # 可以根据需要调整图形大小
        sns.heatmap(hm, annot=True, cmap='coolwarm', center=0, fmt=".2f")
        plt.title('heatmap')
        plt.show()

        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(f"{self.save_dir}/corr_heatmap/{phase_epoch}.png", dpi=300)
        plt.close()  # 关闭图形，避免在非交互环境中显示
