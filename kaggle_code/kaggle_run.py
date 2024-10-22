import socket
import sys
import shutil
import os
from pytorch_lightning.callbacks.progress import ProgressBar
run_in_local = True
# os.environ['SACRED_IGNORE_ADD_CONFIG'] = '1'


def clear_directory(path_to_clear):
    try:
        shutil.rmtree(path_to_clear)
        print(f"clear disk {path_to_clear} done.")
    except FileNotFoundError:
        print(f"The directory {path_to_clear} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to remove the directory: {e}")


if run_in_local:
    print("run in local.\n")
    results_dir = f"/tmp/pycharm_project_499/kaggle_results"
else:
    print("run in remote.\n")
    results_dir = "../working"
    for dir_root in ["../input/modalmissing", "../input"]:
        if dir_root not in sys.path:
            sys.path.append(dir_root)
    print(sys.path)
    clear_directory(results_dir)  # 清空输出目录


# import logging
# logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="kaggle_code.vilt.modules.vision_transformer_prompts",
                        message="^Overwriting vit[a-zA-Z0-9_]+ in registry with")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="pytorch_lightning",
                        message="^The dataloader, [a-zA-Z0-9_ ]+, "+
                        "does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument`")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['SACRED_IGNORE_ADD_CONFIG'] = '1'
import copy
import pytorch_lightning as pl
from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.modmis_datamodule import MODMISDataModule
# from pytorch_lightning.profiler import SimpleProfiler


# 有一些权重，暂时用不了，要跑重新来跑过,114跑的是缺少T1C的
#115跑T1
#116跑表格和图像
@ex.named_config
def local_task_modmis():
    # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    batch_size = 64  # 64

    # cache dir
    cache_root = f"{results_dir}/modmis_cache/"

    # missing modality config
    missing_ratio = {'train': 0.2, 'val': 0.2, 'test': 0.2}  # missing_type设为mix时 missing_ratio无效 改为设置mix_ratios
    missing_type = {'train': 'mix', 'val': 'mix', 'test': 'mix'}  # f_normal, f_ebv, both, mix
    # missing_type设为both的时候才会影响both_ratio
    both_ratio = 0.2  # 0.2
    missing_table_root = None
    simulate_missing = False  # False
    # follows are added for the setting of missing type 'mix'
    restrict_modal_count = 1  # restrict the count of missing modal in one sample
    missing_code_type = ['none', 'f_normal', 'f_ebv', 'T1_image', 'T1C_image', 'T2_image']  # missing code-type mapper
    missing_type_code = {  # missing type-code mapper
        modal_name: i for i, modal_name in enumerate(missing_code_type) if i > 0
    }
    mix_ratios = {  # missing mix ratio
        'train': {'T2_image': 0.8, },
        'val': {'T2_image': 0.8, },
        'test': {'T2_image': 0.8, },
    }
    # 还要把缺少T1C、T2跑一下就差不多

    # missing_aware_prompts config
    prompt_type = 'input'  # 'input'
    prompt_length = 16  # 16
    learnt_p = True  # True
    prompt_layers = [0, 1, 2, 3, 4, 5]  # [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = True  # True

    # Transformer Setting
    vit = "vit_base_patch32_224_in21k"  # vit_base_patch32_384 vit_base_patch16_224 vit_base_patch32_224_in21k
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4  # 4
    drop_rate = 0.1  # 0.1

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 224 if "224" in vit else 384
    max_image_len = -1
    patch_size = 16 if "patch16" in vit else 32
    draw_false_image = 1
    image_only = False

    # PL Trainer Setting
    finetune_first = False  # False

    # Modmis task data using
    modmis_label_used = (0,)
    # ['f_base', 'f_conventional', 'f_special', 'f_blood', 'f_complication']
    modmis_field_used = ['f_base', 'f_conventional', 'f_special', 'f_blood', 'f_complication']

    # Loss function Setting
    used_loss = "focal"  # "bcewl", "focal"  # "bcewl"
    bcewl_pos_weight = [1.0,] * len(modmis_label_used)  # [1.0,]
    focal_gamma = 2  # 2

    # feats show setting
    feats_show_phase = []  # ["val"]

    # Optimizer Setting
    optim_type = "adamw"  # "adamw"
    learning_rate = 1e-4  # 1e-4
    weight_decay = 0.01  # 0.01
    decay_power = 1  # 1
    max_epoch = 150  # 100
    max_steps = 25000  # 25000
    warmup_steps = 2500  # 2500
    end_lr = 0  # 0
    lr_mult = 1  # multiply lr for downstream heads  # 1

    # below params varies with the environment
    data_root = "/tmp/pycharm_project_499/datasets/modmis/pickles/datasets"
    log_dir = results_dir
    per_gpu_batchsize = 16  # 每个gpu的batch
    num_gpus = 1  # 每个主机上gpu的数量
    vit_load_path = f"/tmp/pycharm_project_499/pretrained/vit/vit_base_p{patch_size}_{image_size}" + ("_in21k" if "_in21k" in vit else "") + ".pth"
    precision = 32

    # modmis specific
    loss_names = {"modmis_bin": 1, "modmis_cls": 0, "modmis_reg": 0}
    loss_weight = {"modmis_bin": 1.0, "modmis_cls": 0.0, "modmis_reg": 0.0}
    label_class_count = [2,] * len(modmis_label_used)

    # PL Trainer Setting
    resume_from = "None"  # 原来为None，路径需要../，跑完设置为None
    fast_dev_run = False  # False
    val_check_interval = 1.0  # 1.0
    test_only = True  # False
    finetune_first = False  # False

    assert loss_names.keys() == loss_weight.keys()
    assert len(modmis_label_used) == len(bcewl_pos_weight) == len(label_class_count)
    assert used_loss in ("bcewl", "focal")




class CustomProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def disable_validation_bar(self):
        self.val_progress_bar = None

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


@ex.main
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ======== 数据读取模块 和 模型
    dm = MODMISDataModule(_config)
    model = ViLTransformerSS(_config, dm.get_dataset_info())

    # ======== 日志和检查点设置
    exp_name = f'{_config["exp_name"]}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=_config["pl_log_name"],
    )
    # 这里的checkpoint根据监视值monitor的情况来判断应做的保存模型的行为
    # 当前为不保存checkpoint的设置
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,  # 1
        verbose=True,  # True
        monitor="val/the_metric",  # "val/the_metric"
        mode="max",  # "max"
        save_last=True,  # True
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    progress_bar = CustomProgressBar()  # 禁用验证进度显示
    progress_bar.disable_validation_bar()
    callbacks = [checkpoint_callback, lr_callback, progress_bar]

    # ======== 训练器 pl.Trainer 的配置
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])
#     max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # 原本的加速策略为ddp. ddp：多机多卡 | dp：单机多卡
        accelerator="dp",
        benchmark=True,
        deterministic=True,
        # max_epochs=_config["max_epoch"] if max_steps is None else 8,  # 原来是1000
        max_epochs=_config["max_epoch"],
        max_steps=_config["max_steps"],
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,  #
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,  #
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",  #
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # profiler=SimpleProfiler(),  #
    )

    # 使用训练器 让模型拟合数据
    # 要训练的话修改这里
    if   _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        print("\n\nlast checkpoint test:")
        trainer.test(model, datamodule=dm)
        print("\n\nbest checkpoint test:")
        model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.test(model, datamodule=dm)
    else:
        # 1: (82 8225 8515) (86 7478 8350) (87 7844 8555) (88 7084 8855) (89 8886 8989)
        # 2: (83 7358 x6872)
        ckpt_dir = "../kaggle_results/mm-vilt_0/version_121/checkpoints"
        for file_name in os.listdir(ckpt_dir):
            if "last" not in file_name:
                continue
            print(file_name)#这里执行了
            model = model.load_from_checkpoint(f"{ckpt_dir}/{file_name}")
            model.hparams.config["feats_show_phase"] = ["val"]
            # 这里设置为False能跑一下
            trainer.test(model, datamodule=dm)


if run_in_local:
    # print(22222222222)
    # 进入这里
    ex.run(named_configs=['local_task_modmis'])

else:
    ex.run(named_configs=['task_modmis'])



