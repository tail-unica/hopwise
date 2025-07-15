# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1, 2022/7/6
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan, Gaowei Zhang
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn, zgw15630559577@163.com  # noqa: E501

"""hopwise.data.utils
########################
"""

# ruff: noqa: F403, F405

import gc
import importlib
import os
import pickle
import warnings
from typing import Literal

from hopwise.data.dataloader import *
from hopwise.sampler import KGSampler, RepeatableSampler, Sampler
from hopwise.utils import KnowledgeEvaluationType, ModelType, ensure_dir, progress_bar, set_color
from hopwise.utils.argument_list import dataset_arguments


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("hopwise.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
            ModelType.PATH_LANGUAGE_MODELING: "KnowledgePathDataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(config["checkpoint_dir"], f"{config['dataset']}-{dataset_class.__name__}.pth")
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "magenta") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def _get_dataloader_name(config, dataloaders_folder):
    path_gen_args = config["path_sample_args"]

    max_path_per_user = config["MAX_PATHS_PER_USER"]
    max_rw_tries_per_iid = config["MAX_RW_TRIES_PER_IID"]
    restrict_by_phase = path_gen_args["restrict_by_phase"]
    temporal_causality = path_gen_args["temporal_causality"]
    strategy = path_gen_args["strategy"]
    collaborative_path = path_gen_args["collaborative_path"]

    filename = f"{config['dataset']}-for-{config['model']}"
    f"-str {strategy}"
    f"-mppu {max_path_per_user}"
    f"-max_tries {max_rw_tries_per_iid}"
    f"-temp {temporal_causality}"
    f"-col {collaborative_path}"
    f"-restrbyphase {restrict_by_phase}-dataloader.pth"

    if "train_stage" in config and config["MODEL_TYPE"] in [ModelType.PATH_LANGUAGE_MODELING]:
        filename += f"-{config['train_stage']}"

    file_path = os.path.join(
        config["checkpoint_dir"],
        dataloaders_folder,
        filename,
    )

    return file_path


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    if config["MODEL_TYPE"] == ModelType.PATH_LANGUAGE_MODELING:
        dataloaders_folder = f"{config['model']} - {config['dataset']} - dataloaders"
        ensure_dir(os.path.join(config["checkpoint_dir"], dataloaders_folder))
        file_path = _get_dataloader_name(config, dataloaders_folder)
    else:
        file_path = os.path.join(
            config["checkpoint_dir"],
            f"{config['dataset']}-for-{config['model']}-dataloader.pth",
        )

    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "magenta") + f": [{file_path}]")
    serialization_dataloaders = []
    for dataloader in dataloaders:
        if isinstance(dataloader, KnowledgeBasedDataLoader):
            general_generator_state = dataloader.general_dataloader.generator.get_state()
            dataloader.general_dataloader.generator = None
            dataloader.general_dataloader.sampler.generator = None
            kg_generator_state = dataloader.kg_dataloader.generator.get_state()
            dataloader.kg_dataloader.generator = None
            dataloader.kg_dataloader.sampler.generator = None
            serialization_dataloaders += [(dataloader, general_generator_state, kg_generator_state)]
        else:
            generator_state = dataloader.generator.get_state()
            dataloader.generator = None
            dataloader.sampler.generator = None
            serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    if config["MODEL_TYPE"] == ModelType.PATH_LANGUAGE_MODELING:
        dataloaders_folder = f"{config['model']} - {config['dataset']} - dataloaders"
        dataloaders_save_path = _get_dataloader_name(config, dataloaders_folder)

    else:
        default_file = os.path.join(
            config["checkpoint_dir"],
            f"{config['dataset']}-for-{config['model']}-dataloader.pth",
        )
        # used if you want to load a specific dataloader
        dataloaders_save_path = config["dataloaders_save_path"] or default_file

    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            for dataloader_saved_data in pickle.load(f):
                if isinstance(dataloader_saved_data[0], KnowledgeBasedDataLoader):
                    data_loader, general_generator_state, kg_generator_state = dataloader_saved_data
                    general_generator = torch.Generator()
                    general_generator.set_state(general_generator_state)
                    data_loader.general_dataloader.generator = general_generator
                    data_loader.general_dataloader.sampler.generator = general_generator
                    kg_generator = torch.Generator()
                    kg_generator.set_state(kg_generator_state)
                    data_loader.kg_dataloader.generator = kg_generator
                    data_loader.kg_dataloader.sampler.generator = kg_generator
                    dataloaders.append(data_loader)
                else:
                    data_loader, generator_state = dataloader_saved_data
                    generator = torch.Generator()
                    generator.set_state(generator_state)
                    data_loader.generator = generator
                    data_loader.sampler.generator = generator
                    dataloaders.append(data_loader)

        eval_lp_args = config["eval_lp_args"]
        if eval_lp_args is not None and eval_lp_args["knowledge_split"] is not None:
            train_data, valid_inter_data, valid_kg_data, test_inter_data, test_kg_data = dataloaders
        else:
            train_data, valid_data, test_data = dataloaders
    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if isinstance(train_data, KnowledgeBasedDataLoader):
            general_config = train_data.general_dataloader.config
            kg_config = train_data.kg_dataloader.config
            if config[arg] != general_config[arg] and config[arg] != kg_config[arg]:
                return None
        elif config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    if eval_lp_args is not None and eval_lp_args["knowledge_split"] is not None:
        valid_inter_data.update_config(config)
        valid_kg_data.update_config(config)
        test_inter_data.update_config(config)
        test_kg_data.update_config(config)

        valid_data = [valid_inter_data, valid_kg_data]
        test_data = [test_inter_data, test_kg_data]
    else:
        valid_data.update_config(config)
        test_data.update_config(config)
    logger = getLogger()
    logger.info(set_color("Load split dataloaders from", "magenta") + f": [{dataloaders_save_path}]")
    return train_data, valid_data, test_data


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """  # noqa: E501
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        model_type = config["MODEL_TYPE"]
        model_input_type = config["MODEL_INPUT_TYPE"]
        # model = config["model"]
        built_datasets = dataset.build()

        if model_type in [ModelType.KNOWLEDGE] and model_input_type not in [InputType.USERWISE]:
            if isinstance(built_datasets, dict):
                # then the kg has been split
                train_kg_dataset, valid_kg_dataset, test_kg_dataset = built_datasets[KnowledgeEvaluationType.LP]
                train_inter_dataset, valid_inter_dataset, test_inter_dataset = built_datasets[
                    KnowledgeEvaluationType.REC
                ]

                kg_sampler = KGSampler(
                    train_kg_dataset,
                    config["train_neg_sample_args"]["distribution"],
                    config["train_neg_sample_args"]["alpha"],
                )

                train_inter_sampler, valid_inter_sampler, test_inter_sampler = create_samplers(
                    config, dataset, built_datasets[KnowledgeEvaluationType.REC]
                )

                train_data = get_dataloader(config, "train")(
                    config, train_inter_dataset, train_inter_sampler, kg_sampler, shuffle=True
                )

                valid_kg_sampler = KGSampler(valid_kg_dataset, distribution=None)
                valid_kg_data = get_dataloader(config, "valid", task=KnowledgeEvaluationType.LP)(
                    config, valid_kg_dataset, valid_kg_sampler, shuffle=False
                )
                valid_inter_data = get_dataloader(config, "valid", task=KnowledgeEvaluationType.REC)(
                    config, valid_inter_dataset, valid_inter_sampler, shuffle=False
                )

                test_kg_sampler = KGSampler(test_kg_dataset, distribution=None)
                test_kg_data = get_dataloader(config, "valid", task=KnowledgeEvaluationType.LP)(
                    config, test_kg_dataset, test_kg_sampler, shuffle=False
                )
                test_inter_data = get_dataloader(config, "test", task=KnowledgeEvaluationType.REC)(
                    config, test_inter_dataset, test_inter_sampler, shuffle=False
                )

                if config["save_dataloaders"]:
                    save_split_dataloaders(
                        config,
                        dataloaders=(train_data, valid_inter_data, valid_kg_data, test_inter_data, test_kg_data),
                    )

                valid_data = [valid_inter_data, valid_kg_data]
                test_data = [test_inter_data, test_kg_data]
            else:
                # then we want to use the whole kg
                kg_sampler = KGSampler(
                    dataset,
                    config["train_neg_sample_args"]["distribution"],
                    config["train_neg_sample_args"]["alpha"],
                )

                train_dataset, valid_dataset, test_dataset = built_datasets
                train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)
                train_data = get_dataloader(config, "train")(
                    config, train_dataset, train_sampler, kg_sampler, shuffle=True
                )
                valid_data = get_dataloader(config, "valid")(config, valid_dataset, valid_sampler, shuffle=False)
                test_data = get_dataloader(config, "test")(config, test_dataset, test_sampler, shuffle=False)

                if config["save_dataloaders"]:
                    save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))
        else:
            train_dataset, valid_dataset, test_dataset = built_datasets
            train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
            valid_data = get_dataloader(config, "valid")(config, valid_dataset, valid_sampler, shuffle=False)
            test_data = get_dataloader(config, "test")(config, test_dataset, test_sampler, shuffle=False)

            if config["save_dataloaders"]:
                save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "magenta")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f"[{config['train_batch_size']}]", "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f"[{config['train_neg_sample_args']}]", "yellow")
    )

    if config["eval_lp_args"] is not None and config["eval_lp_args"]["knowledge_split"] is not None:
        eval_lp_args_info = (
            set_color(" eval_lp_args", "cyan") + ": " + set_color(f"[{config['eval_lp_args']}]", "yellow")
        )
    else:
        eval_lp_args_info = ""

    logger.info(
        set_color("[Evaluation]: ", "magenta")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f"[{config['eval_batch_size']}]", "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f"[{config['eval_args']}]", "yellow")
        + eval_lp_args_info
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"], task=KnowledgeEvaluationType.REC):  # noqa: PLR0911
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """

    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError("`phase` can only be 'train', 'valid', 'test' or 'evaluation'.")
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    if config["MODEL_INPUT_TYPE"] == InputType.USERWISE:
        if config["model"] in ["TPRec"]:
            if config["train_stage"] in ["policy"]:
                return _get_user_dataloader(config, phase)
        else:
            return _get_user_dataloader(config, phase)

    model_type = config["MODEL_TYPE"]
    if phase == "train":
        # Return Dataloader based on the modeltype
        if model_type == ModelType.KNOWLEDGE:
            return KnowledgeBasedDataLoader
        else:
            return TrainDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            if model_type == ModelType.PATH_LANGUAGE_MODELING:
                return KnowledgePathEvalDataLoader
            else:
                if task is KnowledgeEvaluationType.LP:
                    return FullSortLPEvalDataLoader
                return FullSortRecEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_user_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Customized function for models that needs only users

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError("`phase` can only be 'train', 'valid', 'test' or 'evaluation'.")
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortRecEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                distribution,
                alpha,
            )
    return sampler


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    train_neg_sample_args = config["train_neg_sample_args"]
    valid_neg_sample_args = config["valid_neg_sample_args"]
    test_neg_sample_args = config["test_neg_sample_args"]
    repeatable = config["repeatable"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
    )
    train_sampler = base_sampler.set_phase("train") if base_sampler else None

    valid_sampler = _create_sampler(
        dataset,
        built_datasets,
        valid_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    valid_sampler = valid_sampler.set_phase("valid") if valid_sampler else None

    test_sampler = _create_sampler(
        dataset,
        built_datasets,
        test_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    test_sampler = test_sampler.set_phase("test") if test_sampler else None
    return train_sampler, valid_sampler, test_sampler


def user_parallel_sampling(sampling_func_factory):
    """Decorator to parallelize path sampling functions."""

    import joblib

    # https://github.com/DLR-RM/stable-baselines3/issues/1645#issuecomment-2194345304
    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()

    def wrapper(*args, **kwargs):
        user_num = kwargs.get("user_num", None)
        tqdm_kws = dict(
            total=user_num - 1,
            ncols=100,
            desc="[red]KG Path Sampling",
        )

        sampling_func = sampling_func_factory(*args, **kwargs)

        parallel_max_workers = kwargs.pop("parallel_max_workers", "")
        if not parallel_max_workers:
            iter_users = map(sampling_func, range(1, user_num))
        else:
            iter_users = joblib.Parallel(n_jobs=parallel_max_workers, prefer="processes", return_as="generator")(
                joblib.delayed(sampling_func)(u) for u in range(1, user_num)
            )

        try:
            iter_users = progress_bar(iter_users, **tqdm_kws)
            return [p for p in iter_users]
        except Exception:
            if hasattr(iter_users, "close"):
                iter_users.close()

            raise

    return wrapper
