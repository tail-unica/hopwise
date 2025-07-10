# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

# @Time   : 2025
# @Author : Alessandro Soccol, Giacomo Medda
# @Email  : alessandro.soccol@unica.it, giacomo.medda@unica.it

"""hopwise.quick_start
########################
"""

import logging
import sys
from collections.abc import MutableMapping
from logging import getLogger

import torch
import torch.distributed as dist

from hopwise.config import Config
from hopwise.data import construct_transform, create_dataset, data_preparation
from hopwise.trainer import HFPathLanguageModelingTrainer
from hopwise.utils import (
    KnowledgeEvaluationType,
    ModelType,
    calculate_valid_score,
    deep_dict_update,
    get_environment,
    get_flops,
    get_model,
    get_trainer,
    init_logger,
    init_seed,
    set_color,
)


def run(
    model,
    dataset,
    run="train",
    checkpoint=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run_hopwise(
            model=model,
            dataset=dataset,
            run=run,
            checkpoint=checkpoint,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_hopwises,
            args=(model, dataset, run, checkpoint, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run_hopwise(
    model=None,
    dataset=None,
    run="train",
    checkpoint=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        run (str, optional): The running mode, 'train' or 'evaluate'. Defaults to ``'train'``.
        checkpoint (str, optional): The path of the saved model file. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """  # noqa: E501

    # Initialize configuration
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )

    if checkpoint is not None:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=checkpoint, updating_config=config
        )

        logger = get_logger(config)
        logger.info(set_color(f"A checkpoint is provided from which to resume training {checkpoint}", "red"))
    else:
        logger = get_logger(config)
        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = get_model(config["model"])(config, train_data.dataset)
        if isinstance(model, torch.nn.Module):
            model = model.to(device=config["device"], dtype=config["weight_precision"])

    logger.info(model)

    transform = construct_transform(config)

    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    if run == "train":
        if checkpoint is not None:
            trainer.resume_checkpoint(checkpoint)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )

    elif run == "evaluate":
        if checkpoint is None:
            raise ValueError("Checkpoint is needed for evaluation")
        trainer.eval_collector.train_data_collect(train_data)

        if isinstance(trainer, HFPathLanguageModelingTrainer):
            trainer.init_hf_trainer(train_data, valid_data, show_progress=config["show_progress"])
            trainer.resume_checkpoint(checkpoint)

        best_valid_result = trainer.evaluate(
            valid_data, load_best_model=False, model_file=checkpoint, show_progress=config["show_progress"]
        )

        best_valid_score = calculate_valid_score(best_valid_result, trainer.valid_metric)
    else:
        raise ValueError(f"Invalid run mode: {run}")

    if best_valid_result is not None:
        if KnowledgeEvaluationType.REC in best_valid_result or KnowledgeEvaluationType.LP in best_valid_result:
            for task, result in best_valid_result.items():
                logger.info(set_color(f"[{task}] best valid ", "yellow") + f": {format_metrics(result)}")
        else:
            logger.info(set_color("best valid result", "yellow") + f": {format_metrics(best_valid_result)}")

    # model evaluation
    test_result = trainer.evaluate(
        test_data,
        load_best_model=saved and run != "evaluate",
        model_file=checkpoint,
        show_progress=config["show_progress"],
    )

    environment_tb = get_environment(config)
    logger.info("The running environment of this training is as follows:\n" + environment_tb.draw())

    if test_result is not None:
        if KnowledgeEvaluationType.REC in test_result or KnowledgeEvaluationType.LP in test_result:
            for task, result in test_result.items():
                logger.info(set_color(f"[{task}] test result ", "yellow") + f": {format_metrics(result)}")
        else:
            logger.info(set_color("test result", "yellow") + f": {format_metrics(test_result)}")

    # In the case of KG-aware tasks, we don't care about the final "best_valid_score"
    # format because it is not used anywhere.
    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def get_logger(config):
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    return logger


def format_metrics(metrics):
    formatted_str = "".join([f"[{key}]: {value} " for key, value in metrics.items()])
    return formatted_str


def run_hopwises(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(f"The last argument of run_hopwises should be a dict, but got {type(kwargs)}")
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_hopwise(
        *args[:5],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True, show_progress=False, callback_fn=None):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        verbose=show_progress,
        show_progress=show_progress,
        saved=saved,
        callback_fn=callback_fn,
    )
    if best_valid_result is not None:
        if KnowledgeEvaluationType.REC in best_valid_result and KnowledgeEvaluationType.REC in best_valid_score:
            best_valid_score, best_valid_result = (
                best_valid_score[KnowledgeEvaluationType.REC],
                best_valid_result[KnowledgeEvaluationType.REC],
            )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file, load_only_data=False, updating_config=None):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.
        load_only_data (bool, optional): Whether to load only the dataset and dataloaders without the model.
            Defaults to ``False``.
        updating_config (Config, optional): A Config object to update the config parameters loaded from checkpoint.
            Defaults to ``None``.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file, weights_only=False)
    config = checkpoint["config"]

    if updating_config is not None:
        deep_dict_update(config.final_config_dict, updating_config.final_config_dict)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])

    if not load_only_data:
        if config["MODEL_TYPE"] == ModelType.PATH_LANGUAGE_MODELING:
            from transformers.modeling_utils import PreTrainedModel

            model_class = get_model(config["model"])
            if not issubclass(model_class, PreTrainedModel):
                model.load_state_dict(checkpoint["state_dict"])
                model.load_other_parameter(checkpoint.get("other_parameter"))
        else:
            model.load_state_dict(checkpoint["state_dict"])
            model.load_other_parameter(checkpoint.get("other_parameter"))
    return config, model, dataset, train_data, valid_data, test_data
