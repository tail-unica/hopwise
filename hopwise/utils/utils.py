# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8, 2022/7/12, 2023/2/11
# @Author : Jiawei Guan, Lei Wang, Gaowei Zhang
# @Email  : guanjw@ruc.edu.cn, zxcptss@gmail.com, zgw2022101006@ruc.edu.cn

# UPDATE
# @Time   : 2025
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""hopwise.utils.utils
################################
"""

import copy
import datetime
import importlib
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from texttable import Texttable
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from hopwise.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    os.makedirs(dir_path, exist_ok=True)


def deep_dict_update(updated_dict, updating_dict):
    overwrite_keys = ["split"]
    if isinstance(updated_dict, dict) and isinstance(updating_dict, dict):
        for key, value in updating_dict.items():
            if isinstance(value, dict) and isinstance(updated_dict.get(key), dict) and key not in overwrite_keys:
                deep_dict_update(updated_dict[key], value)
            else:
                updated_dict[key] = value


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        "general_recommender",
        "context_aware_recommender",
        "sequential_recommender",
        "knowledge_aware_recommender",
        "knowledge_graph_embedding_recommender",
        "path_language_modeling_recommender",
        "exlib_recommender",
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = ".".join(["hopwise.model", submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(f"`model_name` [{model_name}] is not the name of an existing model.")
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    register_table = {
        "PEARLMGPT2": "PEARLMfromscratchTrainer",
        "PEARLMLlama2": "PEARLMfromscratchTrainer",
        "PEARLMLlama3": "PEARLMfromscratchTrainer",
    }

    try:
        return getattr(importlib.import_module("hopwise.trainer"), model_name + "Trainer")
    except AttributeError:
        if model_name in register_table:
            return getattr(importlib.import_module("hopwise.trainer"), register_table[model_name])
        elif model_type == ModelType.PATH_LANGUAGE_MODELING:
            return getattr(importlib.import_module("hopwise.trainer"), "HFPathLanguageModelingTrainer")
        elif model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module("hopwise.trainer"), "KGTrainer")
        elif model_type == ModelType.TRADITIONAL:
            return getattr(importlib.import_module("hopwise.trainer"), "TraditionalTrainer")
        else:
            return getattr(importlib.import_module("hopwise.trainer"), "Trainer")


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    elif value <= best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r"""Return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result["Recall@10"]


def dict2str(result_dict):
    r"""Convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """
    return "    ".join([str(metric) + " : " + str(value) for metric, value in result_dict.items()])


def init_seed(seed, reproducibility):
    r"""Init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r"""Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = "log_tensorboard"

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, "baseFilename")).split(".")[0]
            break
    if dir_name is None:
        dir_name = "{}-{}".format("model", get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.

    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """
    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return f"{reserved:.2f} G/{total:.2f} G"


def get_flops(model, dataset, device, logger, transform, verbose=False):
    r"""Given a model and dataset to the model, compute the per-operator flops
    of the given model.

    Args:
        model: the model to compute flop counts.
        dataset: dataset that are passed to `model` to count flops.
        device: cuda.device. It is the device that the model run on.
        verbose: whether to print information of modules.

    Returns:
        total_ops: the number of flops for each operation.
    """
    if model.type == ModelType.DECISIONTREE:
        return 1
    if model.__class__.__name__ == "Pop":
        return 1

    model = copy.deepcopy(model)

    def count_normalization(m, x, y):
        x = x[0]
        flops = torch.DoubleTensor([2 * x.numel()])
        m.total_ops += flops

    def count_embedding(m, x, y):
        x = x[0]
        nelements = x.numel()
        hiddensize = y.shape[-1]
        m.total_ops += nelements * hiddensize

    class TracingAdapter(torch.nn.Module):
        def __init__(self, rec_model):
            super().__init__()
            self.model = rec_model

        def forward(self, interaction):
            return self.model.predict(interaction)

    custom_ops = {
        torch.nn.Embedding: count_embedding,
        torch.nn.LayerNorm: count_normalization,
    }
    wrapper = TracingAdapter(model)
    inter = dataset[torch.tensor([1])].to(device)
    inter = transform(dataset, inter)
    inputs = (inter,)
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_parameters

    handler_collection = {}
    fn_handles = []
    params_handles = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Register %s() for %s." % (fn.__qualname__, m_type))
        elif m_type not in types_collection and verbose:
            logger.warning("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handle_fn = m.register_forward_hook(fn)
            handle_paras = m.register_forward_hook(count_parameters)
            handler_collection[m] = (
                handle_fn,
                handle_paras,
            )
            fn_handles.append(handle_fn)
            params_handles.append(handle_paras)
        types_collection.add(m_type)

    prev_training_status = wrapper.training

    wrapper.eval()
    wrapper.apply(add_hooks)

    with torch.no_grad():
        wrapper(*inputs)

    def dfs_count(module: nn.Module, prefix="\t"):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params

        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(wrapper)

    # reset wrapper to original status
    wrapper.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")
    for i in range(len(fn_handles)):
        fn_handles[i].remove()
        params_handles[i].remove()

    return total_ops


def list_to_latex(convert_list, bigger_flag=True, subset_columns=[]):
    result = {}
    for d in convert_list:
        for key, value in d.items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]

    df_result = pd.DataFrame.from_dict(result, orient="index").T

    if len(subset_columns) == 0:
        tex = df_result.to_latex(index=False)
        return df_result, tex

    def bold_func(x, bigger_flag):
        if bigger_flag:
            return np.where(x == np.max(x.to_numpy()), "font-weight:bold", None)
        else:
            return np.where(x == np.min(x.to_numpy()), "font-weight:bold", None)

    style = df_result.style
    style.apply(bold_func, bigger_flag=bigger_flag, subset=subset_columns)
    style.format(precision=4)

    num_column = len(df_result.columns)
    column_format = "c" * num_column
    tex = style.hide(axis="index").to_latex(
        caption="Result Table",
        label="Result Table",
        convert_css=True,
        hrules=True,
        column_format=column_format,
    )

    return df_result, tex


def get_environment(config):
    gpu_usage = get_gpu_usage(config["device"]) if torch.cuda.is_available() and config["use_gpu"] else "0.0 / 0.0"

    import psutil

    memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    memory_total = psutil.virtual_memory()[0] / 1024**3
    memory_usage = f"{memory_used:.2f} G/{memory_total:.2f} G"
    cpu_usage = f"{psutil.cpu_percent(interval=1):.2f} %"
    """environment_data = [
        {"Environment": "CPU", "Usage": cpu_usage,},
        {"Environment": "GPU", "Usage": gpu_usage, },
        {"Environment": "Memory", "Usage": memory_usage, },
    ]"""

    table = Texttable()
    table.set_cols_align(["l", "c"])
    table.set_cols_valign(["m", "m"])
    table.add_rows(
        [
            ["Environment", "Usage"],
            ["CPU", cpu_usage],
            ["GPU", gpu_usage],
            ["Memory", memory_usage],
        ]
    )

    return table


def get_sequence_postprocessor(postprocessor_name):
    try:
        postprocessor_name = postprocessor_name + "SequenceScorePostProcessor"
        return getattr(importlib.import_module("hopwise.model.sequence_postprocessor"), postprocessor_name)
    except AttributeError:
        return getattr(
            importlib.import_module("hopwise.model.sequence_postprocessor"), "BeamSearchSequenceScorePostProcessor"
        )


def get_logits_processor(model_name):
    try:
        return getattr(
            importlib.import_module("hopwise.model.logits_processor"), model_name + "LogitsProcessorWordLevel"
        )
    except AttributeError:
        return getattr(
            importlib.import_module("hopwise.model.logits_processor"), "ConstrainedLogitsProcessorWordLevel"
        )


@dataclass
class GenerationOutputs(dict):
    r"""Dataclass to hold the outputs of the generation process.

    Attributes:
        sequences (torch.Tensor): The generated sequences.
        scores (torch.Tensor): The scores for each generated token.
    """

    sequences: torch.Tensor
    scores: torch.Tensor

    def __post_init__(self):
        self.update({"sequences": self.sequences, "scores": self.scores})
