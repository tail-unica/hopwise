# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/8, 2021/6/23, 2020/9/26, 2020/9/26, 2020/10/01, 2020/9/16
# @Author : Zhen Tian, Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan
# @Email  : chenyuwuxinn@gmail.com, zhlin@ruc.edu.cn, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, slmu@ruc.edu.cn, panxy@ruc.edu.cn # noqa: E501

# UPDATE:
# @Time   : 2020/10/8, 2020/10/15, 2020/11/20, 2021/2/20, 2021/3/3, 2021/3/5, 2021/7/18, 2022/7/11, 2023/2/11
# @Author : Hui Wang, Xinyan Fan, Chen Yang, Yibo Li, Lanling Xu, Haoran Cheng, Zhichao Feng, Lei Wang, Gaowei Zhang
# @Email  : hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn, 254170321@qq.com, 2018202152@ruc.edu.cn, xulanling_sherry@163.com, chenghaoran29@foxmail.com, fzcbupt@gmail.com, zxcptss@gmail.com, zgw2022101006@ruc.edu.cn # noqa: E501

# UPDATE:
# @Time   : 2025
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

"""hopwise.trainer.trainer
################################
"""

import os
from collections import defaultdict
from logging import getLogger
from time import time

import numpy as np
import torch
from scipy import sparse
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import rich

from hopwise.data.dataloader import FullSortLPEvalDataLoader, NegSampleDataLoader
from hopwise.data.interaction import Interaction
from hopwise.evaluator import Collector, Collector_KG, Evaluator, Evaluator_KG
from hopwise.utils import (
    EvaluatorType,
    GenerationOutputs,
    KGDataLoaderState,
    WandbLogger,
    calculate_valid_score,
    dict2str,
    early_stopping,
    ensure_dir,
    get_gpu_usage,
    get_local_time,
    get_logits_processor,
    get_ranker,
    get_tensorboard,
    set_color,
)
from hopwise.utils.enum_type import KnowledgeEvaluationType

try:
    grad_scaler = torch.GradScaler
    autocast = torch.autocast
except AttributeError:

    def grad_scaler(device, **kwargs):
        if torch.cuda.is_available():
            return torch.cuda.amp.GradScaler(**kwargs)
        else:
            return torch.cuda.amp.GradScaler(**kwargs)

    def autocast(device_type=None, **kwargs):
        if torch.cuda.is_available():
            return torch.cuda.amp.autocast(**kwargs)
        else:
            return torch.cpu.amp.autocast(**kwargs)


class AbstractTrainer:
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        if not config["single_spec"]:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.distributed_model = DistributedDataParallel(self.model, device_ids=[config["local_rank"]])

    def fit(self, train_data):
        r"""Train the model based on the train data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def set_reduce_hook(self):
        r"""Call the forward function of 'distributed_model' to apply grads
        reduce hook to each parameter of its module.

        """
        t = self.model.forward
        self.model.forward = lambda x: x
        self.distributed_model(torch.LongTensor([0]).to(self.device))
        self.model.forward = t

    def sync_grad_loss(self):
        r"""Ensure that each parameter appears to the loss function to
        make the grads reduce sync in each node.

        """
        sync_loss = 0
        for params in self.model.parameters():
            sync_loss += torch.sum(params) * 0
        return sync_loss


class Trainer(AbstractTrainer):
    """The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super().__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.enable_amp = config["enable_amp"]
        self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config["eval_type"]
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if self.config["reg_weight"] and weight_decay and weight_decay * self.config["reg_weight"] > 0:
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning("Sparse Adam cannot argument received argument [{weight_decay}]")
        else:
            self.logger.warning("Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            rich.tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = grad_scaler(self.device, enabled=self.enable_scaler)
        for batch_idx, batch_interaction in enumerate(iter_data):
            interaction = batch_interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """

        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(set_color("Saving current", "blue") + f": {saved_model_file}")

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        self.best_valid_score = checkpoint["best_valid_score"]

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (
            set_color("epoch %d training", "green") + " [" + set_color("time", "blue") + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = set_color("train_loss%d", "blue") + ": %." + str(des) + "f"
            train_loss_output += ", ".join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = "%." + str(des) + "f"
            train_loss_output += set_color("train loss", "blue") + ": " + des % losses
        return train_loss_output + "]"

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag="Loss/Train"):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learner": self.config["learner"],
            "learning_rate": self.config["learning_rate"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter for parameters in self.config.parameters.values() for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update(
            {para: val for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter}
        )
        for k, hparam in hparam_dict.items():
            if hparam is not None and not isinstance(hparam, (bool, str, float, int)):
                hparam_dict[k] = str(hparam)

        self.tensorboard.add_hparams(hparam_dict, {"hparam/best_valid_result": best_valid_result})

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.train_data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, "valid_step": valid_step}, head="valid")

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _batch_eval(self, batched_data, tot_item_num, neg_sampling=False, item_tensor=None):
        if neg_sampling:
            return self._neg_sample_batch_eval(batched_data, tot_item_num=tot_item_num)
        else:
            return self._full_sort_batch_eval(batched_data, tot_item_num, item_tensor)

    def _full_sort_batch_eval(self, batched_data, tot_item_num, item_tensor):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._split_predict(new_inter, batch_size)

        scores = scores.view(-1, tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data, tot_item_num=None):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._split_predict(interaction, batch_size)

        if self.config["eval_type"] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config["eval_type"] == EvaluatorType.RANKING:
            col_idx = interaction[self.config["ITEM_ID_FIELD"]]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full((batch_user_num, tot_item_num), -np.inf, device=self.device)
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        # self.eval_collector.eval_data_collect(eval_data)

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location=self.device)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            if missing_keys:
                self.logger.info(set_color(f"Missing loaded keys: {missing_keys}", "red"))
            if unexpected_keys:
                self.logger.info(set_color(f"Unexpected loaded keys: {unexpected_keys}", "red"))
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = f"Loading model structure and parameters from {checkpoint_file}"
            self.logger.info(message_output)

        self.model.eval()

        item_tensor = None
        tot_item_num = eval_data._dataset.item_num
        neg_sampling = isinstance(eval_data, NegSampleDataLoader)
        if not neg_sampling:
            item_tensor = eval_data._dataset.get_item_feature().to(self.device)

        iter_data = (
            rich.tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color("Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = self._batch_eval(
                batched_data, tot_item_num, neg_sampling=neg_sampling, item_tensor=item_tensor
            )
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def _map_reduce(self, result, num_sample):
        gather_result = {}
        total_sample = [torch.zeros(1).to(self.device) for _ in range(self.config["world_size"])]
        torch.distributed.all_gather(total_sample, torch.Tensor([num_sample]).to(self.device))
        total_sample = torch.cat(total_sample, 0)
        total_sample = torch.sum(total_sample).item()
        for key, value in result.items():
            result[key] = torch.Tensor([value * num_sample]).to(self.device)
            gather_result[key] = [
                torch.zeros_like(result[key]).to(self.device) for _ in range(self.config["world_size"])
            ]
            torch.distributed.all_gather(gather_result[key], result[key])
            gather_result[key] = torch.cat(gather_result[key], dim=0)
            gather_result[key] = round(
                torch.sum(gather_result[key]).item() / total_sample,
                self.config["metric_decimal_place"],
            )
        return gather_result

    def _split_predict(self, interaction, batch_size):
        split_interaction = dict()
        for key, tensor in interaction.interaction.items():
            split_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in split_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)


class KGTrainer(Trainer):
    r"""KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
    recommendation related task and knowledge related task alternately.

    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.train_rec_step = config["train_rec_step"]
        self.train_kg_step = config["train_kg_step"]
        self.best_valid_score_lp = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result_lp = None
        self.cur_step_lp = 0
        self.tail_tensor = None

        if config["metrics_lp"]:
            self.eval_collector_kg = Collector_KG(config)
            self.evaluator_kg = Evaluator_KG(config)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        elif epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step:
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=self.model.calculate_kg_loss,
                show_progress=show_progress,
            )
        return None

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (Dataloader, list[Dataloader]): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        if isinstance(valid_data, list):
            # then we have also data for the kg part
            valid_data_inter = valid_data[0]
            valid_data_kg = valid_data[1]
            valid_result_inter = self.evaluate(valid_data_inter, load_best_model=False, show_progress=show_progress)
            valid_result_kg = self.evaluate(valid_data_kg, load_best_model=False, show_progress=show_progress)
            valid_score_inter = calculate_valid_score(valid_result_inter, self.valid_metric)
            valid_score_kg = calculate_valid_score(valid_result_kg, self.valid_metric)
            return {
                KnowledgeEvaluationType.REC: [valid_score_inter, valid_result_inter],
                KnowledgeEvaluationType.LP: [valid_score_kg, valid_result_kg],
            }
        else:
            valid_data_inter = valid_data
            valid_result_inter = self.evaluate(valid_data_inter, load_best_model=False, show_progress=show_progress)
            valid_score_inter = calculate_valid_score(valid_result_inter, self.valid_metric)
            return {KnowledgeEvaluationType.REC: [valid_score_inter, valid_result_inter]}

    def _batch_eval(self, batched_data, tot_target_num, neg_sampling=False, task=None, target_tensor=None):
        if neg_sampling:
            return self._neg_sample_batch_eval(batched_data, tot_item_num=tot_target_num)
        else:
            if task == KnowledgeEvaluationType.REC:
                full_sort_predict_fn = self.model.full_sort_predict
                predict_fn = self.model.predict
            else:
                full_sort_predict_fn = self.model.full_sort_predict_kg
                predict_fn = self.model.predict_kg

            return self._full_sort_batch_eval(
                batched_data,
                full_sort_predict_fn,
                predict_fn,
                target_tensor,
                tot_target_num,
            )

    def _full_sort_batch_eval(self, batched_data, full_sort_predict_fn, predict_fn, column_tensor, tot_column_num):
        # in the case of recommendation, positive_h and positive_t are the user and item ids
        interaction, history_index, positive_h, positive_t = batched_data
        try:
            scores = full_sort_predict_fn(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(tot_column_num)
            batch_size = len(new_inter)
            new_inter.update(column_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = predict_fn(new_inter)
            else:
                scores = self._split_predict_fn(new_inter, batch_size, predict_fn)

        scores = scores.view(-1, tot_column_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_h, positive_t

    def _split_predict_fn(self, interaction, batch_size, predict_fn):
        split_interaction = dict()
        for key, tensor in interaction.interaction.items():
            split_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, split_tensor in split_interaction.items():
                current_interaction[key] = split_tensor[i]
            result = predict_fn(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (Dataloader, list[Dataloader]): the eval data.
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = f"Loading model structure and parameters from {checkpoint_file}"
            self.logger.info(message_output)

        self.model.eval()

        results = dict()
        if isinstance(eval_data, list):
            task_eval_data = {KnowledgeEvaluationType.REC: eval_data[0], KnowledgeEvaluationType.LP: eval_data[1]}
        else:
            if isinstance(eval_data, FullSortLPEvalDataLoader):
                kg_eval_type = KnowledgeEvaluationType.LP
            else:
                kg_eval_type = KnowledgeEvaluationType.REC

            task_eval_data = {kg_eval_type: eval_data}

        # REC task
        if KnowledgeEvaluationType.REC in task_eval_data:
            task = KnowledgeEvaluationType.REC
            rec_eval_data = task_eval_data[task]

            item_tensor = None
            tot_item_num = rec_eval_data._dataset.item_num
            neg_sampling = isinstance(rec_eval_data, NegSampleDataLoader)
            if not neg_sampling:
                item_tensor = rec_eval_data._dataset.get_item_feature().to(self.device)

            results[task] = self.evaluate_data_loop(
                rec_eval_data, task, tot_item_num, item_tensor, show_progress=show_progress
            )

        # LP task
        if KnowledgeEvaluationType.LP in task_eval_data:
            task = KnowledgeEvaluationType.LP
            kg_eval_data = task_eval_data[task]

            tot_entity_num = kg_eval_data._dataset.entity_num
            tail_tensor = kg_eval_data._dataset.get_tail_feature().to(self.device)

            results[task] = self.evaluate_data_loop(
                kg_eval_data,
                task,
                tot_entity_num,
                tail_tensor,
                show_progress=show_progress,
            )

        if isinstance(eval_data, list):
            return results
        else:
            return results[kg_eval_type]

    def evaluate_data_loop(self, eval_data, task, tot_target_num, target_tensor, show_progress=True):
        neg_sampling = isinstance(eval_data, NegSampleDataLoader)

        if task == KnowledgeEvaluationType.REC:
            eval_collector = self.eval_collector
            evaluator = self.evaluator
        else:
            eval_collector = self.eval_collector_kg
            evaluator = self.evaluator_kg

        iter_data = (
            rich.tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate {task}", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = self._batch_eval(
                batched_data,
                tot_target_num,
                neg_sampling=neg_sampling,
                task=task,
                target_tensor=target_tensor,
            )
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
            eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        eval_collector.model_collect(self.model)
        struct = eval_collector.get_data_struct()
        result = evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.train_data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if not isinstance(valid_data, list):
                if self.eval_step <= 0 or not valid_data:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    continue
            elif self.eval_step <= 0 or not valid_data[0] or not valid_data[1]:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                return_data = self._valid_epoch(valid_data, show_progress=show_progress)

                best_valid = defaultdict(dict)

                if KnowledgeEvaluationType.LP in return_data:
                    kg_valid_scores = list()
                    kg_valid_results = list()

                update_flag = False
                stop_flag = False

                for task, (valid_score, valid_result) in return_data.items():
                    # TODO
                    #  - add early stopping for KG
                    if task == KnowledgeEvaluationType.REC:
                        (
                            self.best_valid_score,
                            self.cur_step,
                            stop_flag,
                            update_flag,
                        ) = early_stopping(
                            valid_score,
                            self.best_valid_score,
                            self.cur_step,
                            max_step=self.stopping_step,
                            bigger=self.valid_metric_bigger,
                        )
                    else:
                        (
                            self.best_valid_score_lp,
                            self.cur_step_lp,
                            _,
                            _,
                        ) = early_stopping(
                            valid_score,
                            self.best_valid_score_lp,
                            self.cur_step_lp,
                            max_step=self.stopping_step,
                            bigger=self.valid_metric_bigger,
                        )

                    valid_end_time = time()
                    valid_score_output = (
                        set_color(f"epoch %d evaluating {task}", "green")
                        + " ["
                        + set_color("time", "blue")
                        + ": %.2fs, "
                        + set_color("valid_score", "blue")
                        + ": %f]"
                    ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                    valid_result_output = set_color("valid result ", "blue") + ": \n" + dict2str(valid_result)

                    if verbose:
                        self.logger.info(valid_score_output)
                        self.logger.info(valid_result_output)

                    self.tensorboard.add_scalar(f"Valid_score_{task}", valid_score, epoch_idx)
                    self.wandblogger.log_metrics({**valid_result, f"valid_step_{task}": valid_step}, head="valid")

                    if task == KnowledgeEvaluationType.REC and update_flag:
                        if saved:
                            self._save_checkpoint(epoch_idx, verbose=verbose)
                        self.best_valid_result = valid_result

                    if callback_fn:
                        callback_fn(epoch_idx, valid_score)

                    if task == KnowledgeEvaluationType.REC and stop_flag:
                        stop_output = "Finished training, best eval result in epoch %d" % (
                            epoch_idx - self.cur_step * self.eval_step
                        )
                        if verbose:
                            self.logger.info(stop_output)
                        break

                    if task == KnowledgeEvaluationType.LP:
                        # Track valid scores and results
                        kg_valid_scores.append(valid_score)
                        kg_valid_results.append(valid_result)
                        # Track best valid scores and results
                        best_valid["score"][KnowledgeEvaluationType.LP] = max(kg_valid_scores)
                        best_valid["result"][KnowledgeEvaluationType.LP] = kg_valid_results[
                            kg_valid_scores.index(best_valid["score"][KnowledgeEvaluationType.LP])
                        ]
                    valid_step += 1

        best_valid["score"][KnowledgeEvaluationType.REC] = self.best_valid_score
        best_valid["result"][KnowledgeEvaluationType.REC] = self.best_valid_result

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return best_valid["score"], best_valid["result"]


class ExplainableTrainer(Trainer):
    """ExplainableTrainer is designed for explainable recommendation methods,
    in particular those that generate explanations using Reinforcement Learning methods
    such as PGPR and CAFE"""

    def __init__(self, config, model):
        super().__init__(config, model)

    def _full_sort_batch_eval(self, batched_data, tot_item_num, item_tensor):
        paths = None

        interaction, history_index, positive_u, positive_i = batched_data

        # Note: interaction without item ids
        scores = self.model.full_sort_predict(interaction.to(self.device))
        if isinstance(scores, tuple):
            # then the first is the score, the second are paths
            scores, paths = scores

        scores = scores.view(-1, tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf

        if paths is not None:
            return interaction, (scores, paths), positive_u, positive_i
        else:
            return interaction, scores, positive_u, positive_i


class PGPRTrainer(ExplainableTrainer):
    r"""PGPRTrainer is designed for PGPR, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super().__init__(config, model)


class CAFETrainer(ExplainableTrainer):
    r"""CAFETrainer is designed for CAFE, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super().__init__(config, model)


class KGATTrainer(Trainer):
    r"""KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super().__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        # train rs
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)

        # train kg
        train_data.set_mode(KGDataLoaderState.KG)
        kg_total_loss = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=self.model.calculate_kg_loss,
            show_progress=show_progress,
        )

        # update A
        self.model.eval()
        with torch.no_grad():
            self.model.update_attentive_A()

        return rs_total_loss, kg_total_loss


class PretrainTrainer(Trainer):
    r"""PretrainTrainer is designed for pre-training.
    It can be inherited by the trainer which needs pre-training and fine-tuning.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.pretrain_epochs = self.config["pretrain_epochs"]
        self.save_step = self.config["save_step"]

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "other_parameter": self.model.other_parameter(),
        }
        torch.save(state, saved_model_file)
        self.saved_model_file = saved_model_file

    def _get_pretrained_model_path(self, epoch_label=None):
        epoch_label = str(epoch_label) if epoch_label is not None else "pretrained"
        return os.path.join(
            self.checkpoint_dir,
            "{}-{}-{}.pth".format(self.config["model"], self.config["dataset"], epoch_label),
        )

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = self._get_pretrained_model_path(epoch_idx + 1)
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = set_color("Saving current", "blue") + ": %s" % saved_model_file
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result


class S3RecTrainer(PretrainTrainer):
    r"""S3RecTrainer is designed for S3Rec, which is a self-supervised learning based sequential recommenders.
    It includes two training stages: pre-training ang fine-tuning.

    """

    def __init__(self, config, model):
        super().__init__(config, model)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if self.model.train_stage == "pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "finetune":
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError("Please make sure that the 'train_stage' is 'pretrain' or 'finetune'!")


class TPRecTrainer(PretrainTrainer):
    """
    TPRecTrainer is designed for TPRec, which is a knowledge-aware recommendation method.
    """

    def __init__(self, config, model):
        super().__init__(config, model)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if self.model.train_stage == "pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "policy":
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError("Please make sure that the 'train_stage' is 'pretrain' or 'finetune'!")

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.config["train_stage"] == "policy":
            return super()._train_epoch(train_data, epoch_idx, loss_func=loss_func, show_progress=show_progress)

        if self.config["train_rec_step"] is None or self.config["train_kg_step"] is None:
            interaction_state = KGDataLoaderState.RSKG
        elif (
            epoch_idx % (self.config["train_rec_step"] + self.config["train_kg_step"]) < self.config["train_rec_step"]
        ):
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=self.model.calculate_loss,
                show_progress=show_progress,
            )
        return None

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        # self.eval_collector.eval_data_collect(eval_data)

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = f"Loading model structure and parameters from {checkpoint_file}"
            self.logger.info(message_output)

        self.model.eval()

        item_tensor = None
        tot_item_num = eval_data._dataset.item_num
        neg_sampling = isinstance(eval_data, NegSampleDataLoader)
        if not neg_sampling:
            item_tensor = eval_data._dataset.get_item_feature().to(self.device)

        iter_data = (
            rich.tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color("Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)

            if self.model.train_stage == "policy":
                batched_data = (batched_data, eval_data.temporal_weights)  # noqa: PLW2901

            interaction, scores, positive_u, positive_i = self._batch_eval(
                batched_data, tot_item_num, neg_sampling=neg_sampling, item_tensor=item_tensor
            )
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

        saved_model_file = self._get_pretrained_model_path()
        self.save_pretrained_model(epoch_idx, saved_model_file)
        update_output = set_color("Saving pretrained weights", "blue") + ": %s" % saved_model_file
        if verbose:
            self.logger.info(update_output)
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data, tot_item_num, item_tensor):
        if self.model.train_stage == "pretrain":
            return super()._full_sort_batch_eval(batched_data, tot_item_num, item_tensor)

        paths = None
        batched_data, temporal_weights = batched_data

        interaction, history_index, positive_u, positive_i = batched_data

        # Note: interaction without item ids
        scores = self.model.full_sort_predict((interaction.to(self.device), temporal_weights))

        if isinstance(scores, tuple):
            # then the first is the score, the second are paths
            scores, paths = scores

        scores = scores.view(-1, tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf

        if paths is not None:
            return interaction, (scores, paths), positive_u, positive_i
        else:
            return interaction, scores, positive_u, positive_i


class MKRTrainer(Trainer):
    r"""MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self.kge_interval = config["kge_interval"]

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        rs_total_loss, kg_total_loss = 0.0, 0.0

        # train rs
        self.logger.info("Train RS")
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=self.model.calculate_rs_loss,
            show_progress=show_progress,
        )

        # train kg
        if epoch_idx % self.kge_interval == 0:
            self.logger.info("Train KG")
            train_data.set_mode(KGDataLoaderState.KG)
            kg_total_loss = super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=self.model.calculate_kg_loss,
                show_progress=show_progress,
            )

        return rs_total_loss, kg_total_loss


class TraditionalTrainer(Trainer):
    """TraditionalTrainer is designed for Traditional model(Pop,ItemKNN),
    which set the epoch to 1 whatever the config."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self.epochs = 1  # Set the epoch to 1 when running memory based model


class DecisionTreeTrainer(AbstractTrainer):
    """DecisionTreeTrainer is designed for DecisionTree model."""

    def __init__(self, config, model):
        super().__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.label_field = config["LABEL_FIELD"]
        self.convert_token_to_onehot = self.config["convert_token_to_onehot"]

        # evaluator
        self.eval_type = config["eval_type"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.valid_metric = config["valid_metric"].lower()
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)

        # model saved
        self.checkpoint_dir = config["checkpoint_dir"]
        ensure_dir(self.checkpoint_dir)
        temp_file = "{}-{}-temp.pth".format(self.config["model"], get_local_time())
        self.temp_file = os.path.join(self.checkpoint_dir, temp_file)

        temp_best_file = "{}-{}-temp-best.pth".format(self.config["model"], get_local_time())
        self.temp_best_file = os.path.join(self.checkpoint_dir, temp_best_file)

        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.stopping_step = config["stopping_step"]
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None

    def _interaction_to_sparse(self, dataloader):
        r"""Convert data format from interaction to sparse or numpy

        Args:
            dataloader (DecisionTreeDataLoader): DecisionTreeDataLoader dataloader.

        Returns:
            cur_data (sparse or numpy): data.
            interaction_np[self.label_field] (numpy): label.
        """
        interaction = dataloader._dataset[:]
        interaction_np = interaction.numpy()
        cur_data = np.array([])
        columns = []
        for key, interaction_value in interaction_np.items():
            value = np.resize(interaction_value, (interaction_value.shape[0], 1))
            if key != self.label_field:
                columns.append(key)
                if cur_data.shape[0] == 0:
                    cur_data = value
                else:
                    cur_data = np.hstack((cur_data, value))

        if self.convert_token_to_onehot:
            convert_col_list = dataloader._dataset.convert_col_list
            hash_count = dataloader._dataset.hash_count

            new_col = cur_data.shape[1] - len(convert_col_list)
            for key, values in hash_count.items():
                new_col = new_col + values
            onehot_data = sparse.dok_matrix((cur_data.shape[0], new_col))

            cur_j = 0
            new_j = 0

            for key in columns:
                if key in convert_col_list:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, int(new_j + cur_data[i, cur_j])] = 1
                    new_j = new_j + hash_count[key] - 1
                else:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, new_j] = cur_data[i, cur_j]
                cur_j = cur_j + 1
                new_j = new_j + 1

            cur_data = sparse.csc_matrix(onehot_data)

        return cur_data, interaction_np[self.label_field]

    def _interaction_to_lib_datatype(self, dataloader):
        pass

    def _valid_epoch(self, valid_data):
        r"""Args:
        valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.temp_best_file,
            "other_parameter": None,
        }
        torch.save(state, self.saved_model_file)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        for epoch_idx in range(self.epochs):
            self._train_at_once(train_data, valid_data)

            if (epoch_idx + 1) % self.eval_step == 0:
                # evaluate
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self.model.save_model(self.temp_best_file)
                        self._save_checkpoint(epoch_idx)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if self.temp_file:
                        os.remove(self.temp_file)
                    if verbose:
                        self.logger.info(stop_output)
                    break

        return self.best_valid_score, self.best_valid_result

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        raise NotImplementedError

    def _train_at_once(self, train_data, valid_data):
        raise NotImplementedError


class XGBoostTrainer(DecisionTreeTrainer):
    """XGBoostTrainer is designed for XGBOOST."""

    def __init__(self, config, model):
        super().__init__(config, model)

        self.xgb = __import__("xgboost")
        self.boost_model = config["xgb_model"]
        self.silent = config["xgb_silent"]
        self.nthread = config["xgb_nthread"]

        # train params
        self.params = config["xgb_params"]
        self.num_boost_round = config["xgb_num_boost_round"]
        self.evals = ()
        self.early_stopping_rounds = config["xgb_early_stopping_rounds"]
        self.evals_result = {}
        self.verbose_eval = config["xgb_verbose_eval"]
        self.callbacks = None
        self.deval = None
        self.eval_pred = self.eval_true = None

    def _interaction_to_lib_datatype(self, dataloader):
        r"""Convert data format from interaction to DMatrix

        Args:
            dataloader (DecisionTreeDataLoader): xgboost dataloader.

        Returns:
            DMatrix: Data in the form of 'DMatrix'.
        """
        data, label = self._interaction_to_sparse(dataloader)
        return self.xgb.DMatrix(data=data, label=label, silent=self.silent, nthread=self.nthread)

    def _train_at_once(self, train_data, valid_data):
        r"""Args:
        train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_lib_datatype(train_data)
        self.dvalid = self._interaction_to_lib_datatype(valid_data)
        self.evals = [(self.dtrain, "train"), (self.dvalid, "valid")]
        self.model = self.xgb.train(
            self.params,
            self.dtrain,
            self.num_boost_round,
            self.evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=self.verbose_eval,
            xgb_model=self.boost_model,
            callbacks=self.callbacks,
        )

        self.model.save_model(self.temp_file)
        self.boost_model = self.temp_file

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.temp_best_file
            self.model.load_model(checkpoint_file)

        self.deval = self._interaction_to_lib_datatype(eval_data)
        self.eval_true = torch.Tensor(self.deval.get_label())
        self.eval_pred = torch.Tensor(self.model.predict(self.deval))

        self.eval_collector.eval_collect(self.eval_pred, self.eval_true)
        result = self.evaluator.evaluate(self.eval_collector.get_data_struct())
        return result


class LightGBMTrainer(DecisionTreeTrainer):
    """LightGBMTrainer is designed for LightGBM."""

    def __init__(self, config, model):
        super().__init__(config, model)

        self.lgb = __import__("lightgbm")

        # train params
        self.params = config["lgb_params"]
        self.num_boost_round = config["lgb_num_boost_round"]
        self.evals = ()
        self.deval_data = self.deval_label = None
        self.eval_pred = self.eval_true = None

    def _interaction_to_lib_datatype(self, dataloader):
        r"""Convert data format from interaction to Dataset

        Args:
            dataloader (DecisionTreeDataLoader): xgboost dataloader.

        Returns:
            dataset(lgb.Dataset): Data in the form of 'lgb.Dataset'.
        """
        data, label = self._interaction_to_sparse(dataloader)
        return self.lgb.Dataset(data=data, label=label)

    def _train_at_once(self, train_data, valid_data):
        r"""Args:
        train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_lib_datatype(train_data)
        self.dvalid = self._interaction_to_lib_datatype(valid_data)
        self.evals = [self.dtrain, self.dvalid]
        self.model = self.lgb.train(self.params, self.dtrain, self.num_boost_round, self.evals)

        self.model.save_model(self.temp_file)
        self.boost_model = self.temp_file

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.temp_best_file
            self.model = self.lgb.Booster(model_file=checkpoint_file)

        self.deval_data, self.deval_label = self._interaction_to_sparse(eval_data)
        self.eval_true = torch.Tensor(self.deval_label)
        self.eval_pred = torch.Tensor(self.model.predict(self.deval_data))

        self.eval_collector.eval_collect(self.eval_pred, self.eval_true)
        result = self.evaluator.evaluate(self.eval_collector.get_data_struct())
        return result


class RaCTTrainer(PretrainTrainer):
    r"""RaCTTrainer is designed for RaCT, which is an actor-critic reinforcement learning based general recommenders.
    It includes three training stages: actor pre-training, critic pre-training and actor-critic training.

    """

    def __init__(self, config, model):
        super().__init__(config, model)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if self.model.train_stage == "actor_pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "critic_pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "finetune":
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError(
                "Please make sure that the 'train_stage' is 'actor_pretrain', 'critic_pretrain' or 'finetune'!"
            )


class RecVAETrainer(Trainer):
    r"""RecVAETrainer is designed for RecVAE, which is a general recommender."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self.n_enc_epochs = config["n_enc_epochs"]
        self.n_dec_epochs = config["n_dec_epochs"]

        self.optimizer_encoder = self._build_optimizer(params=self.model.encoder.parameters())
        self.optimizer_decoder = self._build_optimizer(params=self.model.decoder.parameters())

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.optimizer = self.optimizer_encoder

        def encoder_loss_func(data):
            return self.model.calculate_loss(data, encoder_flag=True)

        for epoch in range(self.n_enc_epochs):
            super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=encoder_loss_func,
                show_progress=show_progress,
            )

        self.model.update_prior()
        loss = 0.0
        self.optimizer = self.optimizer_decoder

        def decoder_loss_func(data):
            return self.model.calculate_loss(data, encoder_flag=False)

        for epoch in range(self.n_dec_epochs):
            loss += super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=decoder_loss_func,
                show_progress=show_progress,
            )
        return loss


class NCLTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)

        self.num_m_step = config["m_step"]
        assert self.num_m_step is not None

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data.
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.train_data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # only differences from the original trainer
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color("Saving current", "blue") + ": %s" % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color("Saving current best", "blue") + ": %s" % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            rich.tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        scaler = grad_scaler(enabled=self.enable_scaler)

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        for batch_idx, batch_interaction in enumerate(iter_data):
            interaction = batch_interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                if epoch_idx < self.config["warm_up_step"]:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
        return total_loss


class PEARLMfromscratchTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.tokenizer = None
        self.logit_processor = None
        self.ranker = None
        self.path_gen_args = config["path_generation_args"]
        path_hop_length = config["path_hop_length"]
        token_sequence_length = 1 + path_hop_length + path_hop_length + 1
        self.ranker_max_new_tokens = token_sequence_length - 3

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        self.tokenizer = train_data.dataset.tokenizer
        logits_processor_params = dict(
            tokenized_ckg=train_data.dataset.get_tokenized_ckg(),
            tokenized_used_ids=train_data.get_tokenized_used_ids(),
            max_sequence_length=train_data.token_sequence_length,
            tokenizer=self.tokenizer,
            task=KnowledgeEvaluationType.REC,
        )

        ranker_params = dict(
            processing_class=self.tokenizer,
            used_ids=train_data.general_dataloader._sampler.used_ids,
            item_num=train_data.dataset.item_num,
        )

        self.logit_processor = get_logits_processor(self.config, logits_processor_params)
        self.ranker = get_ranker(self.config, ranker_params)

        from torch.utils.data import DataLoader

        train_data = train_data.tokenized_dataset["train"]
        train_data.set_format(type="torch")
        train_data = DataLoader(train_data, batch_size=self.config["train_batch_size"], shuffle=True)

        return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None

        iter_data = (
            rich.tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = grad_scaler(self.device, enabled=self.enable_scaler)
        for batch_idx, batch_interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with autocast(device_type=self.device.type, enabled=self.enable_amp):
                batch_interaction = torch.tensor(batch_interaction["input_ids"]).to(self.device)  # noqa: PLW2901
                losses = loss_func(batch_interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))

        return total_loss

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        # self.eval_collector.eval_data_collect(eval_data)

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = f"Loading model structure and parameters from {checkpoint_file}"
            self.logger.info(message_output)

        self.model.eval()

        item_tensor = None
        tot_item_num = eval_data._dataset.item_num
        # neg_sampling = isinstance(eval_data, NegSampleDataLoader)
        # if not neg_sampling:
        #     item_tensor = eval_data._dataset.get_item_feature().to(self.device)

        # from torch.utils.data import DataLoader
        # eval_paths = eval_data.inference_path_dataset["user_id"]
        # batched_eval_data = DataLoader(eval_paths, batch_size=self.config["eval_batch_size"], shuffle=False)

        iter_data = (
            rich.tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color("Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            # batched_data = eval_data.collate_fn(range(num_sample, num_sample + len(batched_data)))  # noqa: PLW2901
            num_sample += len(batched_data)

            interaction, results, positive_u, positive_i = self._full_sort_batch_eval(
                batched_data, tot_item_num, item_tensor
            )

            scores, paths = results

            if hasattr(self.model, "decode_path"):
                paths = self.model.decode_path(paths)

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def _full_sort_batch_eval(self, batched_data, tot_item_num, item_tensor):
        interaction, history_index, positive_u, positive_i = batched_data
        inputs = self.tokenizer(interaction, return_tensors="pt", add_special_tokens=False).to(self.device)

        predictions, probs = self.model.generate(
            inputs=inputs,
            logit_processor=self.logit_processor,
            max_new_tokens=self.ranker_max_new_tokens,
            top_k=self.path_gen_args["top_k"],
            paths_per_user=self.path_gen_args["paths_per_user"],
        )

        generation_outputs = GenerationOutputs(sequences=predictions, scores=probs)
        scores, paths = self.ranker.get_sequences(
            generation_outputs,
            self.ranker_max_new_tokens,
        )

        scores = scores.view(-1, tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, (scores, paths), positive_u, positive_i


class HFPathLanguageModelingTrainer(Trainer):
    r"""HFPathLanguageModelingTrainer is designed for path-based knowledge-aware recommendation methods.
    It is specifically designed to communicate with the Hugging Face Trainer to use language models and functionalities
    as tokenizers and beam search.
    """

    HOPWISE_SAVE_PATH_SUFFIX = "hopwise-"
    HUGGINGFACE_SAVE_PATH_SUFFIX = "huggingface-"

    def __init__(self, config, model):
        super().__init__(config, model)
        self.config = config
        self.eval_device = config["device"]
        self.path_hop_length = self.config["path_hop_length"]
        self.path_gen_args = self.config["path_generation_args"].copy()
        self.paths_per_user = self.path_gen_args.pop("paths_per_user")

        self.HOPWISE_SAVE_PATH_SUFFIX += f"{config['base_model']}-"
        self.HUGGINGFACE_SAVE_PATH_SUFFIX += f"{config['base_model']}-"

        dirname, basename = os.path.split(self.saved_model_file)
        self.saved_model_file = os.path.join(dirname, self.HOPWISE_SAVE_PATH_SUFFIX + basename)

    def prepare_training_arguments(self, **kwargs):
        from transformers import TrainingArguments

        output_dir = self.saved_model_file.replace(self.HOPWISE_SAVE_PATH_SUFFIX, self.HUGGINGFACE_SAVE_PATH_SUFFIX)

        train_args = dict(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            eval_steps=self.eval_step,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            bf16=False,
            fp16=self.enable_amp,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.config["train_batch_size"],
            per_device_eval_batch_size=self.test_batch_size,
            warmup_steps=self.config["warmup_steps"],
            save_steps=self.eval_step,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=self.valid_metric,
            greater_is_better=self.valid_metric_bigger,
            seed=self.config["seed"],
            report_to="none",
        )
        train_args.update(kwargs)
        return TrainingArguments(**train_args)

    def init_hf_trainer(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        hf_callbacks=None,
        callback_fn=None,
        training_args=None,
    ):
        from hopwise.trainer.hf_path_trainer import HFPathTrainer, HopwiseCallback

        training_args = training_args or {}

        hf_callbacks = hf_callbacks or []
        training_arguments = self.prepare_training_arguments(**training_args)

        callbacks = [
            HopwiseCallback(
                self,
                train_data,
                valid_data=valid_data,
                verbose=verbose,
                saved=saved,
                show_progress=show_progress,
                callback_fn=callback_fn,
                model=self.model,
                model_name=self.model.__class__.__name__,
            ),
            *hf_callbacks,
        ]

        self.hf_trainer = HFPathTrainer(
            train_data,
            self.config,
            callbacks,
            model=self.model,
            args=training_arguments,
            path_hop_length=self.path_hop_length,
            paths_per_user=self.paths_per_user,
            path_generation_args=self.path_gen_args,
            eval_device=self.eval_device,
        )

    @property
    def processing_class(self):
        if hasattr(self, "hf_trainer"):
            return self.hf_trainer.processing_class
        return None

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(set_color("Saving current", "blue") + f": {saved_model_file}")
            hf_output_dir = self.hf_trainer.args.output_dir
            self.logger.info(set_color("HuggingFace model is saved at", "blue") + f": {hf_output_dir}")

    def resume_checkpoint(self, resume_file):
        """
        Load the model parameters and training information based on the directory name,
        and navigate into subdirectories if necessary.
        Also handles both HuggingFace and Hopwise formats by reading corresponding files.

        Args:
            resume_file (str): the path to the directory containing the checkpoint files or subdirectories
        """
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        if not hasattr(self, "hf_trainer"):
            raise ValueError("The HuggingFace Trainer has not been initialized. Please call `init_hf_trainer` first.")

        if os.path.basename(resume_file).startswith(self.HUGGINGFACE_SAVE_PATH_SUFFIX):
            hf_resume_file = resume_file
            hopwise_resume_file = resume_file.replace(self.HUGGINGFACE_SAVE_PATH_SUFFIX, self.HOPWISE_SAVE_PATH_SUFFIX)
        elif os.path.basename(resume_file).startswith(self.HOPWISE_SAVE_PATH_SUFFIX):
            hopwise_resume_file = resume_file
            hf_resume_file = resume_file.replace(self.HOPWISE_SAVE_PATH_SUFFIX, self.HUGGINGFACE_SAVE_PATH_SUFFIX)
        else:
            raise ValueError(f"The directory name [{resume_file}] does not indicate a HuggingFace or Hopwise model.")

        checkpoint = torch.load(hopwise_resume_file, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        self.best_valid_score = checkpoint["best_valid_score"]

        weights = load_file(os.path.join(hf_resume_file, "model.safetensors"))
        self.model.load_state_dict(weights, strict=False)
        self.processing_class.tokenizer = AutoTokenizer.from_pretrained(hf_resume_file)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        self.eval_collector.train_data_collect(train_data)

        if not hasattr(self, "hf_trainer"):
            self.init_hf_trainer(
                train_data,
                valid_data=valid_data,
                verbose=verbose,
                saved=saved,
                show_progress=show_progress,
                callback_fn=callback_fn,
            )

        self.hf_trainer.train()
        self.hf_trainer.save_model()

        return self.best_valid_score, self.best_valid_result

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress, task=KnowledgeEvaluationType.REC
        )
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False, task=KnowledgeEvaluationType.REC
    ):
        if not eval_data:
            return

        if load_best_model:
            self.hf_trainer._load_best_model()
            best_model_checkpoint_path = self.hf_trainer.state.best_model_checkpoint
            message_output = f"Loading model structure and parameters from {best_model_checkpoint_path}"
            self.logger.info(message_output)

        self.model.eval()

        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            rich.tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color("Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, history_index, positive_u, positive_i = batched_data

            inputs = self.processing_class(interaction, return_tensors="pt", add_special_tokens=False).to(self.device)
            scores, paths = self.hf_trainer._full_sort_batch_eval(inputs, task=task)

            if hasattr(self.model, "decode_path"):
                paths = self.model.decode_path(paths)

            scores = scores.view(-1, self.tot_item_num)
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
            self.eval_collector.eval_batch_collect((scores, paths), None, positive_u, positive_i)

        if "rec.paths" in self.eval_collector.data_struct:
            collected_paths = self.eval_collector.data_struct.get("rec.paths")
            path_uids = [path_data[-1][0][-1] for path_data in collected_paths]
            _, topk_sizes = np.unique(path_uids, return_counts=True)
            self.logger.info(f"{set_color('Average paths per user: ', 'blue')}{np.mean(topk_sizes):.2f}")

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result


class PLMTrainer(HFPathLanguageModelingTrainer):
    """PLMTrainer is designed for PLM, which is a path-based language model for knowledge-aware recommendation.
    It includes a logits processor to alternatively generate entity and relation tokens during evaluation.
    """

    def init_hf_trainer(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        hf_callbacks=None,
        callback_fn=None,
        training_args=None,
    ):
        from hopwise.trainer.hf_path_trainer import HFPathTrainer, HopwiseCallback

        training_args = training_args or {}

        hf_callbacks = hf_callbacks or []
        training_arguments = self.prepare_training_arguments(**training_args)

        callbacks = [
            HopwiseCallback(
                self,
                train_data,
                valid_data=valid_data,
                verbose=verbose,
                saved=saved,
                show_progress=show_progress,
                callback_fn=callback_fn,
            ),
            *hf_callbacks,
        ]

        logits_processor_params = dict(
            tokenized_ckg=train_data.dataset.get_tokenized_ckg(),
            tokenized_used_ids=train_data.get_tokenized_used_ids(),
            max_sequence_length=train_data.token_sequence_length,
            tokenizer=train_data.dataset.tokenizer,
            task=KnowledgeEvaluationType.REC,
        )
        self.logits_processor_list = get_logits_processor(self.config, logits_processor_params)

        self.hf_trainer = HFPathTrainer(
            train_data,
            self.config,
            callbacks,
            model=self.model,
            args=training_arguments,
            path_hop_length=self.path_hop_length,
            paths_per_user=self.paths_per_user,
            path_generation_args=self.path_gen_args,
            eval_device=self.device,
            logits_processor_list=self.logits_processor_list,
        )


class PEARLMTrainer(HFPathLanguageModelingTrainer):
    """PEARLMTrainer is designed for PEARLM, which is a path-based language model for knowledge-aware recommendation.
    It includes the knowledge graph constrained decoding (KGCD) logits processor.
    """

    def init_hf_trainer(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        hf_callbacks=None,
        callback_fn=None,
        training_args=None,
    ):
        from hopwise.trainer.hf_path_trainer import HFPathTrainer, HopwiseCallback

        training_args = training_args or {}

        hf_callbacks = hf_callbacks or []
        training_arguments = self.prepare_training_arguments(**training_args)

        callbacks = [
            HopwiseCallback(
                self,
                train_data,
                valid_data=valid_data,
                verbose=verbose,
                saved=saved,
                show_progress=show_progress,
                callback_fn=callback_fn,
            ),
            *hf_callbacks,
        ]

        logits_processor_params = dict(
            tokenized_ckg=train_data.dataset.get_tokenized_ckg(),
            tokenized_used_ids=train_data.get_tokenized_used_ids(),
            max_sequence_length=train_data.token_sequence_length,
            tokenizer=train_data.dataset.tokenizer,
            task=KnowledgeEvaluationType.REC,
        )

        self.logits_processor_list = get_logits_processor(self.config, logits_processor_params)

        self.hf_trainer = HFPathTrainer(
            train_data,
            self.config,
            callbacks,
            model=self.model,
            args=training_arguments,
            path_hop_length=self.path_hop_length,
            paths_per_user=self.paths_per_user,
            path_generation_args=self.path_gen_args,
            eval_device=self.device,
            logits_processor_list=self.logits_processor_list,
        )


class KGGLMTrainer(PEARLMTrainer, PretrainTrainer):
    r"""KGGLM is designed for KGGLM, which is a path-based language model for knowledge-aware recommendation.
    It includes two training stages: link prediction pre-training and recommendation path generation fine-tuning.
    """

    def _get_pretrained_model_path(self, epoch_label=None):
        epoch_label = f"pretrained-{epoch_label}" if epoch_label is not None else "pretrained"
        return os.path.join(
            self.checkpoint_dir,
            self.HUGGINGFACE_SAVE_PATH_SUFFIX
            + "{}-{}-{}.pth".format(self.config["model"], self.config["dataset"], epoch_label),
        )

    def pretrain(self, train_data, verbose=True, show_progress=False):
        from transformers import TrainerCallback

        pretrain_path = self._get_pretrained_model_path()
        pretrain_args = dict(
            output_dir=pretrain_path,
            num_train_epochs=self.pretrain_epochs,
            save_steps=self.save_step,
            eval_strategy="no",
            load_best_model_at_end=False,
        )

        class PretrainSaveCallback(TrainerCallback):
            def __init__(self, hopwise_trainer):
                self.hopwise_trainer = hopwise_trainer

            def on_epoch_end(self, args, state, control, **kwargs):
                if control.should_save:
                    epoch_idx = int(state.epoch)
                    pretrain_path = self.hopwise_trainer._get_pretrained_model_path(epoch_idx)
                    self.hopwise_trainer.hf_trainer.args.output_dir = pretrain_path

        self.init_hf_trainer(
            train_data,
            verbose=verbose,
            saved=True,
            show_progress=show_progress,
            hf_callbacks=[PretrainSaveCallback(self)],
            training_args=pretrain_args,
        )

        self.hf_trainer.train()

        return self.best_valid_score, self.best_valid_result

    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False, task=KnowledgeEvaluationType.REC
    ):
        if load_best_model and self.model.train_stage == "pretrain":
            self.hf_trainer.state.best_model_checkpoint = self.hf_trainer.args.output_dir

        return super().evaluate(
            eval_data,
            load_best_model=load_best_model,
            model_file=model_file,
            show_progress=show_progress,
            task=task,
        )

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if self.model.train_stage == "pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "finetune":
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError(f"Please make sure that the 'train_stage' is in [{self.model.TRAIN_STAGES}]!")
