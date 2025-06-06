# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

# UPDATE:
# @Time   : 2025
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

from time import time

import torch
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, IntervalStrategy, Trainer, TrainerCallback

from hopwise.utils import (
    dict2str,
    early_stopping,
    get_gpu_usage,
    get_logits_processor,
    get_ranker,
    get_tokenized_used_ids,
    set_color,
)
from hopwise.utils.enum_type import KnowledgeEvaluationType


class HFPathTrainer(Trainer):
    def __init__(
        self,
        hopwise_train_data,
        hopwise_config,
        callbacks,
        model=None,
        args=None,
        path_hop_length=3,
        paths_per_user=10,
        path_generation_args=None,
        eval_device="cpu",
        tokenizer=None,
    ):
        hopwise_dataset = hopwise_train_data.dataset
        tokenizer = tokenizer or hopwise_dataset.tokenizer
        super().__init__(
            model=model,
            args=args,
            callbacks=None,
            train_dataset=hopwise_train_data.tokenized_dataset["train"],
            eval_dataset="none",
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Overwrite the callbacks to only use the HopwiseCallback
        self.callback_handler.callbacks = callbacks

        self.hopwise_config = hopwise_config
        self.paths_per_user = paths_per_user
        self.path_generation_args = path_generation_args

        self.path_hop_length = path_hop_length
        self.token_sequence_length = hopwise_train_data.token_sequence_length
        self.eval_device = eval_device

        used_ids = hopwise_train_data.general_dataloader._sampler.used_ids
        tokenized_used_ids = get_tokenized_used_ids(used_ids, self.processing_class)

        # path_hop_length = n_relations => (n_relations + user_starting_node) + n_relations + 2 (BOS, EOS)
        self.token_sequence_length = (1 + path_hop_length) + path_hop_length + 1

        # TODO: add inference template as config param and use that instead of the hardcoded values
        ranker_max_new_tokens = self.token_sequence_length - 2

        ranker_params = dict(
            processing_class=self.processing_class,
            used_ids=used_ids,
            item_num=hopwise_dataset.item_num,
            max_new_tokens=ranker_max_new_tokens,
        )
        logits_processor_params = dict(
            tokenized_ckg=hopwise_dataset.get_tokenized_ckg(),
            tokenized_used_ids=tokenized_used_ids,
            token_sequence_length=self.token_sequence_length,
            processing_class=self.processing_class,
            paths_per_user=self.paths_per_user,
            task=KnowledgeEvaluationType.REC,
        )

        self.ranker_rec = get_ranker(self.hopwise_config, ranker_params)
        self.logits_processor_rec = get_logits_processor(self.hopwise_config, logits_processor_params)

    def _full_sort_batch_eval(self, inputs, task="rec"):
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if task == "rec":
            sequence_len = self.token_sequence_length
            num_return_sequences = self.paths_per_user
            logits_processor = self.logits_processor_rec
            ranker = self.ranker_rec
        else:
            sequence_len = self.SEQUENCE_LEN_LP
            num_return_sequences = self.N_RET_SEQ_LP
            logits_processor = self.logits_processor_lp
            ranker = self.ranker_lp
        outputs = model.generate(
            **inputs,
            max_length=sequence_len,
            min_length=sequence_len,
            num_return_sequences=num_return_sequences,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
            **self.path_generation_args,
        )

        scores, user_topk_sequences, avg_topk_size = ranker.get_sequences(inputs["input_ids"].shape[0], outputs)

        return scores, user_topk_sequences, avg_topk_size

    def evaluate(self, **kwargs):
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=None)

        return self.control.metrics


class HopwiseCallback(TrainerCallback):
    """It handles the training and evaluation communication with the hopwise and HuggingFace trainers."""

    def __init__(
        self,
        hopwise_trainer,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
        train_phase="finetune",
        model=None,
        model_name=None,
    ):
        self.model = model
        self.model_name = model_name
        self.hopwise_trainer = hopwise_trainer
        self.train_data = train_data
        self.valid_data = valid_data
        self.verbose = verbose
        self.saved = saved
        self.show_progress = show_progress
        self.callback_fn = callback_fn
        self.train_phase = train_phase

    def on_train_begin(self, args, state, control, **kwargs):
        self.hopwise_trainer.eval_collector.train_data_collect(self.train_data)
        if self.hopwise_trainer.config["train_neg_sample_args"].get("dynamic", False):
            self.train_data.get_model(self.hopwise_trainer.model)
        self.valid_step = 0

    def on_train_end(self, args, state, control, **kwargs):
        self.hopwise_trainer._add_hparam_to_tensorboard(self.hopwise_trainer.best_valid_score)
        return super().on_train_end(args, state, control, **kwargs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.training_start_time = time()

        len_hf_dataloader = len(self.train_data.tokenized_dataset["train"])
        steps_in_epoch = len_hf_dataloader // self.hopwise_trainer.config["train_batch_size"]
        steps_in_epoch += int(len_hf_dataloader % self.hopwise_trainer.config["train_batch_size"] > 0)
        self.progress_bar = (
            tqdm(
                total=steps_in_epoch,
                ncols=100,
                desc=set_color(f"Train {int(state.epoch):>5}", "pink"),
            )
            if self.show_progress
            else range(steps_in_epoch)
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.show_progress:
            self.progress_bar.close()
        training_end_time = time()
        # Retrieve training loss and other information
        if state.log_history:
            epoch_idx = state.epoch
            train_loss = state.log_history[-1].get("loss")
            self.hopwise_trainer.train_loss_dict[epoch_idx] = train_loss
            train_loss_output = self.hopwise_trainer._generate_train_loss_output(
                epoch_idx, self.training_start_time, training_end_time, train_loss
            )
            if self.verbose:
                self.hopwise_trainer.logger.info(train_loss_output)
            self.hopwise_trainer._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.hopwise_trainer.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

        if self.hopwise_trainer.eval_step <= 0 or not self.valid_data:
            if self.saved:
                control.should_save = True
        elif epoch_idx % self.hopwise_trainer.eval_step == 0:
            control.should_evaluate = True
            self.valid_start_time = time()
        else:
            control.should_evaluate = False

        # update attentive-a
        if hasattr(self.model, "update_attentive_A"):
            with torch.no_grad():
                self.model.update_attentive_A()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        control.should_log = True
        if self.show_progress:
            self.progress_bar.update(1)
        if self.hopwise_trainer.gpu_available and self.show_progress:
            gpu_usage = get_gpu_usage(self.hopwise_trainer.device)
            self.progress_bar.set_postfix_str(set_color("GPU RAM: " + gpu_usage, "yellow"))

        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            # Save the model at the end if we have a save strategy
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True

        return control

    def on_evaluate(self, args, state, control, **kwargs):
        epoch_idx = state.epoch
        valid_score, valid_result = self.hopwise_trainer._valid_epoch(
            self.valid_data, show_progress=self.show_progress
        )

        (
            self.hopwise_trainer.best_valid_score,
            self.hopwise_trainer.cur_step,
            stop_flag,
            update_flag,
        ) = early_stopping(
            valid_score,
            self.hopwise_trainer.best_valid_score,
            self.hopwise_trainer.cur_step,
            max_step=self.hopwise_trainer.stopping_step,
            bigger=self.hopwise_trainer.valid_metric_bigger,
        )
        valid_end_time = time()
        valid_score_output = (
            set_color("epoch %d evaluating", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
            + set_color("valid_score", "blue")
            + ": %f]"
        ) % (epoch_idx, valid_end_time - self.valid_start_time, valid_score)
        valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
        if self.verbose:
            self.hopwise_trainer.logger.info(valid_score_output)
            self.hopwise_trainer.logger.info(valid_result_output)
        self.hopwise_trainer.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
        self.hopwise_trainer.wandblogger.log_metrics({**valid_result, "valid_step": self.valid_step}, head="valid")

        if not self.hopwise_trainer.valid_metric.startswith("eval_"):
            metric_to_check = f"eval_{self.hopwise_trainer.valid_metric}"
            control.metrics = {**valid_result, metric_to_check: valid_score}

        if update_flag:
            if self.saved:
                control.should_save = True
                self.hopwise_trainer._save_checkpoint(epoch_idx, verbose=self.verbose)

            self.hopwise_trainer.best_valid_result = valid_result

        if self.callback_fn:
            self.callback_fn(epoch_idx, valid_score)

        if stop_flag:
            stop_output = "Finished training, best eval result in epoch %d" % (
                epoch_idx - self.hopwise_trainer.cur_step * self.hopwise_trainer.eval_step
            )
            if self.verbose:
                self.hopwise_trainer.logger.info(stop_output)
            control.should_training_stop = True

        self.valid_step += 1

        return control
