from collections import defaultdict
from time import time

import torch
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    IntervalStrategy,
    LogitsProcessorList,
    Trainer,
    TrainerCallback,
)

from hopwise.model.layers import ConstrainedLogitsProcessorWordLevel
from hopwise.utils import (
    PathLanuageModelingTokenType,
    dict2str,
    early_stopping,
    get_gpu_usage,
    set_color,
)


def normalize_tuple(logits_tuple):
    # Normalize each tensor in the tuple
    normalized_tuple = tuple(torch.softmax(logits, dim=-1) for logits in logits_tuple)
    return normalized_tuple


class RankerLP:
    def __init__(self, tokenizer, kg_positives, K=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.kg_positives = kg_positives
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)
        self.max_new_tokens = max_new_tokens
        self.K = K

    def update_topk(self, generate_outputs):
        sorted_scores = generate_outputs.sequences_scores.argsort(descending=True)
        generate_outputs.sequences = generate_outputs.sequences[sorted_scores]
        for sequence in generate_outputs.sequences:
            seq = self.tokenizer.decode(sequence).split(" ")
            head_eid = int(seq[1][1:])
            rel_rid = int(seq[2][1:])
            if len(self.topk[head_eid, rel_rid]) >= self.K:
                continue
            recommended_token = seq[-1]
            recommended_item = int(recommended_token[1:])
            if (
                recommended_item in self.kg_positives[(head_eid, rel_rid)]
                or recommended_item in self.topk[head_eid, rel_rid]
            ):
                continue
            self.topk[head_eid, rel_rid].append(recommended_item)
            self.topk_sequences[head_eid, rel_rid].append(seq)

    def reset_topks(self):
        del self.topk
        del self.topk_sequences
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)


class CumulativeSequenceScoreRanker:
    def __init__(self, tokenizer, used_ids, item_num, topk=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.used_ids = used_ids
        self.item_num = item_num
        self.max_new_tokens = max_new_tokens
        self.topk = topk

    def calculate_sequence_scores(self, normalized_tuple, sequences):
        last_5_tokens = sequences[:, -self.max_new_tokens :]
        sequence_scores = []
        # Iterate over each tensor in the normalized tuple
        for i in range(self.max_new_tokens):
            # Get the probabilities corresponding to the ith token in last_5_tokens
            probs = normalized_tuple[i].gather(1, last_5_tokens[:, [i]])
            sequence_scores.append(probs)
        # Convert the list of tensors into a single tensor
        sequence_scores = torch.cat(sequence_scores, dim=-1)
        # Calculate the average score over the last 5 positions for each sequence
        sequence_scores = sequence_scores.mean(dim=-1)
        return sequence_scores

    def get_sequences(self, batch_len, generation_outputs):
        user_num = batch_len
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = {}

        # TODO: is it really necessary this scorer?
        #       the ordering is made on the last 5 tokens, but original sequences_scores values are likely more robust
        normalized_scores = normalize_tuple(generation_outputs.scores)
        normalized_sequences_scores = self.calculate_sequence_scores(normalized_scores, generation_outputs.sequences)
        ######################################################

        sequences = generation_outputs.sequences
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        sorted_indices = normalized_sequences_scores.argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_sequences_scores = normalized_sequences_scores[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]
        for user_index, sequence, sequence_score in zip(
            sorted_batch_user_index, sorted_sequences, sorted_sequences_scores
        ):
            seq = self.tokenizer.decode(sequence).split(" ")

            uid_token = seq[0]
            recommended_token = seq[-1]
            if not (
                uid_token.startswith(PathLanuageModelingTokenType.USER.value)
                and recommended_token.startswith(PathLanuageModelingTokenType.ITEM.value)
            ):
                continue

            uid = int(uid_token[1:])
            recommended_item = int(recommended_token[1:])
            if (
                torch.isfinite(scores[user_index, recommended_item])  # already scored
                or recommended_item in self.used_ids[uid]
            ):
                continue

            scores[user_index, recommended_item] = sequence_score
            if user_index not in user_topk_sequences:
                user_topk_sequences[uid] = [seq]
            else:
                user_topk_sequences[uid].append(seq)

        return scores, user_topk_sequences


def get_tokenized_used_ids(used_ids, tokenizer):
    """
    Convert the used ids to tokenized ids for the user and item tokens.
    Args:
        used_ids (dict): A dictionary where keys are user ids and values are lists of item ids.
        tokenizer: The tokenizer to convert ids to tokenized ids.
    Returns:
        dict: A dictionary where keys are tokenized user ids and values are lists of tokenized item ids.
    """
    user_token_type = PathLanuageModelingTokenType.USER.value
    item_token_type = PathLanuageModelingTokenType.ITEM.value

    tokenized_used_ids = {}
    for uid in range(used_ids.shape[0]):
        uid_token = tokenizer.convert_tokens_to_ids(user_token_type + str(uid))
        tokenized_used_ids[uid_token] = set(
            [tokenizer.convert_tokens_to_ids(item_token_type + str(item)) for item in used_ids[uid]]
        )
    return tokenized_used_ids


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
        # paths_per_head=50,  # n_sequences_lp
        # n_beams_lp=50,
        eval_device="cpu",
    ):
        hopwise_dataset = hopwise_train_data.dataset
        tokenizer = hopwise_dataset.tokenizer
        super().__init__(
            model=model,
            args=args,
            callbacks=None,
            train_dataset=hopwise_dataset.tokenized_dataset["train"],
            eval_dataset="none",
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Overwrite the callbacks to only use the HopwiseCallback
        self.callback_handler.callbacks = callbacks

        self.hopwise_config = hopwise_config

        self.paths_per_user = paths_per_user
        # self.N_RET_SEQ_LP = n_sequences_lp
        self.path_generation_args = path_generation_args
        # self.N_BEAMS_LP = n_beams_lp

        self.path_hop_length = path_hop_length
        self.eval_device = eval_device

        used_ids = hopwise_train_data.general_dataloader._sampler.used_ids
        tokenized_used_ids = get_tokenized_used_ids(used_ids, self.processing_class)

        # path_hop_length = n_relations => (n_relations + user_starting_node) + n_relations
        self.token_sequence_length = (1 + path_hop_length) + path_hop_length

        # TODO: add inference template as config param and use that instead of the hardcoded values
        ranker_max_new_tokens = self.token_sequence_length - 2
        self.ranker_rec = CumulativeSequenceScoreRanker(
            self.processing_class,
            used_ids,
            hopwise_dataset.item_num,
            topk=10,
            max_new_tokens=ranker_max_new_tokens,
        )

        # Link Prediction Data
        # self.SEQUENCE_LEN_LP = 3 + 1
        # self.test_set_lp = get_set_lp(dataset_name, "test")
        # heads_lp = [head for head, rel in self.test_set_lp.keys()]
        # relations_lp = [rel for head, rel in self.test_set_lp.keys()]

        # self.product_entities = [int(h) for h in get_dataset_id2eid(dataset_name, "product").values()]
        # self.all_entities, self.positive_triplets, self.positive_triplets_token_ids = (
        #     get_kg_positives_and_tokens_ids_lp(dataset_name, tokenizer)
        # )

        # def init_condition_fn_lp(head, rel):
        #     return f"[BOS] E{head} R{rel}" if head not in self.product_entities else f"[BOS] P{head} R{rel}"

        # self.inference_paths_lp = {
        #     "eid_rid": [init_condition_fn_lp(head, rel) for head, rel in zip(heads_lp, relations_lp)]
        # }
        # self.ranker_lp = RankerLP(
        #     tokenizer, kg_positives=self.positive_triplets, K=10, max_new_tokens=self.SEQUENCE_LEN_LP
        # )
        # self.test_dataset_lp = Dataset.from_dict(self.inference_paths_lp)
        # print(f"Sequence length rec: {self.token_sequence_length}")  # , lp: {self.SEQUENCE_LEN_LP}")

        self.logits_processor_rec = LogitsProcessorList(
            [
                ConstrainedLogitsProcessorWordLevel(
                    hopwise_dataset.get_tokenized_ckg(),
                    tokenized_used_ids,
                    self.token_sequence_length,
                    self.processing_class,
                    self.paths_per_user,
                    task=ConstrainedLogitsProcessorWordLevel.RECOMMENDATION_TASK,
                )
            ]
        )

        # self.logits_processor_lp = LogitsProcessorList(
        #     [
        #         ConstrainedLogitsProcessorLP(
        #             tokenized_kg=tokenized_kg,
        #             positive_token_map=self.positive_triplets_token_ids,
        #             tokenizer=tokenizer,
        #             total_length=self.SEQUENCE_LEN_LP,
        #             num_return_sequences=self.paths_per_user,
        #             eos_token_ids=[self.processing_class.convert_tokens_to_ids(self.processing_class.eos_token)],
        #         )
        #     ]
        # )

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
        scores, user_topk_sequences = ranker.get_sequences(inputs["input_ids"].shape[0], outputs)

        return scores

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
    ):
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

        len_hf_dataloader = len(self.train_data.dataset.tokenized_dataset["train"])
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
        elif (epoch_idx + 1) % self.hopwise_trainer.eval_step == 0:
            control.should_evaluate = True
            self.valid_start_time = time()
        else:
            control.should_evaluate = False

        return control

    def on_step_end(self, args, state, control, **kwargs):
        control.should_log = True
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
        self.hopwise_trainer.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
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
