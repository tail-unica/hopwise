# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/16, 2020/8/6, 2020/8/25, 2023/4/24
# @Author : Zhen Tian, Shanlei Mu, Yupeng Hou, Chenglong Ma
# @Email  : chenyuwuxinn@gmail.com, slmu@ruc.edu.cn, houyupeng@ruc.edu.cn, chenglong.m@outlook.com

"""hopwise.model.abstract_recommender
##################################
"""

from logging import getLogger

import numpy as np
import torch
from torch import nn

from hopwise.model.layers import FLEmbedding, FMEmbedding, FMFirstOrderLinear
from hopwise.model.logits_processor import LogitsProcessorList
from hopwise.utils import (
    FeatureSource,
    FeatureType,
    GenerationOutputs,
    InputType,
    KnowledgeEvaluationType,
    ModelType,
    PathLanguageModelingTokenType,
    get_logits_processor,
    get_sequence_postprocessor,
    set_color,
)


class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def __init__(self, _skip_nn_module_init=False):
        self.logger = getLogger()

        if not _skip_nn_module_init:
            super().__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""Full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def full_sort_predict_kg(self, interaction):
        r"""Full sort prediction KG function.
        Given heads, calculate the scores between heads and all candidate tails.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given heads and all candidate tails,
            shape: [n_batch_heads * n_candidate_tails]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color("\nTrainable parameters", "blue") + f": {params}"


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super().__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]


class AutoEncoderMixin:
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(self.history_item_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1, device=self.device).repeat(user.shape[0], self.n_items)
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        return rating_matrix


class SequentialRecommender(AbstractRecommender):
    """This is a abstract sequential recommender. All the sequential model should implement This class."""

    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super().__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class KnowledgeRecommender(AbstractRecommender):
    """This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
    The base knowledge-based recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.KNOWLEDGE

    def __init__(self, config, dataset, _skip_nn_module_init=False):
        super().__init__(_skip_nn_module_init=_skip_nn_module_init)

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.ENTITY_ID = config["ENTITY_ID_FIELD"]
        self.RELATION_ID = config["RELATION_ID_FIELD"]
        self.HEAD_ENTITY_ID = config["HEAD_ENTITY_ID_FIELD"]
        self.TAIL_ENTITY_ID = config["TAIL_ENTITY_ID_FIELD"]
        self.NEG_TAIL_ENTITY_ID = config["NEG_PREFIX"] + self.TAIL_ENTITY_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)

        # load parameters info
        if not _skip_nn_module_init:
            self.device = config["device"]


class ExplainableRecommender:
    """This is a abstract explainable-based recommender. All the explainable-based model should implement this class.
    This class use templates to make the explanation more interpretable.

    """

    def explain(self, interaction):
        r"""
        Explain the prediction function.

        Given users, calculate the scores and paths between users and all candidate items,
        then return the templates filled with path data.

        Args:
            interaction (Interaction): The interaction batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
                with shape [n_batch_users * n_candidate_items].
            pandas.DataFrame: Explanation of the prediction, containing paths and corresponding templates,
                with shape [n_paths * [uid, pid, score, template1, template2, ..., #templates]].
        """
        raise NotImplementedError("explain is not implemented")

    def decode_path(self, path):
        r"""
        Decode the path into a string. Path decoding is specific to each model.

        Args:
            path (list): The path data.

        Returns:
            str: The decoded path string.
        """
        raise NotImplementedError("decode_path is not implemented")


class PathLanguageModelingRecommender(KnowledgeRecommender):
    """This is an abstract path-language-modeling recommender.
    All the path-language-modeling model should implement this class.
    The base path-language-modeling recommender class inherits the knowledge-aware recommender class to
    learn from knowledge graph paths defined by a chain of entity-relation triplets.
    """

    type = ModelType.PATH_LANGUAGE_MODELING
    input_type = InputType.PATHWISE

    def __init__(self, config, dataset, _skip_nn_module_init=True):
        super().__init__(config, dataset, _skip_nn_module_init=_skip_nn_module_init)

        self.n_tokens = len(dataset.tokenizer)
        self.token_sequence_length = dataset.token_sequence_length - 1  # EOS token is not included

        logits_processor = get_logits_processor(config["model"])(
            tokenized_ckg=dataset.get_tokenized_ckg(),
            tokenized_used_ids=dataset.get_tokenized_used_ids(),
            max_sequence_length=self.token_sequence_length,
            tokenizer=dataset.tokenizer,
            task=KnowledgeEvaluationType.REC,
        )
        self.logits_processor_list = LogitsProcessorList([logits_processor])

        self.sequence_postprocessor = get_sequence_postprocessor(config["sequence_postprocessor"])(
            dataset.tokenizer,
            dataset.get_user_used_ids(),
            dataset.item_num,
            topk=config["topk"],
        )

    @torch.no_grad()
    def generate(self, inputs, top_k=None, paths_per_user=1, **kwargs):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            inputs (dict): A dictionary containing the input_ids tensor with shape (b, t).
            top_k (int, optional): If specified, only the top k logits will be considered
                for sampling at each step. Defaults to None.
            paths_per_user (int, optional): How many paths to return for each user.
            **kwargs: Additional keyword arguments for the model. In future, it can be used to pass
                other generation parameters such as temperature, repetition penalty, etc.
        """
        max_new_tokens = self.token_sequence_length - inputs["input_ids"].size(1)

        # How many paths to return?
        inputs["input_ids"] = inputs["input_ids"].repeat_interleave(paths_per_user, dim=0)
        scores = torch.full((inputs["input_ids"].size(0), max_new_tokens, self.n_tokens), -torch.inf).to(self.device)
        for i in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.predict(inputs)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature

            # KGCD
            logits = self.logits_processor_list(inputs["input_ids"], logits)

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores[:, i] = probs
            # sample from the distribution
            path_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            inputs["input_ids"] = torch.cat((inputs["input_ids"], path_next), dim=1)

        return GenerationOutputs(sequences=inputs["input_ids"], scores=torch.unbind(scores, dim=1))


class ExplainablePathLanguageModelingRecommender(PathLanguageModelingRecommender, ExplainableRecommender):
    """This is an abstract explainable path-language-modeling recommender.
    All the explainable path-language-modeling model should implement this class.
    The base explainable path-language-modeling recommender class inherits the path-language-modeling recommender class
    to learn from knowledge graph paths defined by a chain of entity-relation triplets.
    """

    def __init__(self, config, dataset, _skip_nn_module_init=True):
        super().__init__(config, dataset, _skip_nn_module_init=_skip_nn_module_init)

    def explain(self, inputs, **kwargs):
        kwargs["max_length"] = self.token_sequence_length
        kwargs["min_length"] = self.token_sequence_length
        outputs = self.generate(inputs, **kwargs)

        max_new_tokens = self.token_sequence_length - inputs["input_ids"].size(1)

        scores, sequences = self.sequence_postprocessor.get_sequences(outputs, max_new_tokens=max_new_tokens)

        for seq in sequences:
            seq[-1] = self.decode_path(seq[-1])

        return scores, sequences

    def decode_path(self, path):
        """Standardize path format"""
        new_path = []
        # Process the path
        # [BOS] U R I R E/I R I
        for node_idx in range(1, len(path) + 1, 2):
            if path[node_idx].startswith(PathLanguageModelingTokenType.USER.token):
                user_id = int(path[node_idx][1:])
                if node_idx - 1 == 0:
                    relation = "self_loop"
                else:
                    relation = int(path[node_idx - 1][1:])

                new_node = (relation, "user", user_id)
            elif path[node_idx].startswith(PathLanguageModelingTokenType.ITEM.token):
                relation = int(path[node_idx - 1][1:])
                item_id = int(path[node_idx][1:])
                new_node = (relation, "item", item_id)
            else:
                # Is an entity
                relation = int(path[node_idx - 1][1:])
                entity_id = int(path[node_idx][1:])
                new_node = (relation, "entity", entity_id)
            new_path.append(new_node)
        return new_path


class ContextRecommender(AbstractRecommender):
    """This is a abstract context-aware recommender. All the context-aware model should implement this class.
    The base context-aware recommender class provide the basic embedding function of feature fields which also
    contains a first-order part of feature fields.
    """

    type = ModelType.CONTEXT
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__()

        self.field_names = dataset.fields(
            source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ]
        )
        self.LABEL = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.double_tower = config["double_tower"]
        self.numerical_features = config["numerical_features"]
        if self.double_tower is None:
            self.double_tower = False
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.float_seq_field_names = []
        self.float_seq_field_dims = []
        self.num_feature_field = 0

        if self.double_tower:
            self.user_field_names = dataset.fields(source=[FeatureSource.USER, FeatureSource.USER_ID])
            self.item_field_names = dataset.fields(source=[FeatureSource.ITEM, FeatureSource.ITEM_ID])
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += 1
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += 1

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.FLOAT and field_name in self.numerical_features:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.FLOAT_SEQ and field_name in self.numerical_features:
                self.float_seq_field_names.append(field_name)
                self.float_seq_field_dims.append(dataset.num(field_name))
            else:
                continue

            self.num_feature_field += 1
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size
            )
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array((0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.long)
            self.float_embedding_table = FLEmbedding(
                self.float_field_dims, self.float_field_offsets, self.embedding_size
            )
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, self.embedding_size))
        if len(self.float_seq_field_dims) > 0:
            self.float_seq_embedding_table = nn.ModuleList()
            for float_seq_field_dim in self.float_seq_field_dims:
                self.float_seq_embedding_table.append(nn.Embedding(float_seq_field_dim, self.embedding_size))

        self.first_order_linear = FMFirstOrderLinear(config, dataset)

    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return None
        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(float_fields)

        return float_embedding

    def embed_float_seq_fields(self, float_seq_fields, mode="mean"):
        """Embed the float feature columns

        Args:
            float_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len, 2]
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            float_seq_embedding = base * embedding_table(index.long())  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(float_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_float_seq_embedding = float_seq_embedding - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_float_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode="mean"):
        """Embed the token feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.

        """
        if not self.double_tower:
            raise RuntimeError("Please check your model hyper parameters and set 'double tower' as True")
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = torch.split(
                dense_embedding,
                [self.user_float_field_num, self.item_float_field_num],
                dim=1,
            )
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [
                self.user_token_seq_field_num,
                self.item_token_seq_field_num,
                self.user_token_field_num,
                self.item_token_field_num,
            ]
            (
                first_token_seq_embedding,
                second_token_seq_embedding,
                first_token_embedding,
                second_token_embedding,
            ) = torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat([first_token_seq_embedding, first_token_embedding], dim=1)
            second_sparse_embedding = torch.cat([second_token_seq_embedding, second_token_embedding], dim=1)
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (
            first_sparse_embedding,
            first_dense_embedding,
            second_sparse_embedding,
            second_dense_embedding,
        )

    def concat_embed_input_fields(self, interaction):
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:  # noqa: PLR2004
            all_embeddings.append(dense_embedding)
        return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:  # noqa: PLR2004
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field, 2]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])

        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)

        if float_fields_embedding is None:
            dense_embedding = float_seq_fields_embedding
        elif float_seq_fields_embedding is None:
            dense_embedding = float_fields_embedding
        else:
            dense_embedding = torch.cat([float_seq_fields_embedding, float_fields_embedding], dim=1)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field, 2]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        elif token_seq_fields_embedding is None:
            sparse_embedding = token_fields_embedding
        else:
            sparse_embedding = torch.cat([token_seq_fields_embedding, token_fields_embedding], dim=1)

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field, 2] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding
