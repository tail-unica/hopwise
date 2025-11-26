# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/09/28, 2020/08/09
# @Author  :   Kaiyuan Li, Zhichao Feng
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com

"""hopwise.evaluator.utils
################################
"""

import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def pad_sequence(sequences, len_list, pad_to=None, padding_value=0):
    """Pad sequences to a matrix

    Args:
        sequences (list): list of variable length sequences.
        len_list (list): the length of the tensors in the sequences
        pad_to (int, optional): if pad_to is not None, the sequences will pad to the length you set,
                                else the sequence will pad to the max length of the sequences.
        padding_value (int, optional): value for padded elements. Default: 0.

    Returns:
        torch.Tensor: [seq_num, max_len] or [seq_num, pad_to]

    """
    max_len = np.max(len_list) if pad_to is None else pad_to
    min_len = np.min(len_list)
    device = sequences[0].device
    if max_len == min_len:
        result = torch.cat(sequences, dim=0).view(-1, max_len)
    else:
        extra_len_list = np.subtract(max_len, len_list).tolist()
        padding_nums = max_len * len(len_list) - np.sum(len_list)
        padding_tensor = torch.tensor([-np.inf], device=device).repeat(padding_nums)
        padding_list = torch.split(padding_tensor, extra_len_list)
        result = list(itertools.chain.from_iterable(zip(sequences, padding_list)))
        result = torch.cat(result)

    return result.view(-1, max_len)


def trunc(scores, method):
    """Round the scores by using the given method

    Args:
        scores (numpy.ndarray): scores
        method (str): one of ['ceil', 'floor', 'around']

    Raises:
        NotImplementedError: method error

    Returns:
        numpy.ndarray: processed scores
    """
    try:
        cut_method = getattr(np, method)
    except NotImplementedError:
        raise NotImplementedError(f"module 'numpy' has no function named '{method}'")
    scores = cut_method(scores)
    return scores


def cutoff(scores, threshold):
    """Cut of the scores based on threshold

    Args:
        scores (numpy.ndarray): scores
        threshold (float): between 0 and 1

    Returns:
        numpy.ndarray: processed scores
    """
    return np.where(scores > threshold, 1, 0)


def _binary_clf_curve(trues, preds):
    """Calculate true and false positives per binary classification threshold

    Args:
        trues (numpy.ndarray): the true scores' list
        preds (numpy.ndarray): the predict scores' list

    Returns:
        fps (numpy.ndarray): A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]
        preds (numpy.ndarray): An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i].

    Note:
        To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
        in SkLearn and made some optimizations.

    """
    trues = trues == 1

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps


def plot_tsne_embeddings(model, **kwargs):
    import plotly.express as px

    embeddings_list = list()
    identifiers_list = list()

    for embeddings_name, embeddings in kwargs.items():
        embeddings_list.append(embeddings)
        identifiers_list.extend([f"{embeddings_name} {id}" for id in range(embeddings.shape[0])])

    embeddings_list = np.concatenate(embeddings_list, axis=0)

    combined_df = pd.DataFrame(
        {
            "x": embeddings_list[:, 0],
            "y": embeddings_list[:, 1],
            "type": [id.split(" ")[0] for id in identifiers_list],
            "identifier": identifiers_list,
        }
    )

    fig = px.scatter(
        combined_df,
        x="x",
        y="y",
        color="type",
        hover_data=["identifier"],
        labels={
            "x": "Embedding Dimension 1",
            "y": "Embedding Dimension 2",
        },
        title="Visualising Combined Embeddings",
        width=1024,
        height=1024,
        template="plotly_white",
    )
    current_datetime = datetime.now()

    fig.write_html(f"{model} tSNE {current_datetime.strftime('%b-%d-%Y_%H-%M-%S')}.html")


def train_tsne(model, config, load_best_model):
    from openTSNE import TSNE

    tsne = TSNE(
        perplexity=config["perplexity"],
        n_jobs=config["n_jobs"],
        initialization=config["initialization"],
        metric=config["metric"],
        random_state=config["seed"],
        verbose=config["verbose"],
    )

    if (
        (config["plot_on"] == "test" and load_best_model)
        or config["plot_on"] == "validation"
        or config["plot_on"] is not None
    ):
        try:
            tsne_user_embeddings = tsne.fit(model.user_embedding.weight.cpu().detach().numpy())
            tsne_entity_embeddings = tsne.fit(model.entity_embedding.weight.cpu().detach().numpy())
            tsne_relation_embeddings = tsne.fit(model.relation_embedding.weight.cpu().detach().numpy())
        except AttributeError:
            print(
                "The model does not have the required embeddings for the t-SNE display, \
                please check the name of the embeddings: user_embedding, entity_embedding, relation_embedding."
            )

        plot_tsne_embeddings(
            model=config["model"],
            user=tsne_user_embeddings,
            entity=tsne_entity_embeddings,
            relation=tsne_relation_embeddings,
        )
