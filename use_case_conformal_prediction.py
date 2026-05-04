# @Time   : 2026/03/30
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it


"""Case study example
===================

"""

import importlib
import os
import urllib.request
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Callable

import pandas as pd
import torch

from hopwise.quick_start import load_minimal_data_and_model
from hopwise.utils import get_trainer

ALPHAS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9]


def timer(func: Callable) -> Callable:
    def _format_elapsed(seconds: float) -> str:
        seconds_int = int(round(seconds))
        hours, remainder = divmod(seconds_int, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        return results, _format_elapsed(run_time)

    return wrapper


def _checkpoint_display_name(path):
    return Path(path).name


def get_model_family():
    return {
        "BPR": "Pairwise, Non-Sequential",
        "LightGCN": "Pairwise, Non-Sequential",
        "DiffRec": "Generative, Reconstruction-Based Non Sequential",
        "MultiVAE": "Generative, Reconstruction-Based Non Sequential",
        "NeuMF": "Pointwise",
        "DMF": "Pointwise",
        "SASRec": "Pointwise Sequential",
        "BERT4Rec": "Pointwise Sequential",
    }


def _notify_checkpoint(topic, message, base_url="https://ntfy.sh"):
    if not topic:
        return
    try:
        url = f"{base_url.rstrip('/')}/{topic}"
        data = message.encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        # Avoid failing the run if notification fails.
        pass


def _compute_user_stats_from_scores(calibrator, scores, positive_u=None, positive_i=None, lhat=None):
    if positive_u is not None or positive_i is not None:
        calibrator.metric.set_positive_data(positive_u, positive_i)

    if lhat is not None:
        thresholded_scores = torch.where(scores >= lhat, scores, -torch.inf)
    else:
        thresholded_scores = scores

    topk = torch.topk(thresholded_scores, calibrator.topk, dim=-1)

    # user topk set size
    user_set_sizes = torch.sum(torch.isfinite(topk.values), dim=1).to(torch.float32)

    calibrator.metric.set_metric_data(thresholded_scores, topk.indices)

    return thresholded_scores, user_set_sizes


def _select_lhat(bound, lambdas, alpha):
    feasible_lhats = torch.where(bound <= alpha)[0]

    if len(feasible_lhats) == 0:
        # Nothing satisfies risk: use smallest lambda (largest set) as safest fallback.
        lhat_idx = 0
        return lambdas[lhat_idx], lhat_idx

    lhat_idx = feasible_lhats.max()

    return lambdas[lhat_idx], lhat_idx


def run_checkpoint_calibration(checkpoint_path):
    model_family = get_model_family()
    config, model, dataset, train_data, valid_data, test_data = load_minimal_data_and_model(model_file=checkpoint_path)
    model.eval()
    users = train_data.dataset.inter_feat.user_id
    items = train_data.dataset.inter_feat.item_id

    # calculate user popularity
    train_df = pd.DataFrame({"user_id": users.cpu().numpy(), "item_id": items.cpu().numpy()})
    user2pop = train_df.groupby("user_id").size().to_dict()
    user2pop[0] = 0  # add padding user with 0 interactions

    # unpack test data
    calib_data, test_split, full_test_data = test_data

    # set up trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.train_data_collect(train_data)

    # calibration configuration
    crc_cfg = config["calibration"]
    metric_cls = getattr(importlib.import_module("hopwise.evaluator.conformal_metrics"), crc_cfg["loss"])
    calibrator_cls = getattr(importlib.import_module("hopwise.model.conformal_calibration"), crc_cfg["calibrator"])

    eval_config = {
        "load_best_model": True,
        "model_file": checkpoint_path,
        "show_progress": config["show_progress"],
    }
    calibration_data = {
        "train": train_data,
        "calibration": calib_data,
        "test": test_split,
        "full_test": full_test_data,
    }

    checkpoint_name = _checkpoint_display_name(checkpoint_path)
    model_name = str(config["model"])
    dataset_name = str(config["dataset"])
    input_type = model_family[model_name]

    if config["train_neg_sample_args"]["distribution"] == "popularity" and model_name in ["SASRec", "BERT4Rec"]:
        input_type = "Pairwise Sequential"

    user_stats = []

    metric = metric_cls(config)
    calibrator = calibrator_cls(config, metric, trainer, trainer.logger, eval_config.copy(), calibration_data)
    before_calib_results, before_calib_scores, calib_positive_u, calib_positive_i = calibrator.forward(calib_data)

    # set positive data
    calibrator.metric.set_positive_data(calib_positive_u, calib_positive_i)
    # determine candidate threshold lambdas
    lambdas = calibrator.get_lambdas(before_calib_scores)
    # define loss and size tables for all lambdas on the calibration set
    calib_loss_table, calib_size_table = calibrator.get_loss_and_size_tables(before_calib_scores, lambdas)

    before_calib_scores, before_calib_users_topk_size = _compute_user_stats_from_scores(
        calibrator, before_calib_scores
    )

    """raw test evaluation pass"""
    # get raw scores and results on test set without thresholding for comparison
    test_results_raw, test_scores_raw, test_positive_u, test_positive_i = calibrator.forward(test_split)
    test_scores_raw, test_users_topk_size = _compute_user_stats_from_scores(
        calibrator, test_scores_raw, test_positive_u, test_positive_i
    )

    print(f"Running calibration for checkpoint: {checkpoint_name}")

    results_rows = []
    for alpha in ALPHAS:
        """Calibration phase"""
        # set alpha to test
        config["calibration"]["alpha"] = float(alpha)

        # calculate bound
        bound = calibrator.get_bound(calib_loss_table)
        # select lambda (threshold) based on bound and alpha
        lhat, lhat_idx = _select_lhat(bound, lambdas, alpha)

        """Thresholded calibration evaluation pass"""
        # get scores and results on calibration and test sets using the selected lambda (threshold)
        after_calib_results, after_calib_scores, _, _ = calibrator.forward(calib_data, threshold=lhat)
        after_calib_scores, after_calib_users_topk_size = _compute_user_stats_from_scores(
            calibrator, after_calib_scores, calib_positive_u, calib_positive_i
        )

        """Thresholded evaluation pass"""
        # get thresholded scores and results on test set using the selected lambda (threshold)
        thresholded_test_results, test_scores, _, _ = calibrator.forward(test_split, threshold=lhat)
        test_scores, thresholded_test_users_topk_size = _compute_user_stats_from_scores(
            calibrator, test_scores, test_positive_u, test_positive_i, lhat=lhat
        )

        data = (
            ("before_calibration", before_calib_users_topk_size, before_calib_results, before_calib_scores),
            ("after_calibration", after_calib_users_topk_size, after_calib_results, after_calib_scores),
            ("test", test_users_topk_size, test_results_raw, test_scores_raw),
            ("thresholded_test", thresholded_test_users_topk_size, thresholded_test_results, test_scores),
        )

        """calculate user specific metrics"""
        for split_name, users_topk_sizes, _, _ in data:
            for user, topk_size in enumerate(users_topk_sizes):
                row = {
                    "user": int(user),
                    "user_set_size": float(topk_size.item()),
                    "split_name": split_name,
                    "checkpoint": checkpoint_name,
                    "model": model_name,
                    "dataset": dataset_name,
                    "type": input_type,
                    "alpha": float(alpha),
                    "bound": float(bound[lhat_idx].item()),
                    "risk": float(calib_loss_table[:, lhat_idx].mean().item()),
                    "lambda": float(lhat),
                    "lambda_idx": int(lhat_idx),
                }
                user_stats.append(row)

        """save metrics results"""
        for split_name, _, split_metrics, _ in data:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "split": split_name,
                "checkpoint": checkpoint_name,
                "type": input_type,
                "alpha": float(alpha),
            }
            row.update(split_metrics)
            results_rows.append(row)

        print(f"Completed alpha={alpha:.2f} for checkpoint: {checkpoint_name}")

    user_stats_df = pd.DataFrame(user_stats)
    metrics_df = pd.DataFrame(results_rows)

    return user_stats_df, metrics_df


@timer
def run_all_checkpoints(checkpoints, notify_topic=None, notify_base_url="https://ntfy.sh"):
    processed_checkpoints = set()

    metrics_df = pd.DataFrame()
    users_stats_df = pd.DataFrame()

    total = len(checkpoints)
    print(f"Total checkpoints to process: {total}")
    for idx, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = _checkpoint_display_name(checkpoint_path)
        if checkpoint_name in processed_checkpoints:
            message = f"Skipped checkpoint {checkpoint_name} (already in partials)"
            _notify_checkpoint(notify_topic, message, base_url=notify_base_url)
            continue

        ckpt_users_stats_df, ckpt_metrics_df = run_checkpoint_calibration(checkpoint_path)

        users_stats_df = pd.concat([users_stats_df, ckpt_users_stats_df], ignore_index=True)
        metrics_df = pd.concat([metrics_df, ckpt_metrics_df], ignore_index=True)

        percent_done = ((idx + 1) / total) * 100 if total > 0 else 100.0
        message = f"Completed checkpoint {checkpoint_name} ({percent_done:.1f}%)"
        _notify_checkpoint(notify_topic, message, base_url=notify_base_url)
        processed_checkpoints.add(checkpoint_name)

    return users_stats_df, metrics_df


if __name__ == "__main__":
    os.makedirs("reproducibility_study", exist_ok=True)

    notify_topic = "xxxxxx"
    notify_base_url = os.getenv("NTFY_BASE_URL", "https://ntfy.sh")

    checkpoints = [
        "checkpoints/DMF-Apr-20-2026_12-08-57.pth",  # DMF, lastfm-1m
        "checkpoints/DMF-Apr-20-2026_12-01-57.pth",  # DMF, ml-1m
        "checkpoints/NeuMF-Apr-20-2026_11-48-44.pth",  # NeuMF, lastfm-1m
        "checkpoints/NeuMF-Apr-20-2026_11-44-49.pth",  # NeuMF, ml-1m
        "checkpoints/DiffRec-Apr-14-2026_15-47-15.pth",  # DiffRec, lastfm-1m
        "checkpoints/DiffRec-Apr-14-2026_15-42-24.pth",  # DiffRec, ml-1m
        "checkpoints/LightGCN-Apr-14-2026_08-21-25.pth",  # LightGCN, lastfm-1m
        "checkpoints/LightGCN-Apr-14-2026_08-06-28.pth",  # LightGCN, ml-1m
        "checkpoints/MultiVAE-Apr-14-2026_15-37-44.pth",  # MultiVAE, lastfm-1m
        "checkpoints/MultiVAE-Apr-14-2026_15-34-25.pth",  # MultiVAE, ml-1m
        "checkpoints/BPR-Apr-14-2026_07-59-00.pth",  # BPR, lastfm-1m
        "checkpoints/BPR-Apr-14-2026_07-53-41.pth",  # BPR, ml-1m
        "checkpoints/BERT4Rec-Apr-14-2026_17-40-37.pth",  # lastfm-1m, neg sampling
        "checkpoints/BERT4Rec-Apr-15-2026_08-16-05.pth",  # lastfm-1m
        "checkpoints/BERT4Rec-Apr-14-2026_09-03-10.pth",  # ml-1m
        "checkpoints/BERT4Rec-Apr-14-2026_15-36-11.pth",  # ml-1m, neg sampling
        "checkpoints/SASRec-Apr-14-2026_08-23-51.pth",  # SASRec, lastfm-1m
        "checkpoints/SASRec-Apr-14-2026_13-49-02.pth",  # SASRec, lastfm-1m, neg sampling
        "checkpoints/SASRec-Apr-14-2026_07-53-49.pth",  # SASRec, ml-1m
        "checkpoints/SASRec-Apr-14-2026_12-10-46.pth",  # SASRec, ml-1m, neg sampling
    ]

    (users_final_df, metrics_df), elapsed_time = run_all_checkpoints(
        checkpoints,
        notify_topic=notify_topic,
        notify_base_url=notify_base_url,
    )
    print(f"Completed all checkpoints in {elapsed_time}.")
    _notify_checkpoint(notify_topic, f"Completed all checkpoints in {elapsed_time}.", base_url=notify_base_url)

    # save results
    metrics_output_path = os.path.join("reproducibility_study", "metrics_results.tsv")
    metrics_df.to_csv(metrics_output_path, sep="\t", index=False)
    print(f"Saved summary results to: {metrics_output_path}")

    users_stats_output_path = os.path.join("reproducibility_study", "users_stats.tsv")
    users_final_df.to_csv(users_stats_output_path, sep="\t", index=False)
    print(f"Saved users stats database to: {users_stats_output_path}")
