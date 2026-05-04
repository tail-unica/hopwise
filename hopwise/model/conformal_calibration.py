# @Time   : 2026
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""hopwise.model.conformal_calibration
#############################
Model calibration using conformal prediction (Conformal Risk Control)

source: https://arxiv.org/abs/2208.02814, https://github.com/aangelopoulos/conformal-risk

"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from hopwise.utils import set_color


class AbstractCalibration:
    """
    Abstract class for calibration. It contains the common code for the calibration process, such as loading the data and the trainer, and defining the parameters for the conformal risk control. The actual calibration process is implemented in the Calibration class, which inherits from this abstract class.
    """  # noqa: E501

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        self.config = config
        self.logger = logger
        self.metric = metric
        self.crc_config = config["calibration"]
        self.topk = max(config["topk"])

        self.use_threads = False

        # data
        self.train_data = calibration_data["train"]
        self.calib_data = calibration_data["calibration"]
        self.test_data = calibration_data["test"]
        self.full_test_data = calibration_data["full_test"]

        # crc parameters
        self.alpha = self.crc_config["alpha"]
        # self.delta = self.crc_config['delta']
        self.n_lambdas = self.crc_config["n_lambdas"]
        self.maximum_metric_score = self.crc_config["maximum_metric_score"]

        # calibration parameters for expectation check
        self.n_trials = self.crc_config["n_trials"]
        self.n_calibration_users = int(
            (self.full_test_data.dataset.user_num - 1) * self.crc_config["n_calibration_users"]
        )

        # trainer and evaluation config
        self.trainer = trainer
        self.evaluation_config = eval_config

    def forward(self, data, **kwargs):
        raise NotImplementedError

    def calibrate(self):
        raise NotImplementedError


class ConformalRiskControl(AbstractCalibration):
    """Calibration is used to calibrate the model using conformal risk control."""

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        super().__init__(config, metric, trainer, logger, eval_config, calibration_data)

    def get_lambdas(self, conf_scores):
        """
        Return the thresholds to use to test the risk constraint. Instead of using all the possible predicted score by the full_sort_prediction function, we use a grid of lambda values choose randomly in a range between the minimum and the maximum values in conf_scores. # noqa: E501

        Args:
            conf_scores (n_users x topk size)): full_sort_prediction confidence scores at cut off k

        Returns:
            lambdas: possible thresholds to use to test the risk constraint, chosen randomly in a range between the minimum and the maximum values in conf_scores.
        """  # noqa: E501
        conf_scores = conf_scores.reshape(-1)
        conf_scores = conf_scores[torch.isfinite(conf_scores)]
        lambdas = torch.linspace(conf_scores.min(), conf_scores.max(), self.n_lambdas)
        return lambdas

    def get_bound(self, calib_loss_table):
        """
        Given the loss table for the calibration set, we compute the risk estimate for each lambda threshold

        Args:
            calib_loss_table (users x n_lambdas): The loss table for the calibration set, where each row is a user and each column is a lambda threshold.

        Returns:
            bound: The risk estimate for each lambda threshold.
        """  # noqa: E501
        n = calib_loss_table.shape[0]
        rhat = calib_loss_table.mean(dim=0)
        bound = (n / (n + 1)) * rhat + (self.maximum_metric_score / (n + 1))
        return bound

    def select_lhat(self, bound, lambdas):
        feasible_lhats = torch.where(bound <= self.alpha)[0]

        if len(feasible_lhats) == 0:
            # Nothing satisfies risk: use smallest lambda (largest set) as safest fallback.
            lhat_idx = 0

            self.logger.info(
                set_color(
                    f"No lambda satisfies the risk constraint; using smallest lambda "
                    f"{lambdas[lhat_idx]:.4f} with bound {bound[lhat_idx]:.4f}",
                    "red",
                )
            )
            return lambdas[lhat_idx], lhat_idx

        lhat_idx = feasible_lhats.max()

        return lambdas[lhat_idx], lhat_idx

    def forward(self, data, **kwargs):
        """
            Make a forward pass on the data using the trainer, and return the results, the losses and the confidence scores. If a threshold is provided, apply it to the confidence scores to get the calibrated results.

        Args:
            data: dataloader to evaluate on
            threshold (_type_, optional): Threshold to apply to the confidence scores to get the calibrated results. If None, return the uncalibrated results. Defaults to None.

        Returns:
            result: The metric results of the evaluation.
            losses: The losses for each user and each cut off k.
            confidence_scores: The confidence scores for each user and each cut off k.
        """  # noqa: E501
        # update self.evaluation config with kwargs
        for key, value in kwargs.items():
            self.evaluation_config[key] = value

        return self.trainer.evaluate(data, **self.evaluation_config)

    def get_loss_and_size_tables(self, conf_scores, lambdas):
        """
            Given the confidence scores, the losses and the lambda thresholds, compute the loss and size tables for each lambda threshold. The loss table contains the loss for each example and each lambda greater than the threshold, while the size table contains the size of the predicted set for each example and each lambda threshold.

            in case the higher the confidence score, the better, then you should have k = np.sum(conf_scores >= lmbda, axis=1). Otherwise, in case the higher the confidence score, the worse, then you should have k = np.sum(conf_scores >= 1 - lmbda, axis=1)

        Args:
            conf_scores (n_users x topk size)): it contains the full_sort_prediction confidence scores at cut off k
            losses (n_users x topk size)): it contains the metric score for each user and each cut off k
            lambdas : possible thresholds

        Returns:
            loss_table: (n_users x n_lambdas) metric value greater than the threshold for each user and each lambda
            size_table: (n_users x n_lambdas) size of the predicted set for each user and each lambda
        """  # noqa: E501

        if self.use_threads:

            def _compute_loss_and_size_for_lambda(conf_scores, lmbda):
                mask = conf_scores >= lmbda
                conf_scores_masked = torch.where(mask, conf_scores, float("-inf"))

                topk = torch.topk(conf_scores_masked, self.topk, dim=-1)
                size = torch.sum(torch.isfinite(topk.values), dim=1)

                self.metric.set_metric_data(conf_scores_masked, topk.indices)

                result = self.metric.calculate_metric()[:, -1]

                return result, size

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(_compute_loss_and_size_for_lambda, conf_scores, lmbda) for lmbda in lambdas]
                losses = []
                sizes = []
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    loss, size = fut.result()
                    losses.append(loss)
                    sizes.append(size)

            loss_table = np.column_stack(losses)
            size_table = np.column_stack(sizes)
        else:
            loss_table = torch.zeros((len(conf_scores), len(lambdas)))
            size_table = torch.zeros((len(conf_scores), len(lambdas)))
            for i, lmbda in enumerate(tqdm(lambdas)):
                # Lambda defines the set: keep only scores above threshold.
                mask = conf_scores >= lmbda
                conf_scores_masked = torch.where(mask, conf_scores, float("-inf"))

                # topk indices and values (descending)
                topk = torch.topk(conf_scores_masked, self.topk, dim=-1)

                self.metric.set_metric_data(conf_scores_masked, topk.indices)

                # Per-user metric curve over prefix sizes (k = 1..num_items).
                result = self.metric.calculate_metric()[:, -1]

                loss_table[:, i] = result
                size_table[:, i] = torch.sum(torch.isfinite(topk.values), dim=1)

        return loss_table, size_table

    def calibrate(self):
        """
        Given the losses and confidence scores for the calibration set, compute the lambda threshold lhat that satisfies the risk constraint, and return it along with the estimated risk and average size of the predicted sets at that threshold.

        Args:
            calib_losses (users x topk size): Metric scores at cut-off k
            confidence_scores (users x topk): full_sort_predict scores

        Returns:
            lhat: The lambda threshold that satisfies the risk constraint.
            risk: The estimated risk (average of the metric scores) at the chosen lambda threshold.
            avg_size: The average size of the topk after applying the lambda threshold lhat (i.e., the average number of items in the predicted set for each user after applying the threshold).
        """  # noqa: E501
        results, confidence_scores, positive_u_list, positive_i_list = self.forward(self.calib_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list)

        # get the lambda thresholds to test for the risk constraint
        lambdas = self.get_lambdas(confidence_scores)

        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas)

        # calculate CRC upper bound
        bound = self.get_bound(loss_table)

        # select lambda threshold lhat that satisfies the risk constraint at the alpha level
        lhat, lhat_idx = self.select_lhat(bound, lambdas)

        # return logs
        self._logging(lhat, lhat_idx, bound, loss_table, size_table, lambdas)

        calibrated_results, _, _, _ = self.forward(self.test_data, threshold=lhat)

        return results, calibrated_results

    def _logging(self, lhat, lhat_idx, bound, loss_table, size_table, lambdas):
        # get the risk and size at the chosen lambda threshold lhat
        risk = loss_table[:, lhat_idx].mean()
        avg_size = size_table[:, lhat_idx].mean()

        self.logger.info(set_color(f"\nRunning calibration on top-{self.topk} results.", "green"))
        self.logger.info(
            set_color(
                f"[CALIBRATION STEP] Calibrating results with conformal risk control (threshold={lhat:.4f}), risk {risk:.4f} with min bound {bound.min():.4f} and max bound {bound.max():.4f}, average topk size (inefficiency) {avg_size:.2f}",  # noqa: E501
                "green",
            )
        )
        self.logger.info(
            set_color(
                f"[CALIBRATION STEP]Risk at min lambda (default without crc) {lambdas[0]:.4f} is {loss_table[:, 0].mean():.4f} with bound {bound[0]:.4f} and average topk size {size_table[:, 0].mean():.2f}\n",  # noqa: E501
                "green",
            )
        )

    def prove_expectation(self):
        """
        Prove that the risk is controlled in expectation by running multiple trials of the calibration process. In each trial, we randomly split the full test set into a calibration user set and a test user set, compute the lambda threshold lhat on the calibration set, and then compute the risk and average size on the test set using that lhat. Finally, we average the risk and size across trials to get an estimate of the expected risk and size.
        """  # noqa: E501

        _, confidence_scores, positive_u_list, positive_i_list = self.forward(self.full_test_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list)

        # get the lambda thresholds to test for the risk constraint
        lambdas = self.get_lambdas(confidence_scores)

        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas)

        risks, lhats, sizes = list(), list(), list()
        for _ in tqdm(
            range(self.n_trials),
            desc=f"Computing conformal risk control in expectation using {self.crc_config['n_trials']} trials",
        ):
            lhat, risk, size = self.compute_trial(loss_table, size_table, lambdas)
            risks.append(risk)
            lhats.append(lhat)
            sizes.append(size)

        self.logger.info(set_color(f"Average risk across trials = {np.mean(risks):.3f}", "red"))
        self.logger.info(set_color(f"Average threshold across trials = {np.mean(lhats):.3f}", "red"))
        self.logger.info(set_color(f"Average size across trials = {np.mean(sizes):.3f}", "red"))

    def compute_trial(self, loss_table, size_table, lambdas):
        """Compute risk and sizes for a trial.

        Args:
            loss_table: [num_examples, num_lambdas] losses by lambda from small to large.
            size_table: [num_examples, num_lambdas] sizes by lambda from small to large.
            lambdas: [num_lambdas] lambda values from small to large.

        Returns:
            lhat: Confidence score threshold.
            avg_loss: Average set loss.
            avg_size: Average set size.
        """
        # Split to calibration and test.
        perm = np.random.permutation(len(loss_table))
        loss_table, size_table = loss_table[perm], size_table[perm]
        calib_loss_table = loss_table[: self.n_calibration_users]

        valid_loss_table = loss_table[self.n_calibration_users :]
        valid_size_table = size_table[self.n_calibration_users :]

        # Compute threshold
        bound = self.get_bound(calib_loss_table)
        lhat, lhat_idx = self.select_lhat(bound, lambdas)

        # Compute losses and size.
        avg_loss = valid_loss_table[:, lhat_idx].mean()
        avg_size = valid_size_table[:, lhat_idx].mean()

        return lhat, avg_loss, avg_size


class ConformalRiskControlTopMargin(ConformalRiskControl):
    """Calibration using Conformal Risk Control with top-relative margins.

    Instead of a global absolute threshold lambda on raw scores, we calibrate
    a user-relative margin delta. For each user u, we keep items i such that:

        score[u, i] >= max_score[u] - delta

    This is more appropriate for pairwise ranking models such as BPR, where
    score magnitudes are often meaningful mainly within each user.
    """

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        super().__init__(config, metric, trainer, logger, eval_config, calibration_data)

    def get_lambdas(self, conf_scores):
        """
        Return candidate deltas for top-relative thresholding.

        For each user, the effective threshold is:
            row_max(user) - delta

        Args:
            conf_scores (torch.Tensor): shape [n_users, n_items], raw full-sort scores.

        Returns:
            torch.Tensor: candidate delta values in [0, max_delta].
        """
        finite_mask = torch.isfinite(conf_scores)

        row_max = torch.where(finite_mask, conf_scores, float("-inf")).max(dim=1).values
        row_min = torch.where(finite_mask, conf_scores, float("inf")).min(dim=1).values

        valid_rows = torch.isfinite(row_max) & torch.isfinite(row_min)
        if not torch.any(valid_rows):
            raise ValueError("No finite confidence scores available to build calibration deltas.")

        max_delta = (row_max[valid_rows] - row_min[valid_rows]).max()

        lambdas = torch.linspace(
            0.0,
            max_delta,
            self.n_lambdas,
            device=conf_scores.device,
            dtype=conf_scores.dtype,
        )
        return lambdas

    def get_bound(self, calib_loss_table):
        """
        Compute CRC upper bound for each candidate delta.

        Args:
            calib_loss_table (torch.Tensor): shape [n_users, n_lambdas]

        Returns:
            torch.Tensor: CRC bound per delta.
        """
        n = calib_loss_table.shape[0]
        rhat = calib_loss_table.mean(dim=0)
        bound = (n / (n + 1)) * rhat + (self.maximum_metric_score / (n + 1))
        return bound

    def select_lhat(self, bound, lambdas):
        """
        Select the smallest feasible delta, since larger delta means larger sets
        and therefore usually lower FNR.

        If nothing satisfies the risk constraint, use the largest delta as
        safest fallback (largest set).
        """
        feasible_lhats = torch.where(bound <= self.alpha)[0]

        if len(feasible_lhats) == 0:
            lhat_idx = len(lambdas) - 1

            self.logger.info(
                set_color(
                    f"No delta satisfies the risk constraint; using largest delta "
                    f"{lambdas[lhat_idx]:.4f} with bound {bound[lhat_idx]:.4f}",
                    "red",
                )
            )
            return lambdas[lhat_idx], lhat_idx

        lhat_idx = feasible_lhats.min()
        return lambdas[lhat_idx], lhat_idx

    def forward(self, data, **kwargs):
        """
        Make a forward pass and return evaluation outputs.

        Args:
            data: dataloader to evaluate on
            **kwargs: extra evaluation settings

        Returns:
            Whatever self.trainer.evaluate returns.
        """
        for key, value in kwargs.items():
            self.evaluation_config[key] = value

        return self.trainer.evaluate(data, **self.evaluation_config)

    def get_loss_and_size_tables(self, conf_scores, lambdas):
        """
        For each candidate delta, build the prediction set using top-relative thresholding:

            keep item i for user u iff score[u, i] >= row_max[u] - delta

        Then compute:
        - loss_table[u, j]: user-level loss under delta_j
        - size_table[u, j]: resulting set size under delta_j

        Args:
            conf_scores (torch.Tensor): shape [n_users, n_items]
            lambdas (torch.Tensor): candidate delta values

        Returns:
            loss_table (torch.Tensor): shape [n_users, n_lambdas]
            size_table (torch.Tensor): shape [n_users, n_lambdas]
        """
        finite_scores = torch.isfinite(conf_scores)
        row_max = torch.where(finite_scores, conf_scores, float("-inf")).max(dim=1, keepdim=True).values

        if self.use_threads:

            def _compute_loss_and_size_for_lambda(conf_scores, row_max, lmbda):
                mask = conf_scores >= (row_max - lmbda)
                conf_scores_masked = torch.where(mask, conf_scores, float("-inf"))

                topk = torch.topk(conf_scores_masked, self.topk, dim=-1)
                size = torch.sum(torch.isfinite(topk.values), dim=1)

                self.metric.set_metric_data(conf_scores_masked, topk.indices)
                result = self.metric.calculate_metric()[:, -1]

                return result.detach().cpu().numpy(), size.detach().cpu().numpy()

            losses = []
            sizes = []

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(_compute_loss_and_size_for_lambda, conf_scores, row_max, lmbda)
                    for lmbda in lambdas
                ]

                for fut in tqdm(as_completed(futures), total=len(futures)):
                    loss, size = fut.result()
                    losses.append(loss)
                    sizes.append(size)

            # as_completed does not preserve order, so sort back by lambda index if needed.
            # Minimal fix: use executor.map instead for stable order.
            # To avoid changing behavior too much, we rebuild in-order below instead.
            # Simpler and safer: rerun with map-style ordering.
            losses = []
            sizes = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(
                    tqdm(
                        executor.map(
                            lambda lmbda: _compute_loss_and_size_for_lambda(conf_scores, row_max, lmbda), lambdas
                        ),
                        total=len(lambdas),
                    )
                )
            for loss, size in results:
                losses.append(loss)
                sizes.append(size)

            loss_table = torch.tensor(np.column_stack(losses), device=conf_scores.device, dtype=conf_scores.dtype)
            size_table = torch.tensor(np.column_stack(sizes), device=conf_scores.device, dtype=conf_scores.dtype)

        else:
            loss_table = torch.zeros(
                (len(conf_scores), len(lambdas)), device=conf_scores.device, dtype=conf_scores.dtype
            )
            size_table = torch.zeros(
                (len(conf_scores), len(lambdas)), device=conf_scores.device, dtype=conf_scores.dtype
            )

            for i, lmbda in enumerate(tqdm(lambdas)):
                mask = conf_scores >= (row_max - lmbda)
                conf_scores_masked = torch.where(mask, conf_scores, float("-inf"))

                topk = torch.topk(conf_scores_masked, self.topk, dim=-1)

                self.metric.set_metric_data(conf_scores_masked, topk.indices)
                result = self.metric.calculate_metric()[:, -1]

                loss_table[:, i] = result
                size_table[:, i] = torch.sum(torch.isfinite(topk.values), dim=1)

        return loss_table, size_table

    def calibrate(self):
        """
        Calibrate the top-relative delta using CRC.

        Returns:
            results: uncalibrated evaluation results on calibration pass
            calibrated_results: calibrated evaluation results on test data
        """
        results, confidence_scores, positive_u_list, positive_i_list = self.forward(self.calib_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list)

        # Candidate deltas
        lambdas = self.get_lambdas(confidence_scores)

        # Per-delta loss and set-size tables
        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas)

        # CRC bound
        bound = self.get_bound(loss_table)

        # Select calibrated delta
        lhat, lhat_idx = self.select_lhat(bound, lambdas)

        self._logging(lhat, lhat_idx, bound, loss_table, size_table, lambdas)

        calibrated_results, _, _, _ = self.forward(self.test_data, threshold=lhat)

        return results, calibrated_results


class LearnThenTest(AbstractCalibration):
    """Learn-then-test is used to calibrate the model using a simple learn-then-test approach"""

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        super().__init__(config, metric, trainer, logger, eval_config, calibration_data)


class ConsumerFairnessConformalRiskControl(ConformalRiskControl):
    """Calibration is used to calibrate the model using conformal risk control."""

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        super().__init__(config, metric, trainer, logger, eval_config, calibration_data)

    def _logging(self, lhat, lhat_idx, bound, loss_table, size_table, lambdas, **kwargs):
        risk = loss_table[:, lhat_idx].mean()
        avg_size = size_table[:, lhat_idx].mean()

        mask = kwargs["mask"]

        males = mask
        females = ~mask

        self.logger.info(set_color(f"Running calibration on top-{self.topk} results.", "green"))

        self.logger.info(
            set_color(
                f"Calibrating results with conformal risk control (threshold={lhat:.4f}), risk {risk:.4f} with min bound {bound.min():.4f} and max bound {bound.max():.4f}, average topk size {avg_size:.2f}",  # noqa: E501
                "green",
            )
        )

        self.logger.info(
            set_color(
                f"Risk at min lambda (default without crc) {lambdas[0]:.4f} is {loss_table[:, 0].mean():.4f} with bound {bound[0]:.4f} and average topk size {size_table[:, 0].mean():.2f}",  # noqa: E501
                "yellow",
            )
        )

        self.logger.info(
            set_color(
                f"Males min topk size {size_table[males, lhat_idx].min()} max topk size {size_table[males, lhat_idx].max()}, mean topk size {size_table[males, lhat_idx].mean():.2f}",  # noqa: E501
                "green",
            )
        )

        self.logger.info(
            set_color(
                f"Females users min topk size {size_table[females, lhat_idx].min()} max topk size {size_table[females, lhat_idx].max()}, mean topk size {size_table[females, lhat_idx].mean():.2f}",  # noqa: E501
                "yellow",
            )
        )

    def calibrate(self):
        """
        Given the losses and confidence scores for the calibration set, compute the lambda threshold lhat that satisfies the risk constraint, and return it along with the estimated risk and average size of the predicted sets at that threshold.

        Args:
            calib_losses (users x topk size): Metric scores at cut-off k
            confidence_scores (users x topk): full_sort_predict scores

        Returns:
            lhat: The lambda threshold that satisfies the risk constraint.
            risk: The estimated risk (average of the metric scores) at the chosen lambda threshold.
            avg_size: The average size of the topk after applying the lambda threshold lhat (i.e., the average number of items in the predicted set for each user after applying the threshold).
        """  # noqa: E501
        results, confidence_scores, positive_u_list, positive_i_list = self.forward(self.calib_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list)

        # get the lambda thresholds to test for the risk constraint
        lambdas = self.get_lambdas(confidence_scores)

        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas)

        # calculate CRC upper bound
        bound = self.get_bound(loss_table)

        # select lambda threshold lhat that satisfies the risk constraint at the alpha level
        lhat, lhat_idx = self.select_lhat(bound, lambdas)

        mask = self._get_mask()
        self._logging(lhat, lhat_idx, bound, loss_table, size_table, lambdas, mask=mask)

        calibrated_results, _, _, _ = self.forward(self.test_data, threshold=lhat)

        return results, calibrated_results

    def _get_mask(self):
        import pandas as pd

        user_feat = self.train_data.dataset.user_feat.cpu().numpy()
        id2gender = {id: gender for gender, id in self.train_data.dataset.field2token_id["gender"].items()}

        df = pd.DataFrame(user_feat)
        df.gender = df.gender.apply(lambda x: id2gender.get(x))

        # return mask for males, skip pad
        return np.array(df.gender == "M")[1:]


class UserColdStartConformalRiskControl(ConformalRiskControl):
    """UserColdStartConformalRiskControl is used to calibrate the model using conformal risk control, but only on cold-start users (i.e., users with 3 interactions (2 in case of sequential) in the training set)."""  # noqa: E501

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        super().__init__(config, metric, trainer, logger, eval_config, calibration_data)

        self.cold_start_n_inters = self.crc_config["cold_start_n_inters"]

    def get_lambdas(self, conf_scores, mask=None):
        """
        Return the thresholds to use to test the risk constraint. Instead of using all the possible predicted score by the full_sort_prediction function, we use a grid of lambda values choose randomly in a range between the minimum and the maximum values in conf_scores. # noqa: E501

        Args:
            conf_scores (n_users x topk size)): full_sort_prediction confidence scores at cut off k

        Returns:
            lambdas: possible thresholds to use to test the risk constraint, chosen randomly in a range between the minimum and the maximum values in conf_scores.
        """  # noqa: E501
        if mask is not None:
            conf_scores = conf_scores[mask]

        conf_scores = conf_scores.reshape(-1)
        conf_scores = conf_scores[torch.isfinite(conf_scores)]
        lambdas = torch.linspace(conf_scores.min(), conf_scores.max(), self.n_lambdas)
        return lambdas

    def get_bound(self, calib_loss_table, mask=None):
        """
        Given the loss table for the calibration set, we compute the risk estimate for each lambda threshold

        Args:
            calib_loss_table (users x n_lambdas): The loss table for the calibration set, where each row is a user and each column is a lambda threshold.

        Returns:
            bound: The risk estimate for each lambda threshold.
        """  # noqa: E501
        n = calib_loss_table.shape[0]

        if mask is not None:
            calib_loss_table = calib_loss_table[mask]

        rhat = calib_loss_table.mean(dim=0)
        bound = (n / (n + 1)) * rhat + (self.maximum_metric_score / (n + 1))
        return bound

    def select_lhat(self, bound, lambdas):
        """Select the smallest feasible lambda (largest prediction set)."""
        # I want the bound to be between (alpha - delta) and (alpha + delta). So if I select an alpha of 0.5, I want the bound to be between 0.45 and 0.55. This means that I want to select the lambda such that the bound is less than or equal to 0.45 and less than or equal to 0.55. In other words, I want to select the lambda such that the bound is less than or equal to (alpha - delta) and less than or equal to (alpha + delta). # noqa: E501
        # feasible_lhats = np.where((bound >= (self.alpha-self.delta)) & (bound <= (self.alpha+self.delta)))[0]

        feasible_lhats = torch.where(bound <= torch.maximum(torch.ceil(bound.min() * 10) / 10, self.alpha))[0]

        # feasible_lhats = torch.where(bound <= self.alpha)[0]

        if len(feasible_lhats) == 0:
            # Nothing satisfies risk: use smallest lambda (largest set) as safest fallback.
            lhat_idx = -1
            self.logger.info(
                set_color(
                    f"No lambda satisfies the risk constraint; using smallest lambda "
                    f"{lambdas[lhat_idx]:.4f} with bound {bound[lhat_idx]:.4f}",
                    "red",
                )
            )
            return lambdas[lhat_idx], lhat_idx

        lhat_idx = feasible_lhats.max()

        return lambdas[lhat_idx], lhat_idx

    def get_loss_and_size_tables(self, conf_scores, lambdas, mask=None):
        """
            Given the confidence scores, the losses and the lambda thresholds, compute the loss and size tables for each lambda threshold. The loss table contains the loss for each example and each lambda greater than the threshold, while the size table contains the size of the predicted set for each example and each lambda threshold. # noqa: E501

            in case the higher the confidence score, the better, then you should have k = np.sum(conf_scores >= lmbda, axis=1). Otherwise, in case the higher the confidence score, the worse, then you should have k = np.sum(conf_scores >= 1 - lmbda, axis=1) # noqa: E501

        Args:
            conf_scores (n_users x topk size)): it contains the full_sort_prediction confidence scores at cut off k
            losses (n_users x topk size)): it contains the metric score for each user and each cut off k
            lambdas : possible thresholds

        Returns:
            loss_table: (n_users x n_lambdas) metric value greater than the threshold for each user and each lambda
            size_table: (n_users x n_lambdas) size of the predicted set for each user and each lambda
        """  # noqa: E501

        loss_table = torch.zeros((len(conf_scores), len(lambdas)))
        size_table = torch.zeros((len(conf_scores), len(lambdas)))
        for i, lmbda in enumerate(tqdm(lambdas)):
            # Lambda defines the set: keep only scores above threshold.
            mask = conf_scores >= lmbda
            conf_scores_masked = torch.where(mask, conf_scores, float("-inf"))

            if mask is not None:
                conf_scores_masked[~mask] = conf_scores[~mask]

            # topk indices and values (descending)
            topk = torch.topk(conf_scores_masked, self.topk, dim=-1)

            self.metric.set_metric_data(conf_scores_masked, topk.indices)

            # Per-user metric curve over prefix sizes (k = 1..num_items).
            result = self.metric.calculate_metric()[:, -1]

            loss_table[:, i] = result
            size_table[:, i] = torch.sum(torch.isfinite(topk.values), dim=1)

        return loss_table, size_table

    def calibrate(self):
        """
        Given the losses and confidence scores for the calibration set, compute the lambda threshold lhat that satisfies the risk constraint, and return it along with the estimated risk and average size of the predicted sets at that threshold. # noqa: E501

        Args:
            calib_losses (users x topk size): Metric scores at cut-off k
            confidence_scores (users x topk): full_sort_predict scores

        Returns:
            lhat: The lambda threshold that satisfies the risk constraint.
            risk: The estimated risk (average of the metric scores) at the chosen lambda threshold.
            avg_size: The average size of the topk after applying the lambda threshold lhat (i.e., the average number of items in the predicted set for each user after applying the threshold).
        """  # noqa: E501
        results, confidence_scores, positive_u_list, positive_i_list = self.forward(self.calib_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list)

        mask = self._get_mask()

        # get the lambda thresholds to test for the risk constraint
        lambdas = self.get_lambdas(confidence_scores, mask=mask)

        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas, mask=mask)

        # calculate CRC upper bound
        bound = self.get_bound(loss_table, mask=mask)

        # select lambda threshold lhat that satisfies the risk constraint at the alpha level
        lhat, lhat_idx = self.select_lhat(bound, lambdas)

        self._logging(lhat, lhat_idx, bound, loss_table, size_table, lambdas, mask=mask)

        calibrated_results, _, _, _ = self.forward(self.test_data, threshold=lhat)

        return results, calibrated_results

    def _logging(self, lhat, lhat_idx, bound, loss_table, size_table, lambdas, **kwargs):
        risk = loss_table[:, lhat_idx].mean()
        avg_size = size_table[:, lhat_idx].mean()

        cold_start_users_mask = kwargs["cold_start_users_mask"]

        self.logger.info(set_color(f"Running calibration on top-{self.topk} results.", "green"))

        self.logger.info(
            set_color(
                f"Calibrating results with conformal risk control (threshold={lhat:.4f}), risk {risk:.4f} with min bound {bound.min():.4f} and max bound {bound.max():.4f}, average topk size {avg_size:.2f}",  # noqa: E501
                "green",
            )
        )

        self.logger.info(
            set_color(
                f"Risk at min lambda (default without crc) {lambdas[0]:.4f} is {loss_table[:, 0].mean():.4f} with bound {bound[0]:.4f} and average topk size {size_table[:, 0].mean():.2f}",  # noqa: E501
                "yellow",
            )
        )

        self.logger.info(
            set_color(
                f"Cold start users min topk size {size_table[cold_start_users_mask, lhat_idx].min()} max topk size {size_table[cold_start_users_mask, lhat_idx].max()}, mean topk size {size_table[cold_start_users_mask, lhat_idx].mean():.2f}",  # noqa: E501
                "green",
            )
        )

        self.logger.info(
            set_color(
                f"NO Cold start users min topk size {size_table[~cold_start_users_mask, lhat_idx].min()} max topk size {size_table[~cold_start_users_mask, lhat_idx].max()}, mean topk size {size_table[~cold_start_users_mask, lhat_idx].mean():.2f}",  # noqa: E501
                "yellow",
            )
        )

    def _get_mask(self):
        """
        Get a boolean mask indicating which users in the calibration set are cold-start users (i.e., users with 3 interactions in the training set).

        Returns:
            cold_start_users_mask: A boolean array of shape (n_calibration_users,) where True indicates a cold-start user and False indicates a non-cold-start user.
        """  # noqa: E501
        import pandas as pd

        from hopwise.utils.enum_type import ModelType

        train_df = pd.DataFrame(
            {
                "user_id": self.train_data.dataset.inter_feat["user_id"].cpu().numpy(),
                "item_id": self.train_data.dataset.inter_feat["item_id"].cpu().numpy(),
            }
        )
        train_user_counts = train_df.groupby("user_id").size()
        n_users = self.calib_data.dataset.user_num
        user_ids = np.arange(n_users)
        train_user_interactions = train_user_counts.reindex(user_ids, fill_value=0)

        if self.config["MODEL_TYPE"] == ModelType.SEQUENTIAL:
            cold_start_users_mask = train_user_interactions == self.cold_start_n_inters - 1
        else:
            cold_start_users_mask = train_user_interactions == self.cold_start_n_inters

        return cold_start_users_mask[1:]
