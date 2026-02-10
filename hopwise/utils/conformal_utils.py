import numpy as np
from statsmodels.stats.multitest import multipletests


def get_tokenized_paths_dict(tokenized_dataset):
    user_paths = {}
    for path in tokenized_dataset.input_ids:
        user_prefix = tuple(path[:3].tolist())

        if user_prefix not in user_paths:
            user_paths[user_prefix] = list()

        user_paths[user_prefix].append(list(path[3:].tolist()))

    return user_paths


def calibrate_predictions(losses, crc_config):
    alphas = [crc_config[f"alpha{i+1}"] for i in range(5)]
    n = losses.shape[1]
    risks = losses.mean(1)

    valid_lambdas = bonferroni_HB(risks, alphas, n, crc_config["delta"])

    return valid_lambdas


def hb_p_value(r_hat, n, alpha):
    from scipy.stats import binom

    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)

    def h1(y, mu):
        with np.errstate(divide="ignore"):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))

    hoeffding_p_value = np.exp(-n * h1(min(r_hat, alpha), alpha))

    return min(bentkus_p_value, hoeffding_p_value)


def bonferroni(p_values, delta):
    seq_rejections = []
    for pv in p_values:
        rejections, _, _, _ = multipletests(pv, delta, method="holm", is_sorted=False, returnsorted=False)
        seq_rejections.append(np.nonzero(rejections)[0])

    return [v[0] for v in seq_rejections]


def bonferroni_HB(loss_table, alphas, n, delta):
    seq_length = loss_table.shape[0]
    n_lambdas = loss_table.shape[1]
    p_values = []
    for i in range(seq_length):
        step_p_values = []
        for j in range(n_lambdas):
            step_p_values.append(hb_p_value(loss_table[i][j].cpu().numpy(), n, alphas[i]))

        p_values.append(step_p_values)

    return bonferroni(np.array(p_values), delta)
