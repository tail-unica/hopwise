# @Time   : 2020/7/19 19:06
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : hyper_tuning.py

# UPDATE:
# @Time   : 2022/7/7, 2023/2/11
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com

# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

"""hopwise.trainer.hyper_tuning
############################
"""

import os
from ast import literal_eval
from datetime import datetime
from enum import Enum
from functools import partial

import numpy as np

from hopwise.utils import dict2str


def _recursiveFindNodes(root, node_type="switch"):
    from hyperopt.pyll.base import Apply

    nodes = []
    if isinstance(root, (list, tuple)):
        for node in root:
            nodes.extend(_recursiveFindNodes(node, node_type))
    elif isinstance(root, dict):
        for node in root.values():
            nodes.extend(_recursiveFindNodes(node, node_type))
    elif isinstance(root, (Apply)):
        if root.name == node_type:
            nodes.append(root)

        for node in root.pos_args:
            if node.name == node_type:
                nodes.append(node)
        for _, node in root.named_args:
            if node.name == node_type:
                nodes.append(node)
    return nodes


def _parameters(space):
    # Analyze the domain instance to find parameters
    parameters = {}
    if isinstance(space, dict):
        space = list(space.values())
    for node in _recursiveFindNodes(space, "switch"):
        # Find the name of this parameter
        paramNode = node.pos_args[0]
        assert paramNode.name == "hyperopt_param"
        paramName = paramNode.pos_args[0].obj

        # Find all possible choices for this parameter
        values = [literal.obj for literal in node.pos_args[1:]]
        parameters[paramName] = np.array(range(len(values)))
    return parameters


def _spacesize(space):
    # Compute the number of possible combinations
    params = _parameters(space)
    return np.prod([len(values) for values in params.values()])


class ExhaustiveSearchError(Exception):
    r"""ExhaustiveSearchError"""

    pass


def exhaustive_search(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    r"""This is for exhaustive search in HyperTuning."""
    from hyperopt import pyll
    from hyperopt.base import miscs_update_idxs_vals

    # Build a hash set for previous trials
    hashset = set(
        [
            hash(
                frozenset(
                    [
                        (key, value[0]) if len(value) > 0 else ((key, None))
                        for key, value in trial["misc"]["vals"].items()
                    ]
                )
            )
            for trial in trials.trials
        ]
    )

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                },
            )
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval


class HyperTuning:
    r"""HyperTuning Class is used to manage the parameter tuning process of recommender system models.
    Given objective funciton, parameters range and optimization algorithm, using HyperTuning can find
    the best result among these parameters.

    Note:
        HyperTuning provides three tuner tools:
            - hyperopt (https://github.com/hyperopt/hyperopt)
            - ray (https://docs.ray.io/en/latest/tune/index.html)
            - optuna (https://optuna.org/)

        Thanks to sbrodeur for the exhaustive search code.
        https://github.com/hyperopt/hyperopt/issues/200
    """

    PARAMS_PER_ROW = 3
    TUNER_TYPES = Enum("TUNER_TYPES", {"HYPEROPT": "hyperopt", "RAY": "ray", "OPTUNA": "optuna"})

    def __init__(
        self,
        objective_function,
        tuner="optuna",
        space=None,
        params_file=None,
        params_dict=None,
        fixed_config_file_list=None,
        display_file=None,
        algo=None,
        max_evals=100,
        early_stop=10,
        output_path=None,
        timeout=None,
        show_progress=False,
        study_name=None,
        resume=False,
    ):
        self.tuner = self.TUNER_TYPES[tuner.upper()]
        self.best_score = None
        self.best_params = None
        self.best_test_result = None
        self.params2result = {}
        self.params_list = []
        self.score_list = []

        self.show_progress = show_progress
        self.objective_function = objective_function
        self.max_evals = max_evals
        self.timeout = timeout
        self.fixed_config_file_list = fixed_config_file_list
        self.display_file = display_file
        self.output_path = output_path or "."
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.study_name = study_name or f"hyper_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
        self.resume = resume

        if space:
            self.space = space
        elif params_file:
            self.space = self.build_space_from_file(params_file)
        elif params_dict:
            self.space = self.build_space_from_dict(params_dict)
        else:
            raise ValueError("at least one of `space`, `params_file` and `params_dict` is provided")

        self.select_algo(algo)
        self.select_early_stop(early_stop)

    def select_algo(self, algo):
        r"""Select the algorithm for hyperparameter tuning
        Args:
            algo (str or callable): the algorithm name or function
        """
        if self.tuner == self.TUNER_TYPES.HYPEROPT:
            if algo is None:
                self.algo = partial(exhaustive_search, nbMaxSucessiveFailures=1000)
                self.max_evals = _spacesize(self.space)
            elif isinstance(algo, str):
                from hyperopt import anneal, rand, tpe

                if algo == "exhaustive":
                    self.algo = partial(exhaustive_search, nbMaxSucessiveFailures=1000)
                    self.max_evals = _spacesize(self.space)
                elif algo == "random":
                    self.algo = rand.suggest
                elif algo == "bayes":
                    self.algo = tpe.suggest
                elif algo == "anneal":
                    self.algo = anneal.suggest
                else:
                    raise ValueError(f"Illegal algo [{algo}]")
            else:
                self.algo = algo
        elif self.tuner == self.TUNER_TYPES.RAY:
            from ray.tune import schedulers, search

            if algo is None:
                self.algo = {"search_alg": None, "scheduler": "async_hyperband"}
            elif isinstance(algo, str):
                if "-" in algo:
                    search_alg, scheduler = algo.split("-")
                    search_alg = search.SEARCH_ALG_IMPORT.get(search_alg, None)
                    self.algo = {"search_alg": search_alg, "scheduler": scheduler}
                else:
                    search_alg = search.SEARCH_ALG_IMPORT.get(algo, None)
                    scheduler = algo if algo in schedulers.SCHEDULER_IMPORT else None
                    self.algo = {
                        "search_alg": search_alg,
                        "scheduler": scheduler,
                    }
        elif self.tuner == self.TUNER_TYPES.OPTUNA:
            import optuna

            if algo is None:
                self.algo = {
                    "sampler": None,
                    "pruner": optuna.pruners.MedianPruner(),
                }
            elif isinstance(algo, str):
                grid_space = {k: (v[2] if v[0] == "choice" else list(v[2:])) for k, v in self.space.items()}

                if "-" in algo:
                    sampler, pruner = algo.split("-")
                    sampler_args = [grid_space] if sampler == "GridSampler" else []
                    self.algo = {
                        "sampler": getattr(optuna.samplers, sampler)(*sampler_args),
                        "pruner": getattr(optuna.pruners, pruner)(),
                    }
                else:
                    sampler = getattr(optuna.samplers, algo) if hasattr(optuna.samplers, algo) else None
                    pruner = getattr(optuna.pruners, algo) if hasattr(optuna.pruners, algo) else None
                    if sampler is not None and sampler is optuna.samplers.GridSampler:
                        sampler_args = [grid_space]
                    else:
                        sampler_args = []
                    self.algo = {
                        "sampler": sampler(*sampler_args) if sampler is not None else None,
                        "pruner": pruner() if pruner is not None else None,
                    }

    def select_early_stop(self, early_stop_steps):
        from hyperopt.early_stop import no_progress_loss

        self.early_stop_fn = no_progress_loss(early_stop_steps)

    def _get_tuner_distributions(self):
        if self.tuner == self.TUNER_TYPES.HYPEROPT:
            from hyperopt import hp

            def choice(name, values):
                return hp.choice(name, values)

            def uniform(name, low, high):
                return hp.uniform(name, low, high)

            def quniform(name, low, high, q):
                return hp.quniform(name, low, high, q)

            def loguniform(name, low, high):
                return hp.loguniform(name, low, high)
        elif self.tuner == self.TUNER_TYPES.RAY:
            from ray import tune

            def choice(name, values):
                return tune.choice(values)

            def uniform(name, low, high):
                return tune.uniform(low, high)

            def quniform(name, low, high, q):
                return tune.quniform(low, high, q)

            def loguniform(name, low, high):
                return tune.uniform(np.exp(low), np.exp(high))
        elif self.tuner == self.TUNER_TYPES.OPTUNA:

            def choice(name, values):
                return "choice", name, values

            def uniform(name, low, high):
                return "uniform", name, low, high

            def quniform(name, low, high, q):
                return "quniform", name, low, high, q

            def loguniform(name, low, high):
                return "loguniform", name, low, high

        return choice, uniform, quniform, loguniform

    def build_space_from_file(self, file):
        choice, uniform, quniform, loguniform = self._get_tuner_distributions()

        space = self._build_space_from_file(
            file,
            choice_fn=choice,
            uniform_fn=uniform,
            quniform_fn=quniform,
            loguniform_fn=loguniform,
        )

        return space

    def build_space_from_dict(self, config_dict):
        choice, uniform, quniform, loguniform = self._get_tuner_distributions()

        space = self._build_space_from_dict(
            config_dict,
            choice_fn=choice,
            uniform_fn=uniform,
            quniform_fn=quniform,
            loguniform_fn=loguniform,
        )

        return space

    @staticmethod
    def _build_space_from_file(
        file,
        choice_fn=None,
        uniform_fn=None,
        quniform_fn=None,
        loguniform_fn=None,
    ):
        config_dict = {}
        with open(file) as fp:
            for line in fp:
                para_list = line.strip().split(" ")
                if len(para_list) < HyperTuning.PARAMS_PER_ROW:
                    continue
                para_name, para_type, para_value = (
                    para_list[0],
                    para_list[1],
                    "".join(para_list[2:]),
                )
                if para_type == "choice":
                    config_dict.setdefault("choice", {})
                    config_dict["choice"][para_name] = literal_eval(para_value)
                elif para_type == "uniform":
                    config_dict.setdefault("uniform", {})
                    low, high = para_value.strip().split(",")
                    config_dict["uniform"][para_name] = (float(low), float(high))
                elif para_type == "quniform":
                    config_dict.setdefault("quniform", {})
                    low, high, q = para_value.strip().split(",")
                    config_dict["quniform"][para_name] = (float(low), float(high), float(q))
                elif para_type == "loguniform":
                    config_dict.setdefault("loguniform", {})
                    low, high = para_value.strip().split(",")
                    config_dict["loguniform"][para_name] = (float(low), float(high))
                else:
                    raise ValueError(f"Illegal param type [{para_type}]")

        space = HyperTuning._build_space_from_dict(
            config_dict,
            choice_fn=choice_fn,
            uniform_fn=uniform_fn,
            quniform_fn=quniform_fn,
            loguniform_fn=loguniform_fn,
        )
        return space

    @staticmethod
    def _build_space_from_dict(
        config_dict,
        choice_fn=None,
        uniform_fn=None,
        quniform_fn=None,
        loguniform_fn=None,
    ):
        space = {}
        for para_type in config_dict:
            if para_type == "choice":
                for para_name in config_dict["choice"]:
                    para_value = config_dict["choice"][para_name]
                    space[para_name] = choice_fn(para_name, para_value)
            elif para_type == "uniform":
                for para_name in config_dict["uniform"]:
                    para_value = config_dict["uniform"][para_name]
                    low = para_value[0]
                    high = para_value[1]
                    space[para_name] = uniform_fn(para_name, float(low), float(high))
            elif para_type == "quniform":
                for para_name in config_dict["quniform"]:
                    para_value = config_dict["quniform"][para_name]
                    low = para_value[0]
                    high = para_value[1]
                    q = para_value[2]
                    space[para_name] = quniform_fn(para_name, float(low), float(high), float(q))
            elif para_type == "loguniform":
                for para_name in config_dict["loguniform"]:
                    para_value = config_dict["loguniform"][para_name]
                    low = para_value[0]
                    high = para_value[1]
                    space[para_name] = loguniform_fn(para_name, float(low), float(high))
            else:
                raise ValueError(f"Illegal param type [{para_type}]")
        return space

    def build_optuna_space(self, trial):
        r"""Build the space for optuna

        Args:
            trial (optuna.trial): the trial object
        """
        params = {}
        for para_name in self.space:
            para_type, _, *para_value = self.space[para_name]
            if para_type == "choice":
                para_value = para_value[0]
                params[para_name] = trial.suggest_categorical(para_name, para_value)
            elif para_type == "uniform":
                low = para_value[0]
                high = para_value[1]
                params[para_name] = trial.suggest_float(para_name, low, high)
            elif para_type == "quniform":
                low = para_value[0]
                high = para_value[1]
                q = para_value[2]
                params[para_name] = trial.suggest_float(para_name, low, high, step=q)
            elif para_type == "loguniform":
                low = para_value[0]
                high = para_value[1]

                params[para_name] = np.exp(trial.suggest_float(para_name, low, high))
            else:
                raise ValueError(f"  Illegal param type [{para_type}]")

        return params

    @staticmethod
    def params2str(params):
        r"""Convert dict to str

        Args:
            params (dict): parameters dict
        Returns:
            str: parameters string
        """
        params_str = ""
        for param_name in params:
            params_str += param_name + ":" + str(params[param_name]) + ", "
        return params_str[:-2]

    @staticmethod
    def _print_result(result_dict: dict):
        print("current best valid score: %.4f" % result_dict["best_valid_score"])
        print("current best valid result:")
        print(result_dict["best_valid_result"])
        print("current test result:")
        print(result_dict["test_result"])
        print()

    def export_result(self, output_path=None):
        r"""Write the searched parameters and corresponding results to the file

        Args:
            output_path (str): the output file

        """
        output_path = output_path or self.output_path
        output_file = os.path.join(output_path, self.study_name + ".txt")

        with open(output_file, "w") as fp:
            fp.write("***Best trial***\n")
            fp.write("Best valid score: %.4f\n" % self.best_score)
            fp.write("Best parameters: " + dict2str(self.best_params) + "\n")
            fp.write("Best valid result:\n" + dict2str(self.best_valid_result) + "\n")
            fp.write("Best test result:\n" + dict2str(self.best_test_result) + "\n\n")

            for params in self.params2result:
                fp.write(params + "\n")
                fp.write("Valid result:\n" + dict2str(self.params2result[params]["best_valid_result"]) + "\n")

                fp.write("Test result:\n" + dict2str(self.params2result[params]["test_result"]) + "\n\n")

            if self.tuner == self.TUNER_TYPES.OPTUNA:
                if not hasattr(self, "study"):
                    raise ValueError("Optuna study not created. Call `run` method first.")

                optuna_df = self.study.trials_dataframe()
                fp.write("Optuna trials dataframe:\n")
                optuna_df.to_string(fp, index=False)

    def trial(self, params):
        r"""Given a set of parameters, return results and optimization status

        Args:
            params (dict): the parameter dictionary
        """
        config_dict = params.copy()
        params_str = self.params2str(params)
        self.params_list.append(params_str)
        print("running parameters:", config_dict)
        result_dict = self.objective_function(config_dict, self.fixed_config_file_list, saved=False)
        self.params2result[params_str] = result_dict
        model, score, bigger = (
            result_dict["model"],
            result_dict["best_valid_score"],
            result_dict["valid_score_bigger"],
        )
        self.model = model
        self.score_list.append(score)

        if not self.best_score or (bigger and score > self.best_score) or (not bigger and score < self.best_score):
            self.best_score = score
            self.best_params = params
            self.best_valid_result = result_dict["best_valid_result"]
            self.best_test_result = result_dict["test_result"]
            self._print_result(result_dict)

        if bigger:
            score = -score

        return {**result_dict, "hyper_score": score}

    def plot_hyper(self):
        import pandas as pd
        import plotly.graph_objs as go
        from plotly.offline import plot

        data_dict = {"valid_score": self.score_list, "params": self.params_list}
        trial_df = pd.DataFrame(data_dict)
        trial_df["trial_number"] = trial_df.index + 1
        trial_df["trial_number"] = trial_df["trial_number"].astype(dtype=np.str)

        trace = go.Scatter(
            x=trial_df["trial_number"],
            y=trial_df["valid_score"],
            text=trial_df["params"],
            mode="lines+markers",
            marker=dict(color="green"),
            showlegend=True,
            textposition="top center",
            name=self.model + " tuning process",
        )

        data = [trace]
        layout = go.Layout(
            title="hyperparams_tuning",
            xaxis=dict(title="trials"),
            yaxis=dict(title="valid_score"),
        )
        fig = go.Figure(data=data, layout=layout)

        plot(fig, filename=self.display_file)

    def run(self):
        r"""Begin to search the best parameters"""
        if self.resume:
            print("\n# Resume " + "-" * 40)
            print(f"Resuming from {os.path.join(self.output_path, self.study_name)} if exists.\n")

        if self.tuner == self.TUNER_TYPES.HYPEROPT:
            import hyperopt

            def hyperopt_objective(params):
                try:
                    result_dict = self.trial(params)
                except Exception as e:
                    print(f"Error occurred during trial: {e}")
                    import traceback

                    traceback.print_exc()
                    return {"loss": np.nan, "status": hyperopt.STATUS_FAIL}

                return {"loss": result_dict["hyper_score"], "status": hyperopt.STATUS_OK}

            if os.path.exists(os.path.join(self.output_path, self.study_name)) and not self.resume:
                raise FileExistsError(
                    f"File {os.path.join(self.output_path, self.study_name)} already exists. "
                    "Please remove it or set `--resume True` to continue."
                )

            hyperopt.fmin(
                hyperopt_objective,
                self.space,
                algo=self.algo,
                max_evals=self.max_evals,
                early_stop_fn=self.early_stop_fn,
                timeout=self.timeout,
                trials_save_file=os.path.join(self.output_path, self.study_name),
            )
        elif self.tuner == self.TUNER_TYPES.RAY:
            import ray
            from ray import tune

            def ray_objective(params):
                result_dict = self.trial(params)
                ray.train.report({"hyper_score": result_dict["hyper_score"]})

                return result_dict

            if not ray.is_initialized():
                ray.init()
            tune.register_trainable("ray-trial", ray_objective)
            if self.algo["scheduler"] is not None:
                scheduler = tune.create_scheduler(
                    self.algo["scheduler"],
                    metric="hyper_score",
                    mode="min",
                    max_t=self.max_evals,
                    grace_period=1,
                    reduction_factor=2,
                )
            else:
                scheduler = None

            if self.algo["search_alg"] is not None:
                from ray.tune import search

                if self.algo["search_alg"]() is search.BasicVariantGenerator:
                    search_alg = search.BasicVariantGenerator()
                else:
                    search_alg = self.algo["search_alg"]()(
                        metric="hyper_score",
                        mode="min",
                    )
            else:
                search_alg = None

            tune.run(
                ray_objective,
                config=self.space,
                num_samples=self.max_evals,
                scheduler=scheduler,
                search_alg=search_alg,
                storage_path=self.output_path,
                name=self.study_name,
                log_to_file=self.study_name,
                resume=self.resume,
            )
        elif self.tuner == self.TUNER_TYPES.OPTUNA:
            import optuna

            try:
                self.study = optuna.create_study(
                    direction="minimize",
                    study_name=self.study_name,
                    storage=f"sqlite:///{os.path.join(self.output_path, self.study_name)}.db",
                    pruner=self.algo["pruner"],
                    sampler=self.algo["sampler"],
                    load_if_exists=self.resume,
                )
            except optuna.exceptions.DuplicatedStudyError as e:
                raise optuna.exceptions.DuplicatedStudyError(
                    f"Study {os.path.join(self.output_path, self.study_name)} already exists. "
                    "Please use --resume True to load an existing study checkpoint."
                ) from e

            def optuna_objective(trial):
                params = self.build_optuna_space(trial)

                def trial_callback(epoch_idx, valid_score):
                    trial.report(valid_score, epoch_idx)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if isinstance(self.objective_function, partial):
                    self.objective_function = partial(
                        self.objective_function.func,
                        callback_fn=trial_callback,
                    )
                else:
                    self.objective_function = partial(
                        self.objective_function,
                        callback_fn=trial_callback,
                    )

                result_dict = self.trial(params)

                for key, value in result_dict["test_result"].items():
                    trial.set_user_attr(key, value)

                return result_dict["hyper_score"]

            self.study.optimize(
                optuna_objective,
                n_trials=self.max_evals,
                timeout=self.timeout,
            )

        if self.display_file is not None:
            self.plot_hyper()
