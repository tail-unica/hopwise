# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29, 2022/7/13, 2022/7/18
# @Author : Zihan Lin, Yupeng Hou, Gaowei Zhang, Lei Wang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, zgw15630559577@163.com, zxcptss@gmail.com

import argparse
import math
import os
from datetime import datetime

import optuna
import ray
from optuna.trial import TrialState
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from hopwise.config import Config
from hopwise.quick_start import objective_function, objective_function_optuna
from hopwise.trainer import HyperTuning


def hyperopt_tune(args):
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    # in other case, max_evals needs to be set manually
    config_file_list = args.config_files.strip().split(" ") if args.config_files else None
    hp = HyperTuning(
        objective_function,
        algo="exhaustive",
        early_stop=10,
        max_evals=100,
        params_file=args.params_file,
        fixed_config_file_list=config_file_list,
        display_file=args.display_file,
    )
    hp.run()
    hp.export_result(output_file=args.output_file)
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])


def ray_tune(args):
    config_file_list = args.config_files.strip().split(" ") if args.config_files else None
    config_file_list = [os.path.join(os.getcwd(), file) for file in config_file_list] if args.config_files else None
    params_file = os.path.join(os.getcwd(), args.params_file) if args.params_file else None
    ray.init(address="auto")
    tune.register_trainable("train_func", objective_function)
    config = {}
    with open(params_file) as fp:
        for line in fp:
            para_list = line.strip().split(" ")
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = (
                para_list[0],
                para_list[1],
                "".join(para_list[2:]),
            )
            if para_type == "choice":
                para_value = eval(para_value)
                config[para_name] = tune.choice(para_value)
            elif para_type == "uniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.uniform(float(low), float(high))
            elif para_type == "quniform":
                low, high, q = para_value.strip().split(",")
                config[para_name] = tune.quniform(float(low), float(high), float(q))
            elif para_type == "loguniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.loguniform(math.exp(float(low)), math.exp(float(high)))
            else:
                raise ValueError(f"Illegal param type [{para_type}]")
    # choose different schedulers to use different tuning optimization algorithms
    # For details, please refer to Ray's official website https://docs.ray.io
    scheduler = ASHAScheduler(metric="recall@10", mode="max", max_t=10, grace_period=1, reduction_factor=2)

    local_dir = "./ray_log"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=5,
        log_to_file=args.output_file,
        scheduler=scheduler,
        local_dir=local_dir,
        resources_per_trial={"gpu": 0},
    )

    best_trial = result.get_best_trial("ndcg@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


def optuna_tune(args):
    config_file_list = args.config_files.strip().split(" ") if args.config_files else None
    config = {}
    config = Config(config_dict={}, config_file_list=config_file_list)

    def objective(trial):
        with open(args.params_file) as fp:
            for line in fp:
                para_list = line.strip().split(" ")
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = (
                    para_list[0],
                    para_list[1],
                    "".join(para_list[2:]),
                )
                if para_type == "choice":
                    para_value = eval(para_value)
                    config[para_name] = trial.suggest_categorical(para_name, para_value)
                elif para_type == "uniform":
                    low, high = map(float, para_value.strip().split(","))
                    config[para_name] = trial.suggest_float(para_name, low, high)
                elif para_type == "loguniform":
                    low, high = map(float, para_value.strip().split(","))
                    config[para_name] = trial.suggest_loguniform(para_name, low, high)
                elif para_type == "quniform":
                    low, high, q = map(float, para_value.strip().split(","))
                    config[para_name] = trial.suggest_float(para_name, low, high, step=q)
                else:
                    raise ValueError(f"Illegal param type [{para_type}]")
        # Call the objective function with the suggested configuration
        result = objective_function_optuna(config=config, trial=trial)
        return result["best_valid_score"]

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.trials)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    print("Number of finished trials: ", len(study.trials))
    print("Number of complete trials: ", len(complete_trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Best Trial: ", study.best_trial)
    print("Best Params: ", study.best_params)
    print("Best Value: ", study.best_value)

    if args.save_trials:
        df = study.trials_dataframe()
        df.columns = df.columns.str.replace(r"^(datetime_|params_|user_attrs_)", "", regex=True)
        df.to_csv(f"{config['model']}_{args.output_file}", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_files", type=str, default=None, help="fixed config files")
    parser.add_argument("--params_file", type=str, default=None, help="parameters file")
    parser.add_argument("--output_file", type=str, default="hyper_example.result", help="output file")
    parser.add_argument("--display_file", type=str, default=None, help="visualization file")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials number")
    parser.add_argument("--tool", type=str, default="Optuna", help="{ray, hyperopt, optuna}")
    parser.add_argument(
        "--study_name",
        type=str,
        default=f"optuna_{datetime.now().strftime('%d_%m_%Y_%H:%M:%S')}",
        help="optuna study name",
    )
    parser.add_argument("--save_trials", type=bool, default=True, help="whether to save optuna trials in a csv file")
    args, _ = parser.parse_known_args()

    if args.tool == "Hyperopt":
        hyperopt_tune(args)
    elif args.tool == "Ray":
        ray_tune(args)
    elif args.tool == "Optuna":
        optuna_tune(args)
    else:
        raise ValueError(f"The tool [{args.tool}] should in ['Hyperopt', 'Ray', 'Optuna']")
