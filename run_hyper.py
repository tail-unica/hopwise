# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29, 2022/7/13, 2022/7/18
# @Author : Zihan Lin, Yupeng Hou, Gaowei Zhang, Lei Wang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, zgw15630559577@163.com, zxcptss@gmail.com

import argparse
from datetime import datetime

from setproctitle import setproctitle

from hopwise.quick_start import objective_function
from hopwise.trainer import HyperTuning


def tune(args):
    # plz set algo='exhaustive' to use exhaustive search with hyperopt, in this case, max_evals is auto set
    # in other case, max_evals needs to be set manually
    ht = HyperTuning(
        objective_function,
        tuner=args.tool,
        algo=args.algo,
        early_stop=10,
        max_evals=args.max_evals,
        params_file=args.params_file,
        fixed_config_file_list=args.config_file_list,
        display_file=args.display_file,
        output_path=args.output_path,
        study_name=args.study_name,
        show_progress=args.show_progress,
        load_previous_study=args.load_previous_study,
    )
    ht.run()
    ht.export_result(output_path=args.output_path)
    print("best params: ", ht.best_params)
    print("best result: ")
    print(ht.params2result[ht.params2str(ht.best_params)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, help="parameters file")
    parser.add_argument("--config_files", type=str, default=None, help="fixed config files")
    parser.add_argument("--output_path", type=str, default="saved/hyper", help="output file")
    parser.add_argument("--display_file", type=str, default=None, help="visualization file")
    parser.add_argument("--max_evals", type=int, default=10, help="max evaluations")
    parser.add_argument(
        "--load_previous_study",
        type=bool,
        default=True,
        help="if true and study_name exists in sqlite db, load previous study",
    )
    parser.add_argument("--proc_title", type=str, default=None, help="processor title, shown in top, nvidia-smi, ecc.")
    parser.add_argument(
        "--show_progress", type=bool, default=True, help="whether to show progress bar during training and evaluation"
    )
    tool_action = parser.add_argument(
        "--tool", type=str, default="optuna", choices=["hyperopt", "ray", "optuna"], help="tuning tool"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=f"hyper_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}",
        help="Trial study name for hyper tuning.",
    )
    hyperopt_algo = parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Algorithm used by the tuner.\n"
        "For hyperopt it can be: random, tpe, anneal, or exhaustive.\n"
        "For optuna it identifies sampler and pruner separated by '-'. One of them can be omitted. "
        "Examples: TPESampler, HyperbandPruner, RandomSampler-MedianPruner.\n"
        "For ray it identifies searcher and scheduler separated by '-'. One of them can be omitted. "
        "Examples: async_hyperband, bohb, bohb-pbt_replay. \n"
        "For more details, please refer to the official website of the corresponding tool.",
    )
    args, _ = parser.parse_known_args()

    args.config_file_list = args.config_files.strip().split(" ") if args.config_files else None

    if args.proc_title is None:
        args.proc_title = f"[hopwise - hyper] {args.study_name}"
    setproctitle(args.proc_title)

    tune(args)
