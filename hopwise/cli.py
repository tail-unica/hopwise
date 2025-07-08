import importlib
import os
import sys
import traceback
from datetime import datetime

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
from setproctitle import setproctitle

from hopwise.quick_start import objective_function, run
from hopwise.trainer import HyperTuning
from hopwise.utils import list_to_latex


class HopwiseClickCommand(click.Command):
    def parse_args(self, ctx, args):
        """Override to filter out HopWise parameters before Click's validation"""

        click_args = []
        hopwise_args = []

        for arg in args:
            if arg.startswith("--") and "=" in arg:
                hopwise_args.append(arg)
            else:
                click_args.append(arg)

        result = super().parse_args(ctx, click_args)
        ctx.args.extend(hopwise_args)

        return result


console = Console()
debug_message = """[dim]
    Use --debug for full traceback or --rich-traceback for enhanced formatting. Please, be careful
    that it should be placed after hopwise command and before any subcommand, e.g.:
    hopwise train --debug [model] [dataset] [--config-files config1 config2 ...]
    hopwise train --rich-traceback [model] [dataset] [--config-files config1 config2 ...]
    [/dim]"""


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug mode with full tracebacks")
@click.option("--rich-traceback", is_flag=True, help="Use Rich's enhanced traceback formatting")
@click.pass_context
def cli(ctx, debug, rich_traceback):
    """
    🔮 HopWise - Advanced Knowledge Graph-Enhanced Recommendation System

    HopWise extends RecBole with knowledge graphs, path-based reasoning,
    and language modeling for explainable recommendations.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["rich_traceback"] = rich_traceback

    # Only install Rich traceback if specifically requested
    if rich_traceback:
        install(console=console, show_locals=True)
    else:
        # Install minimal Rich traceback without the enhanced features
        install(console=console, show_locals=False, suppress=[click])


@cli.command(
    cls=HopwiseClickCommand,
    context_settings=dict(
        allow_interspersed_args=True,
    ),
)
@click.option("--model", "-m", default="BPR", help="Model name to train")
@click.option("--dataset", "-d", default="ml-100k", help="Dataset name")
@click.option("--config-files", help="Space-separated config files")
@click.option("--checkpoint", help="Checkpoint (.pth) file path")
@click.option("--nproc", default=1, help="Number of processes")
@click.option("--ip", default="localhost", help="Master node IP")
@click.option("--port", default="5678", help="Master node port")
@click.option("--world-size", default=-1, help="Total number of jobs")
@click.option("--group-offset", default=0, help="Global rank offset")
@click.option("--proc-title", default=None, help="Processor Title, shown in top, nvidia utils, etc.")
@click.pass_context
def train(ctx, model, dataset, config_files, nproc, checkpoint, ip, port, world_size, group_offset, proc_title):
    """Train or evaluate a single model."""

    if proc_title is None:
        proc_title = f"[hopwise] {model} {dataset} training"
    setproctitle(proc_title)

    config_file_list = config_files.strip().split(" ") if config_files else None

    try:
        run(
            model,
            dataset,
            "train",
            checkpoint,
            config_file_list=config_file_list,
            nproc=nproc,
            world_size=world_size,
            ip=ip,
            port=port,
            group_offset=group_offset,
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️  Training interrupted by user[/bold yellow]")
        sys.exit(130)

    except Exception as e:
        if ctx.obj.get("debug", False):
            if ctx.obj.get("rich_traceback", False):
                # Rich will handle this automatically with enhanced formatting
                raise
            else:
                console.print(f"[bold red]✗ Training failed:[/bold red] {type(e).__name__}: {str(e)}")
                console.print("\n[dim]Traceback:[/dim]")
                traceback.print_exc()
                sys.exit(1)
        else:
            console.print(f"[bold red]✗ Training failed:[/bold red] {type(e).__name__}: {str(e)}")
            console.print(debug_message)
            sys.exit(1)


@cli.command(
    cls=HopwiseClickCommand,
    context_settings=dict(
        allow_interspersed_args=True,
    ),
)
@click.option("--model", "-m", default="BPR", help="Model name to train")
@click.option("--dataset", "-d", default="ml-100k", help="Dataset name")
@click.option("--config-files", help="Space-separated config files")
@click.option("--checkpoint", help="Checkpoint (.pth) file path")
@click.option("--nproc", default=1, help="Number of processes")
@click.option("--ip", default="localhost", help="Master node IP")
@click.option("--port", default="5678", help="Master node port")
@click.option("--world-size", default=-1, help="Total number of jobs")
@click.option("--group-offset", default=0, help="Global rank offset")
@click.option("--proc-title", default=None, help="Processor Title, shown in top, nvidia utils, etc.")
@click.pass_context
def evaluate(ctx, model, dataset, config_files, nproc, checkpoint, ip, port, world_size, group_offset, proc_title):
    """Train or evaluate a single model."""

    if proc_title is None:
        proc_title = f"[hopwise] {model} {dataset} evaluation"
    setproctitle(proc_title)

    config_file_list = config_files.strip().split(" ") if config_files else None

    try:
        run(
            model,
            dataset,
            "evaluate",
            checkpoint,
            config_file_list=config_file_list,
            nproc=nproc,
            world_size=world_size,
            ip=ip,
            port=port,
            group_offset=group_offset,
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️  Evaluation interrupted by user[/bold yellow]")
        sys.exit(130)

    except Exception as e:
        if ctx.obj.get("debug", False):
            if ctx.obj.get("rich_traceback", False):
                # Rich will handle this automatically with enhanced formatting
                raise
            else:
                console.print(f"[bold red]✗ Evaluation failed:[/bold red] {type(e).__name__}: {str(e)}")
                console.print("\n[dim]Traceback:[/dim]")
                traceback.print_exc()
                sys.exit(1)
        else:
            console.print(f"[bold red]✗ Evaluation failed:[/bold red] {type(e).__name__}: {str(e)}")
            console.print(debug_message)
            sys.exit(1)


@cli.command(
    cls=HopwiseClickCommand,
    context_settings=dict(
        allow_interspersed_args=True,
    ),
)
@click.option("--models", "-m", required=True, help="Comma-separated model names")
@click.option("--dataset", "-d", default="ml-100k", help="Dataset name")
@click.option("--config-files", help="Space-separated config files")
@click.option("--valid-latex", default="./latex/valid.tex", help="Valid results LaTeX file")
@click.option("--test-latex", default="./latex/test.tex", help="Test results LaTeX file")
@click.option("--nproc", default=1, help="Number of processes")
@click.option("--ip", default="localhost", help="Master node IP")
@click.option("--port", default="5678", help="Master node port")
@click.option("--world-size", default=-1, help="Total number of jobs")
@click.option("--group-offset", default=0, help="Global rank offset")
@click.option("--proc-title", default=None, help="Processor Title, shown in top, nvidia utils, etc.")
def benchmark(
    models, dataset, config_files, valid_latex, test_latex, nproc, ip, port, world_size, group_offset, proc_title
):
    """
    Run scientific benchmark experiments across multiple models.

    Trains multiple models on the same dataset and generates comparative results
    in LaTeX table format for scientific publications. Ideal for reproducing
    paper results or conducting systematic model comparisons.

    Example:
        hopwise benchmark --models "BPR,LightGCN,KGAT" --dataset ml-100k --show-progress
    """

    if proc_title is None:
        proc_title = f"[hopwise - benchmark] {models} {dataset}"
    setproctitle(proc_title)

    model_list = [m.strip() for m in models.split(",")]
    config_file_list = config_files.strip().split(" ") if config_files else None

    os.makedirs(os.path.dirname(valid_latex), exist_ok=True)
    os.makedirs(os.path.dirname(test_latex), exist_ok=True)

    console.print(
        Panel(
            f"[bold blue]Model Benchmark[/bold blue]\n"
            f"Models: [green]{', '.join(model_list)}[/green]\n"
            f"Dataset: [green]{dataset}[/green]\n"
            f"Total runs: [yellow]{len(model_list)}[/yellow]",
            box=box.ROUNDED,
        )
    )

    valid_result_list = []
    test_result_list = []

    for idx, model in enumerate(model_list):
        console.print(f"\n[bold blue]📊 Training {model} ({idx + 1}/{len(model_list)})[/bold blue]")

        try:
            result = run(
                model,
                dataset,
                config_file_list=config_file_list,
                nproc=nproc,
                world_size=world_size,
                ip=ip,
                port=port,
                group_offset=group_offset,
            )

            valid_res_dict = {"Model": model}
            test_res_dict = {"Model": model}
            valid_res_dict.update(result["best_valid_result"])
            test_res_dict.update(result["test_result"])

            valid_result_list.append(valid_res_dict)
            test_result_list.append(test_res_dict)

            console.print(f"[green]✅ {model} completed successfully[/green]")

        except Exception as e:
            console.print(f"[red]❌ {model} failed: {str(e)}[/red]")

    successful = len(valid_result_list)
    failed = len(model_list) - successful
    console.print(f"\n[bold green]📋 Summary:[/bold green] {successful} successful, {failed} failed")

    # Generate LaTeX tables
    try:
        bigger_flag = result["valid_score_bigger"]
        subset_columns = list(result["best_valid_result"].keys())

        df_valid, tex_valid = list_to_latex(valid_result_list, bigger_flag, subset_columns)
        df_test, tex_test = list_to_latex(test_result_list, bigger_flag, subset_columns)

        with open(valid_latex, "w") as f:
            f.write(tex_valid)
        with open(test_latex, "w") as f:
            f.write(tex_test)

        console.print(f"[bold green]✓[/bold green] Results saved to {valid_latex} and {test_latex}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to generate LaTeX: {str(e)}")


@cli.command(
    cls=HopwiseClickCommand,
    context_settings=dict(
        allow_interspersed_args=True,
    ),
)
@click.argument("params-file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--config-files", help="Fixed config files")
@click.option("--output-path", default="saved/hyper", help="Output directory")
@click.option("--display-file", help="Visualization file")
@click.option("--max-evals", default=10, help="Maximum evaluations")
@click.option("--tool", type=click.Choice(["hyperopt", "ray", "optuna"]), default="optuna", help="Tuning tool")
@click.option("--study-name", help="Study name for tuning")
@click.option("--algo", help="Algorithm for the tuner")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
@click.option("--proc-title", default=None, help="Processor Title, shown in top, nvidia utils, etc.")
@click.pass_context
def tune(
    ctx, params_file, config_files, output_path, display_file, max_evals, tool, study_name, algo, resume, proc_title
):
    """Run hyperparameter tuning."""

    if proc_title is None:
        proc_title = f"[hopwise - hyper] {study_name}"
    setproctitle(proc_title)

    if not study_name:
        study_name = f"hyper_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

    config_file_list = config_files.strip().split(" ") if config_files else None

    console.print(
        Panel(
            f"[bold blue]Hyperparameter Tuning[/bold blue]\n"
            f"Tool: [green]{tool}[/green]\n"
            f"Max Evaluations: [yellow]{max_evals}[/yellow]\n"
            f"Study: [cyan]{study_name}[/cyan]",
            box=box.ROUNDED,
        )
    )

    try:
        ht = HyperTuning(
            objective_function,
            tuner=tool,
            algo=algo,
            early_stop=10,
            max_evals=max_evals,
            params_file=params_file,
            fixed_config_file_list=config_file_list,
            display_file=display_file,
            output_path=output_path,
            study_name=study_name,
            resume=resume,
        )

        console.print("[bold green]🚀[/bold green] Starting hyperparameter tuning...")
        ht.run()
        ht.export_result(output_path=output_path)

        console.print(
            Panel(
                f"[bold green]✓ Tuning Completed![/bold green]\n"
                f"Best params: [cyan]{ht.best_params}[/cyan]\n"
                f"Best result: [yellow]{ht.params2result[ht.params2str(ht.best_params)]}[/yellow]",
                title="Results",
                box=box.ROUNDED,
            )
        )

    except Exception as e:
        if ctx.obj.get("debug", False):
            if ctx.obj.get("rich_traceback", False):
                raise
            else:
                console.print(f"[bold red]✗ Tuning failed:[/bold red] {type(e).__name__}: {str(e)}")
                console.print("\n[dim]Traceback:[/dim]")
                traceback.print_exc()
                sys.exit(1)
        else:
            console.print(f"[bold red]✗[/bold red] Tuning failed: {str(e)}")
            console.print(debug_message)
            sys.exit(1)


@cli.command()
@click.option("--verbose", is_flag=True, help="Show detailed model list (docstrings)")
@click.option(
    "--type",
    "model_types",
    type=click.Choice(
        ["all", "Context", "Exlib", "General", "KG-aware", "KG-embed", "PathLM", "Sequential"], case_sensitive=False
    ),
    default=["all"],
    multiple=True,
    help="Filter by model type. Default is 'all' which shows all models.",
)
def models(verbose, model_types):
    """List available models."""
    model_types_map = {
        "Context": "context_aware_recommender",
        "Exlib": "exlib_recommender",
        "General": "general_recommender",
        "KG-aware": "knowledge_aware_recommender",
        "KG-embed": "knowledge_graph_embedding_recommender",
        "PathLM": "path_language_modeling_recommender",
        "Sequential": "sequential_recommender",
        "all": "all",
    }
    model_types = [model_types_map[t] for t in model_types]

    console.print("[bold blue]Available Models:[/bold blue]")
    models_dir = os.path.join(os.path.dirname(__file__), "model")
    models_info = []
    for dir_content in os.scandir(models_dir):
        if dir_content.is_dir():
            if "all" in model_types or dir_content.name.lower() in model_types:
                for filename in os.listdir(dir_content.path):
                    if filename.endswith(".py") and not filename.startswith("_"):
                        model_type = dir_content.name.lower()
                        model_lcase = filename[:-3]
                        model_module = importlib.import_module(f"hopwise.model.{model_type}.{model_lcase}")
                        model_name = [name for name in dir(model_module) if name.lower() == model_lcase][0]
                        model_module = getattr(model_module, model_name)
                        models_info.append(
                            {
                                "name": model_name,
                                "type": model_type,
                                "doc": getattr(model_module, "__doc__", "No documentation available"),
                            }
                        )
    for info in models_info:
        console.print(f"- [blue]{info['type']}[/blue] [green]{info['name']}[/green]")
        if verbose:
            console.print(f"  [yellow]{info['doc']}[/yellow]")


if __name__ == "__main__":
    cli()
