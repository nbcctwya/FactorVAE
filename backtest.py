"""
Backtest script for FactorVAE — converted from backtest.ipynb.

Pipeline:
    load model -> inference on test set -> merge LABEL0
    -> qlib backtest (TopkDropoutStrategy) -> risk analysis -> RankIC
    -> save scores / risk metrics / report figure

Usage:
    python backtest.py \
        --model_path ./best_models/VAE-Revision2_factor_96_hdn_64_port_128_seed_42.pt \
        --data_path data/csi_300_inference.pkl \
        --qlib_data_path ~/.qlib/qlib_data/cn_data
"""
import argparse
import os
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import qlib
from qlib.backtest import backtest, executor
from qlib.constant import REG_CN, REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils.time import Freq

from config_utils import get_config_section, load_config, parse_config_path
from dataset import init_data_loader
from utils import test_args, load_model, RankIC


@torch.no_grad()
def generate_prediction_scores(model, test_dataloader):
    """Run model.prediction on the test set and return a (datetime, instrument) -> score DataFrame."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model.to(device)
    model.eval()
    ls = []

    with tqdm(total=len(test_dataloader), desc="Inference") as pbar:
        for char_with_label, _ in test_dataloader:
            char = char_with_label[:, :, :-1].to(device)
            predictions = model.prediction(char.float())
            ls.append(predictions.detach().cpu())
            pbar.update(1)

    ls = torch.cat(ls, dim=0)
    indices = test_dataloader.dataset.sampler.get_index()
    multi_index = pd.MultiIndex.from_tuples(indices, names=["datetime", "instrument"])
    return pd.DataFrame(ls.numpy(), index=multi_index, columns=["score"])


def run_backtest(output, args):
    """qlib backtest with TopkDropoutStrategy. Returns (report_df, analysis_freq)."""
    region = REG_US if os.path.basename(args.qlib_data_path.rstrip("/")) == "us_data" else REG_CN
    qlib.init(provider_uri=args.qlib_data_path, region=region)

    strategy_config = {
        "topk": args.topk,
        "n_drop": args.n_drop,
        "signal": output,
    }
    executor_config = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    backtest_config = {
        "start_time": args.test_start,
        "end_time": args.test_end,
        "account": args.account,
        "benchmark": args.benchmark,
        "exchange_kwargs": {
            "freq": args.freq,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }
    strategy_obj = TopkDropoutStrategy(**strategy_config)
    executor_obj = executor.SimulatorExecutor(**executor_config)

    portfolio_metric_dict, _ = backtest(
        executor=executor_obj, strategy=strategy_obj, **backtest_config
    )
    analysis_freq = "{0}{1}".format(*Freq.parse(args.freq))
    report_normal_df, _ = portfolio_metric_dict.get(analysis_freq)
    return report_normal_df, analysis_freq


def save_report_figure(report_normal_df, save_dir):
    """Save qlib's position report as an interactive HTML (plotly)."""
    from qlib.contrib.report import analysis_position
    import qlib.contrib.report as qcr

    print("Available graphs:", qcr.GRAPH_NAME_LIST)
    try:
        fig = analysis_position.report_graph(report_normal_df)
        if fig is not None:
            path = os.path.join(save_dir, "report.html")
            fig.write_html(path)
            print(f"Saved report figure to {path}")
    except Exception as e:
        print(f"Skipping report figure: {e}")


def main():
    config_args = parse_config_path("FactorVAE backtest")
    backtest_config = get_config_section(load_config(config_args.config), "backtest")

    parser = argparse.ArgumentParser(description="FactorVAE backtest")
    parser.add_argument("--config", default=config_args.config, help="path to JSON config file")

    # model structure — must match the trained checkpoint
    parser.add_argument("--model_path", type=str, default=backtest_config.get("model_path"),
                        help="path to the trained .pt checkpoint")
    parser.add_argument("--num_factor", type=int, default=backtest_config.get("num_factor"))
    parser.add_argument("--num_portfolio", type=int, default=backtest_config.get("num_portfolio"))
    parser.add_argument("--hidden_size", type=int, default=backtest_config.get("hidden_size"))
    parser.add_argument("--num_latent", type=int, default=backtest_config.get("num_latent"))
    parser.add_argument("--run_name", type=str, default=backtest_config.get("run_name"))

    # data / inference
    parser.add_argument("--data_path", type=str, default=backtest_config.get("data_path"))
    parser.add_argument("--test_start", type=str, default=backtest_config.get("test_start"))
    parser.add_argument("--test_end", type=str, default=backtest_config.get("test_end"))
    parser.add_argument("--seq_len", type=int, default=backtest_config.get("seq_len"))
    parser.add_argument("--num_cols", type=int, default=backtest_config.get("num_cols"),
                        help="iloc[:, :num_cols]; 158 features + 1 label")

    # qlib backtest
    parser.add_argument("--qlib_data_path", type=str, default=backtest_config.get("qlib_data_path"))
    parser.add_argument("--benchmark", type=str, default=backtest_config.get("benchmark"))
    parser.add_argument("--topk", type=int, default=backtest_config.get("topk"))
    parser.add_argument("--n_drop", type=int, default=backtest_config.get("n_drop"))
    parser.add_argument("--freq", type=str, default=backtest_config.get("freq"))
    parser.add_argument("--account", type=float, default=backtest_config.get("account"))

    # output
    parser.add_argument("--save_dir", type=str, default=backtest_config.get("save_dir"))
    figure_group = parser.add_mutually_exclusive_group()
    figure_group.add_argument("--no_figure", dest="no_figure", action="store_true", help="skip saving the HTML report figure")
    figure_group.add_argument("--figure", dest="no_figure", action="store_false", help="save the HTML report figure")
    parser.set_defaults(no_figure=backtest_config.get("no_figure", False))

    args = parser.parse_args()
    if args.model_path is None:
        parser.error("--model_path must be set in the config file or passed on the command line")
    args.qlib_data_path = os.path.expanduser(args.qlib_data_path)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. load model
    cfg = test_args(
        run_name=args.run_name,
        num_factor=args.num_factor,
        num_portfolio=args.num_portfolio,
        hidden_size=args.hidden_size,
        num_latent=args.num_latent,
        normalize=False,
        select_feature=False,
        use_qlib=False,
    )
    model = load_model(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    print(f"Loaded model from {args.model_path}")

    # 2. load test data
    dataset = pd.read_pickle(args.data_path).iloc[:, : args.num_cols]
    dataset.rename(columns={dataset.columns[-1]: "LABEL0"}, inplace=True)
    test_dataloader = init_data_loader(
        dataset,
        shuffle=False,
        step_len=args.seq_len,
        start=args.test_start,
        end=args.test_end,
    )

    # 3. inference + merge label
    output = generate_prediction_scores(model, test_dataloader)
    output = pd.merge(output, dataset["LABEL0"], right_index=True, left_index=True)
    score_path = os.path.join(args.save_dir, "prediction_score.csv")
    output.to_csv(score_path)
    print(f"Saved prediction scores to {score_path}")

    # 4. backtest + risk analysis
    report_normal_df, analysis_freq = run_backtest(output, args)
    analysis = {
        "excess_return_without_cost": risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"], freq=analysis_freq
        ),
        "excess_return_with_cost": risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"],
            freq=analysis_freq,
        ),
    }
    analysis_df = pd.concat(analysis)
    print("\n===== Risk Analysis =====")
    pprint(analysis_df)
    analysis_df.to_csv(os.path.join(args.save_dir, "risk_analysis.csv"))

    # 5. RankIC
    rankic = RankIC(output.dropna(axis=0), column1="score", column2="LABEL0")
    print("\n===== RankIC =====")
    print(rankic)
    rankic.to_csv(os.path.join(args.save_dir, "rankic.csv"), index=False)

    # 6. (optional) report figure
    if not args.no_figure:
        save_report_figure(report_normal_df, args.save_dir)

    print(f"\nAll results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
