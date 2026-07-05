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
from qlib.contrib.report import analysis_position
import qlib.contrib.report as qcr
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils.time import Freq

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
    parser = argparse.ArgumentParser(description="FactorVAE backtest")

    # model structure — must match the trained checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                        help="path to the trained .pt checkpoint")
    parser.add_argument("--num_factor", type=int, default=96)
    parser.add_argument("--num_portfolio", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_latent", type=int, default=158)
    parser.add_argument("--run_name", type=str, default="VAE-Revision2")

    # data / inference
    parser.add_argument("--data_path", type=str, default="data/csi_300_inference.pkl")
    parser.add_argument("--test_start", type=str, default="2023-01-01")
    parser.add_argument("--test_end", type=str, default="2025-12-31")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--num_cols", type=int, default=159,
                        help="iloc[:, :num_cols]; 158 features + 1 label")

    # qlib backtest
    parser.add_argument("--qlib_data_path", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--benchmark", type=str, default="SH000300")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n_drop", type=int, default=10)
    parser.add_argument("--freq", type=str, default="day")
    parser.add_argument("--account", type=float, default=1e8)

    # output
    parser.add_argument("--save_dir", type=str, default="./backtest_results")
    parser.add_argument("--no_figure", action="store_true", help="skip saving the HTML report figure")

    args = parser.parse_args()
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
