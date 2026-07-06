"""
Generating data using Qlib
Alpha158 is an off-the-shelf dataset provided by Qlib.
"""

import qlib
import pandas as pd
import sys
from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from qlib.contrib.data.handler import Alpha158
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from config_utils import get_config_section, load_config, parse_config_path


UNIVERSE_SETTINGS = {
    "csi300": {
        "region": REG_CN,
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000300",
        "output_file": "csi_data.pkl",
    },
    "sp500": {
        "region": REG_US,
        "provider_uri": "~/.qlib/qlib_data/us_data",
        "benchmark": "^gspc",
        "output_file": "sp500_data.pkl",
    },
}


if __name__ == "__main__":

    config_args = parse_config_path("Generate FactorVAE datasets using Qlib")
    data_config = get_config_section(load_config(config_args.config), "make_dataset")

    # Argument parser 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_args.config, help="path to JSON config file")
    parser.add_argument("--universe", choices=sorted(UNIVERSE_SETTINGS), default=data_config.get("universe", "csi300"))
    parser.add_argument("--data_path", type=str, default=None, help="override Qlib provider path")
    parser.add_argument("--freq", type=str, default=data_config.get("freq"))
    parser.add_argument('--start_time', type=str, default=data_config.get("start_time"))
    parser.add_argument('--end_time', type=str, default=data_config.get("end_time"))
    parser.add_argument('--fit_end_time', type=str, default=data_config.get("fit_end_time"))
    parser.add_argument('--val_start_time', type=str, default=data_config.get("val_start_time"))
    parser.add_argument('--val_end_time', type=str, default=data_config.get("val_end_time"))
    parser.add_argument('--test_start_time', type=str, default=data_config.get("test_start_time"))
    parser.add_argument('--seq_len', type=int, default=data_config.get("seq_len"))
    parser.add_argument("--output_dir", type=str, default=data_config.get("output_dir", "./data"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Qlib을 이용한 데이터 생성
    universe_config = UNIVERSE_SETTINGS[args.universe]
    provider_uri = args.data_path or data_config.get("data_path") or universe_config["provider_uri"]
    provider_uri = str(Path(provider_uri).expanduser())
    qlib.init(provider_uri=provider_uri, region=universe_config["region"])
    benchmark = universe_config["benchmark"]
    market = args.universe

    print(f"provider_uri: {provider_uri}")
    print(f"universe: {args.universe}")
    print(f"freq: {args.freq}")

    data_handler_config = {
        "start_time": args.start_time,
        "end_time": args.end_time,
        "fit_start_time": args.start_time,
        "fit_end_time": args.fit_end_time,
        "instruments": market,
        "infer_processors": [
            # {"class" : "FilterCol", "kwargs" : {"fields_group" : "feature"},},
            {"class" : "RobustZScoreNorm","kwargs" : {"fields_group" : "feature", "clip_outlier" : True}},
            {"class" : "Fillna", "kwargs" : {"fields_group" : "feature"}}],
        "learn_processors": [
            {"class" : "DropnaLabel",}, 
            {"class" : "CSRankNorm", "kwargs" : {"fields_group" : "label"}}, # ! CSZScoreNorm 에서 CSRankNorm으로 변경
            ],
        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
    }

    segments = {
        'train': (args.start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.end_time)
    }
    dataset = Alpha158(**data_handler_config)

    dataframe_L = dataset.fetch(col_set=["feature","label"], data_key=DataHandlerLP.DK_L) 
    dataframe_L.columns = dataframe_L.columns.droplevel(0)

    dataframe_I = dataset.fetch(col_set=["feature","label"], data_key=DataHandlerLP.DK_I)
    dataframe_I.columns = dataframe_I.columns.droplevel(0)

    #? market info not included in the dataset
    dataframe_LM = dataframe_L
    dataframe_IM = dataframe_I
    output_path = output_dir / universe_config["output_file"]

    dataframe_LM.to_pickle(output_path)
    print(f"Saved dataset to {output_path}")

    ## TEST ##
    segments = {
        'train': (args.start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.end_time)
    }

    handler = DataHandlerLP.from_df(dataframe_LM)
    QlibTSDatasetH = TSDatasetH(handler=handler, segments=segments, step_len=args.seq_len)
    temp = QlibTSDatasetH.prepare(segments="train", data_key=DataHandlerLP.DK_L)

    print("------------------ Test QlibTSDatasetH ------------------")
    print(next(iter(temp)))
