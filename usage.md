# Usage

This document records the common workflows for this FactorVAE project: dataset construction, training, backtesting, and running multiple seeds.

## 1. Environment

Activate the conda environment first:

```bash
conda activate factorvae
```

All commands below assume they are run from the project root:

```bash
cd ~/baselines/FactorVAE
```

## 2. Build Datasets

Datasets are generated from Qlib Alpha158 features using `data/make_dataset.py`.

The script supports:

- `csi300`: saves to `data/csi_data.pkl`
- `sp500`: saves to `data/sp500_data.pkl`

Default Qlib provider paths are:

- CSI300: `~/.qlib/qlib_data/cn_data`
- SP500: `~/.qlib/qlib_data/us_data`

Generate CSI300:

```bash
python data/make_dataset.py --universe csi300
```

Generate SP500:

```bash
python data/make_dataset.py --universe sp500
```

If your Qlib data is stored somewhere else, override it with `--data_path`:

```bash
python data/make_dataset.py \
  --universe csi300 \
  --data_path ~/.qlib/qlib_data/cn_data
```

Useful date arguments:

```bash
python data/make_dataset.py \
  --universe csi300 \
  --start_time 2009-01-01 \
  --fit_end_time 2020-12-31 \
  --val_start_time 2021-01-01 \
  --val_end_time 2022-12-31 \
  --test_start_time 2023-01-01 \
  --end_time 2025-12-31
```

## 3. Train A Single Model

Train CSI300 with seed 42:

```bash
python main.py \
  --dataset ./data/csi_data.pkl \
  --run_name FactorVAE-csi300 \
  --seed 42 \
  --num_epochs 30 \
  --num_factor 96 \
  --hidden_size 64 \
  --num_portfolio 128
```

Train SP500 with seed 42:

```bash
python main.py \
  --dataset ./data/sp500_data.pkl \
  --run_name FactorVAE-sp500 \
  --seed 42 \
  --num_epochs 30 \
  --num_factor 96 \
  --hidden_size 64 \
  --num_portfolio 128
```

The best validation model is saved as a plain model state dict:

```text
best_models/FactorVAE-csi300_factor_96_hdn_64_port_128_seed_42.pt
best_models/FactorVAE-sp500_factor_96_hdn_64_port_128_seed_42.pt
```

These `.pt` files are the ones to use for inference and backtesting.

## 4. Resume Training

Training also writes a resumable checkpoint after each completed epoch:

```text
best_models/FactorVAE-csi300_factor_96_hdn_64_port_128_seed_42_checkpoint.pt
```

Resume using the automatically inferred checkpoint path:

```bash
python main.py \
  --dataset ./data/csi_data.pkl \
  --run_name FactorVAE-csi300 \
  --seed 42 \
  --num_epochs 30 \
  --num_factor 96 \
  --hidden_size 64 \
  --num_portfolio 128 \
  --resume auto
```

`--num_epochs 30` means "train until 30 total epochs", not "train 30 more epochs".

Checkpoint files store:

- model weights
- optimizer state
- scheduler state
- current epoch
- best validation loss
- RNG states

You can stop training with `Ctrl+C`. Progress is saved at the end of each epoch, so stopping in the middle of an epoch will lose only that unfinished epoch.

## 5. Batch Training Multiple Seeds

Use the queue script:

```bash
./scripts/run_training_queue.sh
```

It runs in this order:

```text
seed 42: csi300
seed 42: sp500
seed 43: csi300
seed 43: sp500
...
seed 46: csi300
seed 46: sp500
```

The default settings are inside `scripts/run_training_queue.sh`:

```bash
SEEDS=(42 43 44 45 46)
NUM_EPOCHS=30
NUM_FACTOR=96
HIDDEN_SIZE=64
NUM_PORTFOLIO=128
CONDA_ENV="factorvae"
```

The script behavior is:

- If no checkpoint exists, start that seed/universe from scratch.
- If a checkpoint exists and is unfinished, resume it automatically.
- If a checkpoint has already completed `NUM_EPOCHS`, skip it.

To check script syntax without running training:

```bash
bash -n scripts/run_training_queue.sh
```

To actually run training:

```bash
./scripts/run_training_queue.sh
```

## 6. Backtest From A Trained Model

Use `backtest.py` to load a trained `.pt` model, run inference on the test period, and run a Qlib TopK-DropN backtest.

CSI300 example:

```bash
python backtest.py \
  --model_path ./best_models/FactorVAE-csi300_factor_96_hdn_64_port_128_seed_42.pt \
  --data_path ./data/csi_data.pkl \
  --qlib_data_path ~/.qlib/qlib_data/cn_data \
  --benchmark SH000300 \
  --test_start 2023-01-01 \
  --test_end 2025-12-31 \
  --topk 50 \
  --n_drop 10 \
  --save_dir ./backtest_results/csi300_seed42
```

SP500 example:

```bash
python backtest.py \
  --model_path ./best_models/FactorVAE-sp500_factor_96_hdn_64_port_128_seed_42.pt \
  --data_path ./data/sp500_data.pkl \
  --qlib_data_path ~/.qlib/qlib_data/us_data \
  --benchmark ^gspc \
  --test_start 2023-01-01 \
  --test_end 2025-12-31 \
  --topk 50 \
  --n_drop 10 \
  --save_dir ./backtest_results/sp500_seed42
```

Backtest outputs include:

```text
prediction_score.csv
risk_analysis.csv
rankic.csv
report.html
```

Use `--no_figure` if you do not want to save the HTML report:

```bash
python backtest.py \
  --model_path ./best_models/FactorVAE-csi300_factor_96_hdn_64_port_128_seed_42.pt \
  --data_path ./data/csi_data.pkl \
  --qlib_data_path ~/.qlib/qlib_data/cn_data \
  --benchmark SH000300 \
  --save_dir ./backtest_results/csi300_seed42 \
  --no_figure
```

## 7. Qlib-Native Backtest From Prediction Scores

`backtest_qlib.py` is used when you already have a prediction file, such as a pickle file with `(datetime, instrument)` scores.

Example:

```bash
python backtest_qlib.py \
  --pred_path ./scores/csi300_seed42_prediction.pkl \
  --universe csi300 \
  --start_time 2023-01-01 \
  --end_time 2025-12-31 \
  --topk 30 \
  --drop 5
```

Note: `backtest_qlib.py` reads prediction files with `pd.read_pickle`, so use it with pickle prediction files. The main `backtest.py` path above writes `prediction_score.csv`; for that default workflow, use `backtest.py` directly.

## 8. File Naming

Best model files:

```text
best_models/FactorVAE-csi300_factor_96_hdn_64_port_128_seed_42.pt
best_models/FactorVAE-sp500_factor_96_hdn_64_port_128_seed_42.pt
```

Resume checkpoint files:

```text
best_models/FactorVAE-csi300_factor_96_hdn_64_port_128_seed_42_checkpoint.pt
best_models/FactorVAE-sp500_factor_96_hdn_64_port_128_seed_42_checkpoint.pt
```

Use the non-`_checkpoint` `.pt` files for backtesting.
