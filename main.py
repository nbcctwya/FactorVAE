import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import argparse
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from config_utils import get_config_section, load_config, parse_config_path
from dataset import init_data_loader
from train_model import train, validate
from utils import set_seed, DataArgument

try:
    import wandb
except ImportError:
    wandb = None


def main(args, data_args):
    
    set_seed(args.seed)
    print(f"Starting run: {args.run_name}", flush=True)
    print(f"Dataset path: {args.dataset}", flush=True)
    print(
        f"Train: {data_args.start_time} -> {data_args.fit_end_time}; "
        f"Valid: {data_args.val_start_time} -> {data_args.val_end_time}",
        flush=True,
    )
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # create model
    print("Creating model...", flush=True)
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    
    # create dataloaders
    print("Loading dataset...", flush=True)
    dataset = pd.read_pickle(args.dataset).iloc[:, :159] # market info 제외
    dataset.rename(columns={dataset.columns[-1]: 'LABEL0'}, inplace=True) # 마지막 컬럼 이름 변경 'LABEL0'
    print(f"Loaded dataset shape: {dataset.shape}", flush=True)
    print("Building train dataloader...", flush=True)
    train_dataloader = init_data_loader(dataset,
                                        shuffle=True,
                                        step_len=data_args.seq_len, 
                                        start=data_args.start_time,
                                        end=data_args.fit_end_time, 
                                        select_feature=data_args.select_feature)
    
    print("Building validation dataloader...", flush=True)
    valid_dataloader = init_data_loader(dataset,
                                        shuffle=False, 
                                        step_len=data_args.seq_len, 
                                        start=data_args.val_start_time, 
                                        end=data_args.val_end_time, 
                                        select_feature=data_args.select_feature)
    print(
        f"Train batches: {len(train_dataloader)}, validation batches: {len(valid_dataloader)}",
        flush=True,
    )
    
    T_max = len(train_dataloader) * args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*************** Using {device} ***************", flush=True)
    args.device = device
        
    factorVAE.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is required when --wandb is set. Install it or run without --wandb.")
        wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")
        wandb.config.update(args)

    print(f"Starting training for {args.num_epochs} epochs...", flush=True)
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, scheduler, args)
        val_loss = validate(factorVAE, valid_dataloader, args)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}", flush=True) 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #? save model in save_dir
            
            #? torch.save
            save_root = os.path.join(args.save_dir, f'{args.run_name}_factor_{args.num_factor}_hdn_{args.hidden_size}_port_{args.num_portfolio}_seed_{args.seed}.pt')
            torch.save(factorVAE.state_dict(), save_root)
            print(f"Model saved at {save_root}", flush=True)
            
        if args.wandb:
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, "Learning Rate": scheduler.get_last_lr()[0]})
    
    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()
    
if __name__ == '__main__':
    config_args = parse_config_path('Train a FactorVAE model on stock data')
    train_config = get_config_section(load_config(config_args.config), "train")

    parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')
    parser.add_argument('--config', default=config_args.config, help='path to JSON config file')

    parser.add_argument('--num_epochs', type=int, default=train_config.get("num_epochs"), help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=train_config.get("lr"), help='learning rate')

    parser.add_argument('--num_latent', type=int, default=train_config.get("num_latent"), help='number of variables')
    parser.add_argument('--num_portfolio', type=int, default=train_config.get("num_portfolio"), help='number of stocks')

    parser.add_argument('--seq_len', type=int, default=train_config.get("seq_len"), help='sequence length')
    parser.add_argument('--num_factor', type=int, default=train_config.get("num_factor"), help='number of factors')
    parser.add_argument('--hidden_size', type=int, default=train_config.get("hidden_size"), help='hidden size')

    parser.add_argument('--dataset', type=str, default=train_config.get("dataset"), help='dataset to use')
    parser.add_argument('--start_time', type=str, default=train_config.get("start_time"), help='start time')
    parser.add_argument('--fit_end_time', type=str, default=train_config.get("fit_end_time"), help='fit end time')
    parser.add_argument('--val_start_time', type=str, default=train_config.get("val_start_time"), help='validation start time')
    parser.add_argument('--val_end_time', type=str, default=train_config.get("val_end_time"), help='validation end time')
    parser.add_argument('--end_time', type=str, default=train_config.get("end_time"), help='end time')

    parser.add_argument('--seed', type=int, default=train_config.get("seed"), help='random seed')
    parser.add_argument('--run_name', type=str, default=train_config.get("run_name"), help='name of the run')
    parser.add_argument('--save_dir', type=str, default=train_config.get("save_dir"), help='directory to save model')
    parser.add_argument('--num_workers', type=int, default=train_config.get("num_workers"), help='number of workers for dataloader')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=train_config.get("wandb", False), help='whether to use wandb')
    args = parser.parse_args()

    data_args = DataArgument(
        start_time=args.start_time,
        end_time=args.end_time,
        fit_end_time=args.fit_end_time,
        val_start_time=args.val_start_time,
        val_end_time=args.val_end_time,
        seq_len=args.seq_len,
    )

    main(args, data_args)
