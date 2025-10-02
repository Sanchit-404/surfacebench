import os
import sys
import dataclasses
import random
from pathlib import Path
import argparse
import math

import yaml
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.utils.tensorboard import SummaryWriter
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from bench.datamodules import get_datamodule

sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
import sga

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
root: Path = sga.utils.get_root(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--problem_name', type=str, required=True)

    args, unknown_args = parser.parse_known_args()
    if args.dataset_path == "None":
        args.dataset_path = None
    cfg = sga.config.TrainConfig(path=args.path, 
                                 dataset_name=args.dataset_name,
                                 dataset_path=args.dataset_path)
    cfg.update(unknown_args)

    torch_device = torch.device(f'cuda:{cfg.gpu}') 

    log_root = root / 'log'
    if Path(cfg.path).is_absolute():
        exp_root = Path(cfg.path)
    else:
        exp_root = log_root / cfg.path
    if exp_root.is_relative_to(log_root):
        exp_name = str(exp_root.relative_to(log_root))
    else:
        exp_name = str(exp_root)
    state_root = exp_root / 'state'
    ckpt_root = exp_root / 'ckpt'
    sga.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    state_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    with (exp_root / 'config.yaml').open('w') as f:
        yaml.safe_dump(cfg_dict, f)

    writer = SummaryWriter(exp_root, purge_step=0)

    full_py = Path(cfg.physics.env.physics.path).read_text('utf-8')
    if 'for i in' in full_py:
        raise RuntimeError('dead loop detected')
    if 'for b in' in full_py:
        raise RuntimeError('dead loop detected')
    if 'for f in' in full_py:
        raise RuntimeError('dead loop detected')
    print(cfg.physics.env.physics.__dict__)
    full_py = full_py.format(**cfg.physics.env.physics.__dict__)
    physics_py_path = exp_root / 'physics.py'
    physics_py_path.write_text(full_py, 'utf-8')

    physics: nn.Module = sga.utils.get_class_from_path(physics_py_path, 'SymbolicEquation')()
    physics.to(torch_device)
    print(physics_py_path)


    parametric = len(list(physics.parameters())) > 0

    if parametric:
        if cfg.optim.optimizer == 'adam':
            optimizer = torch.optim.Adam(physics.parameters(), lr=cfg.optim.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optim.optimizer}')

        if cfg.optim.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.num_epochs)
        elif cfg.optim.scheduler == 'none':
            scheduler = None
        else:
            raise ValueError(f'Unknown scheduler: {cfg.optim.scheduler}')
    
    dm = get_datamodule(name=cfg.dataset_name, 
                        root_folder=cfg.dataset_path)
    dm.setup()
    problem = dm.problems[dm.name2id[args.problem_name]]
    samples = problem.train_samples

    num_epochs = 1000
    # num_epochs = 1000
    exp_name = "SR"
    tpos = cfg.tpos
    # X_train, y_gt_train = samples[:, 1:], samples[:, 0]
    # X_train = torch.tensor(X_train, dtype=torch.float).to(torch_device)
    # y_gt_train = torch.tensor(y_gt_train, dtype=torch.float).to(torch_device)

    X, y_gt = samples[:, 1:], samples[:, 0]
    X = torch.tensor(X, dtype=torch.float).to(torch_device)
    y_gt = torch.tensor(y_gt, dtype=torch.float).to(torch_device)
    
    t = trange(num_epochs + 1, 
               desc=f'[train] {exp_name}', file=sys.stdout, position=tpos, leave=None)
    for epoch in t:
        torch.save(physics.state_dict(), ckpt_root / f'{epoch:04d}.pt')

        if parametric:
            for name, param in physics.named_parameters():
                writer.add_scalar(f'param/{name}', param.item(), epoch)
        
        physics.train()
        
        # sel_inds = torch.randperm(X_train.shape[0])[:4000]
        # X = X_train[sel_inds]
        # y_gt = y_gt_train[sel_inds]

        y = physics(*[X[:, i] for i in range(X.shape[1])])

        y = y.view(-1)
        y_gt = y_gt.view(-1)

        loss_y = sga.utils.loss_fn(y, y_gt)

        # state_recorder = sga.utils.StateRecorder()
        # state_recorder.add(x=x_gt, v=v_gt)

        # for step in range(num_steps):

        #     is_teacher = step == 0 or (num_teacher_steps > 0 and step % num_teacher_steps == 0)
        #     if is_teacher:
        #         x, v, C, F = x_gt, v_gt, C_gt, F_gt

        #     stress = elasticity(F)
        #     x, v, C, F = diff_sim(step, x, v, C, F, stress)
        #     # state_recorder.add(x=x, v=v)

        #     x_gt, v_gt, C_gt, F_gt, _ = dataset[step + 1]
        #     loss_x += sga.utils.loss_fn(x, x_gt) / num_steps * cfg.optim.alpha_position
        #     loss_v += sga.utils.loss_fn(v, v_gt).item() / num_steps * cfg.optim.alpha_velocity

        # state_recorder.save(state_root / f'{epoch:04d}.pt')

        loss_y_item = loss_y.item()
        # print("Loss:", loss_y_item)
        # loss_v_item = loss_v

        if epoch % 10 == 0:
            output_symbol = problem.gt_equation.symbols[0]
            if math.isnan(loss_y_item):
                writer.add_scalar(f'loss/{output_symbol}', float('nan') , epoch)
                break
            else:
                writer.add_scalar(f'loss/{output_symbol}', loss_y_item, epoch)

        if math.isnan(loss_y_item):
            tqdm.write('loss is nan')
            break

        t.set_postfix(l_y=loss_y_item)
        if epoch == num_epochs:
            t.refresh()
            break

        if not parametric:
            break

        optimizer.zero_grad()
        try:
            loss_y.backward()
            clip_grad_norm_(physics.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            tqdm.write(str(e))
            break
    t.close()

    torch.save(physics.state_dict(), ckpt_root / 'final.pt')
    writer.close()

if __name__ == '__main__':
    main()
