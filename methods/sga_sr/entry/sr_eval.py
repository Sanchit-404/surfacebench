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
import warp as wp


import sga

root: Path = sga.utils.get_root(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--problem_name', type=str, required=True)

    args, unknown_args = parser.parse_known_args()
    # cfg = sga.config.TrainConfig(path=args.path, dataset_path=args.dataset_path)
    cfg = sga.config.EvalConfig(path=args.path, dataset_path=args.dataset_path)
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

    print(exp_name, exp_root)

    state_root = exp_root / 'state'
    ckpt_root = exp_root / 'ckpt'
    output_root = exp_root / 'outputs'
    sga.utils.mkdir(exp_root, overwrite=False, resume=True)
    state_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(state_root, ckpt_root, output_root)

    cfg_dict = dataclasses.asdict(cfg)
    with (exp_root / 'config.yaml').open('w') as f:
        yaml.safe_dump(cfg_dict, f)
    

    # writer = SummaryWriter(exp_root, purge_step=0)

    print(cfg.physics.env.physics.path)
    full_py = Path(cfg.physics.env.physics.path).read_text('utf-8')
    full_py = full_py.format(**cfg.physics.env.physics.__dict__)
    physics_py_path = exp_root / 'physics.py'
    physics_py_path.write_text(full_py, 'utf-8')

    # print(cfg.physics.env.physics.__dict__)
    # print(full_py)


    physics: nn.Module = sga.utils.get_class_from_path(physics_py_path, 'SymbolicEquation')()
    print(cfg.ckpt_path)
    
    if cfg.ckpt_path is not None:
        ckpt_path = Path(cfg.ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        physics.load_state_dict(ckpt)
    
    physics.to(torch_device)
    physics.requires_grad_(False)
    physics.eval()


    # parametric = len(list(physics.parameters())) > 0
    # # if parametric:
    # #     if cfg.optim.optimizer == 'adam':
    # #         optimizer = torch.optim.Adam(physics.parameters(), lr=cfg.optim.lr)
    # #     else:
    # #         raise ValueError(f'Unknown optimizer: {cfg.optim.optimizer}')

    # #     if cfg.optim.scheduler == 'cosine':
    # #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.num_epochs)
    # #     elif cfg.optim.scheduler == 'none':
    # #         scheduler = None
    # #     else:
    # #         raise ValueError(f'Unknown scheduler: {cfg.optim.scheduler}')

    print(cfg.dataset_path)
    dm = sga.sr.datamodules.get_datamodule(cfg.dataset_path)
    dm.setup()

    problem = dm.problems[dm.name2id[args.problem_name]]
    # # samples = problem.test_samples

    # exp_name = "SR"
    # tpos = cfg.tpos
    
    # state_recorder = sga.utils.StateRecorder()
    # state_recorder.add_hyper(key_indices=cfg.physics.env.shape.key_indices)

    with torch.no_grad():
        id_X = torch.tensor(problem.test_samples[:, 1:], dtype=torch.float).to(torch_device)
        print(id_X.shape)
        id_y = physics(*[id_X[:, i] for i in range(id_X.shape[1])]).cpu().squeeze().numpy()
        assert id_y.shape == problem.test_samples[:, 0].shape, str(id_y.shape) + ", " + str(problem.test_samples[:, 0].shape)
        save_results = {
            "id_y": id_y,
        }
        
        if problem.ood_test_samples is not None:
            ood_X = torch.tensor(problem.ood_test_samples[:, 1:], dtype=torch.float).to(torch_device)
            ood_y = physics(*[ood_X[:, i] for i in range(ood_X.shape[1])]).cpu().squeeze().numpy()
            assert ood_y.shape == problem.ood_test_samples[:, 0].shape, str(ood_y.shape) + ", " + str(problem.ood_test_samples[:, 0].shape)
            save_results['ood_y'] = ood_y

    np.savez(output_root / "outputs.npz", **save_results)

    # y_gt = torch.tensor(y_gt, dtype=torch.float).to(torch_device)
    # loss_y = sga.utils.loss_fn(y, y_gt)



        


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

    #     loss_y_item = loss_y.item()
    #     # loss_v_item = loss_v

    #     if math.isnan(loss_y_item):
    #         writer.add_scalar('loss/output', float('nan') , epoch)
    #         tqdm.write('loss is nan')
    #         break

    #     writer.add_scalar('loss/output', loss_y_item, epoch)
    #     t.set_postfix(l_y=loss_y_item)

    #     if epoch == num_epochs:
    #         t.refresh()
    #         break

    #     if not parametric:
    #         break

    #     optimizer.zero_grad()
    #     try:
    #         loss_y.backward()
    #         clip_grad_norm_(physics.parameters(), 1.0)
    #         optimizer.step()
    #         if scheduler is not None:
    #             scheduler.step()
    #     except RuntimeError as e:
    #         tqdm.write(str(e))
    #         break
    # t.close()

    # state_recorder.save(state_root / 'ckpt.pt')
    # writer.close()

if __name__ == '__main__':
    main()
