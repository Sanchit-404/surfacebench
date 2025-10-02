from typing import List

import os, sys
from pathlib import Path
import dataclasses
import yaml, json

import math
import random
import numpy as np
import torch

from tqdm import trange, tqdm

from bench.data_classes import Equation, SEDTask, SearchResult
from bench.searchers.base import BaseSearcher
import sga
from sga.agent import SRPhysicist, Population


def get_perf_feedback(losses: dict[str, list[float]], params: dict[str, list[float]], output_symbol="output") -> list[str]:
    feedbacks = []

    parametric = len(params) > 0

    if parametric:
        best_idx = min(range(len(losses[output_symbol])), key=lambda i: losses[output_symbol][i] if not math.isnan(
            losses[output_symbol][i]) else float('inf'))

        feedbacks.append('#### Optimized Physical parameter')
        feedbacks.append('')  # add a blank line
        for tag, traj in sorted(params.items()):
            # msg = ', '.join([f'{loss:.2f}' for loss in traj])
            # msg = f'- {tag}: [{msg}] (Best: {traj[best_idx]:.2f})'
            msg = f'- {tag}: (Best: {traj[best_idx]:.2e})'
            feedbacks.append(msg)

        # feedbacks.append('')  # add a blank line
        feedbacks.append('#### Loss training curves (versus iteration)')
        feedbacks.append('')  # add a blank line
        for tag, traj in sorted(losses.items()):
            msg = ', '.join([f'{loss:.4e}' for loss in traj])
            if tag == output_symbol:
                tag = f'{tag} (Key loss)'
            msg = f'- {tag}: [{msg}] (Best: {traj[best_idx]:.4e})'
            feedbacks.append(msg)
    else:
        feedbacks.append('#### Evaluation loss (since it is a non-parametric model)')
        feedbacks.append('')  # add a blank line
        for tag, traj in sorted(losses.items()):
            msg = f'{traj[-1]:.4e}'
            if tag == output_symbol:
                tag = f'{tag} (Key loss)'
            msg = f'- {tag}: [{msg}]'
            feedbacks.append(msg)
    return feedbacks

class SGASearcher(BaseSearcher):
    def __init__(self, 
                 name, 
                 root,
                 path, 
                 python_path,
                 dataset_name,
                 dataset_path,
                 llm_model,
                 llm_api_url,
                 llm_api_key,
                 ) -> None:
        super().__init__(name)
        cfg = sga.config.DefaultConfig(path=path, 
                                       dataset_name=dataset_name,
                                       dataset_path=dataset_path,
                                       overwrite=True)
        cfg.llm.entry = "sr"
        cfg.llm.api_key = llm_api_key
        cfg.llm.model = llm_model
        cfg.llm.api_url = llm_api_url
        
        train_py_path = root / 'entry' / 'sr_train.py'
        eval_py_path = root / 'entry' / 'sr_eval.py'
        self._base_train_cmds = [python_path, train_py_path]
        self._base_eval_cmds = [python_path, eval_py_path]
        
        self.cfg = cfg

    def discover(self, task: SEDTask) -> List[Equation]:
        my_env = os.environ.copy()
        cfg = self.cfg
        unknown_args = ['--overwrite', '1']

        tpos = cfg.tpos
        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        assert Path(cfg.path).is_absolute()
        exp_root = Path(cfg.path) / task.name

        primitive_root = exp_root / 'primitive'
        offspring_root = exp_root / 'offspring'
        iteration_root = exp_root / 'iteration'
        sga.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume, verbose=True)
        primitive_root.mkdir(parents=True, exist_ok=True)
        offspring_root.mkdir(parents=True, exist_ok=True)
        iteration_root.mkdir(parents=True, exist_ok=True)

        cfg_dict = dataclasses.asdict(cfg)
        yaml.safe_dump(cfg_dict, (exp_root / 'config.yaml').open('w'))

        dataset_feedback = ''
        physicist = SRPhysicist(cfg.llm, 
                            seed=seed, 
                            task_info=task,
                            env_info=dataset_feedback,)
        population = Population(cfg.llm)

        primitive_code = primitive_root / "primitive_code.py"
        primitive_code.write_text(physicist.primitive_code, 'utf-8')

        cfg.llm.primitives = ['linear']
        exp_name = task.name
        for i_ind, ind_physics in enumerate(tqdm(cfg.llm.primitives, 
                                                 desc=f'[primitive] {exp_name}', 
                                                 file=sys.stdout, 
                                                 position=tpos)):
            ind_root = primitive_root / f'{i_ind:04d}'
            ind_root.mkdir(parents=True, exist_ok=True)
            

            # train
            train_args = {
                'tpos': tpos + 1,
                'path': ind_root,
                'dataset_name': cfg.dataset_name,
                'dataset_path': cfg.dataset_path,
                'problem_name': task.name,
                # 'physics.env.physics': ind_physics
                'physics.env.physics.path': primitive_code,
            }
            error = sga.utils.run_exp(self._base_train_cmds, 
                                      train_args, 
                                      unknown_args, 
                                      my_env)

            losses, params = sga.utils.parse_tensorboard(ind_root)
            print(list(losses.keys()))
            states = None
            output_symbol = task.symbols[0].lower()
            fitness = min(losses[output_symbol], key=lambda x: x if not math.isnan(x) else float('inf'))

            feedbacks = []
            feedbacks += get_perf_feedback(losses, params, output_symbol=output_symbol)
            feedback = '\n'.join(feedbacks)
            code_path = ind_root / 'physics.py'
        population.add_primitive(code_path, feedback, fitness, losses, params, states, ind_root)
        

        for i_iter in trange(cfg.llm.num_iters + 1, desc=f'[iteration] {exp_name}', file=sys.stdout, position=tpos):
            iter_ind_root = offspring_root / f'{i_iter:04d}'
            iter_ind_root.mkdir(parents=True, exist_ok=True)

            iter_root = iteration_root / f'{i_iter:04d}'
            iter_root.mkdir(parents=True, exist_ok=True)

            indices = population.sample(iter_root)
            msgs = physicist.get_msgs(population, indices, iter_root / 'messages')
            if i_iter == cfg.llm.num_iters:
                break
            response = physicist.generate(msgs, iter_root / 'choices', iter_root)

            for i_ind, ind_choice in enumerate(tqdm(response.choices, desc=f'[offspring] {exp_name}', file=sys.stdout, position=tpos + 1)):

                try:
                    ind_root = iter_ind_root / f'{i_ind:04d}'
                    ind_root.mkdir(parents=True, exist_ok=True)

                    code_path = ind_choice.dump_root / 'code.py'

                    if len(ind_choice.code) == 0:
                        raise RuntimeError('No code generated or generated solution violated format requirements.')

                    # train
                    train_args = {
                        'tpos': tpos + 2,
                        'path': ind_root,
                        'dataset_name': cfg.dataset_name,
                        'dataset_path': cfg.dataset_path,
                        'problem_name': task.name,
                        'physics.env.physics.path': code_path
                    }
                    error = sga.utils.run_exp(self._base_train_cmds, train_args, unknown_args, my_env)
                    print(error)
                    # breakpoint()
                    if len(error) > 0 and "error" in str(error).lower():
                        (ind_root / "error.txt").write_text(str(error), 'utf-8')
                        raise RuntimeError(error.rsplit('\n', maxsplit=1)[-1])
                    losses, params = sga.utils.parse_tensorboard(ind_root)
                    states = None
                    output_symbol = task.symbols[0].lower()
                    fitness = min(losses[output_symbol], key=lambda x: x if not math.isnan(x) else float('inf'))
                    print("Off fit:", fitness)
                    feedbacks = []

                    # if cfg.llm.name.startswith('openai-gpt-4'):
                    #     # evaluate and render
                    #     for eval_key in ['final']:
                    #         eval_args = {
                    #             'is_dataset': False,
                    #             'tpos': tpos + 2,
                    #             'path': ind_root / 'eval' / eval_key,
                    #             'physics.env.physics.path': code_path,
                    #             'ckpt_path': ind_root / 'ckpt' / f'{eval_key}.pt'
                    #         }
                    #         error = sga.utils.run_exp(base_eval_cmds, eval_args, unknown_args, my_env)
                    #     states = torch.load(ind_root / 'eval' / eval_key / 'state' / 'ckpt.pt', map_location='cpu')
                    #     feedbacks += get_state_feedback(states, cfg.llm.state_size)
                    #     feedbacks.append('')  # add a blank line

                    feedbacks += get_perf_feedback(losses, params, output_symbol=output_symbol)
                    feedback = '\n'.join(feedbacks)
                    print(feedback)
                    population.add_offspring(ind_choice, feedback, fitness, losses, params, states, ind_root)
                except Exception as e:
                    feedback = str(e)
                    fitness = float('inf')
                    losses = None
                    params = None
                    states = None
                    population.add_offspring(ind_choice, feedback, fitness, losses, params, states, ind_root)
        
        return [
            # SearchResult(
            #     equation=best_equation,
            #     aux={"best_program_sample_order": profiler._cur_best_program_sample_order, "best_program_score": profiler._cur_best_program_score},
            # )
        ]
    
    def _get_best_program(self, exp_root):
        ranking_path = exp_root / "iteration" / "0005" / "all.json"
        with open(exp_root) as f:
            ind_root = Path(json.load(f)[0]['root'])
        assert(ind_root.exists())