from typing import List
from pathlib import Path
import json
import numpy as np
import time
from .data_classes import Problem, SearchResult
from .searchers.base import BaseSearcher
from scipy.spatial.distance import cdist

def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Calculates the Chamfer Distance between two point clouds."""
    if pc1 is None or pc2 is None or pc1.size == 0 or pc2.size == 0:
        return float('inf')
    if pc1.ndim == 1: pc1 = pc1.reshape(-1, 1)
    if pc2.ndim == 1: pc2 = pc2.reshape(-1, 1)
    
    dist_matrix = cdist(pc1, pc2)
    dist_pc1_to_pc2 = np.mean(np.min(dist_matrix, axis=1))
    dist_pc2_to_pc1 = np.mean(np.min(dist_matrix, axis=0))
    return float(dist_pc1_to_pc2 + dist_pc2_to_pc1)

def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Calculates the Hausdorff Distance between two point clouds."""
    if pc1 is None or pc2 is None or pc1.size == 0 or pc2.size == 0:
        return float('inf')
    if pc1.ndim == 1: pc1 = pc1.reshape(-1, 1)
    if pc2.ndim == 1: pc2 = pc2.reshape(-1, 1)
    
    dist_matrix = cdist(pc1, pc2)
    dist_pc1_to_pc2 = np.max(np.min(dist_matrix, axis=1))
    dist_pc2_to_pc1 = np.max(np.min(dist_matrix, axis=0))
    return float(max(dist_pc1_to_pc2, dist_pc2_to_pc1))

class EvaluationPipeline:
    def run_and_evaluate(self, searcher: BaseSearcher, problem: Problem):
        start_time = time.time()
        search_results: List[SearchResult] = searcher.discover(problem.create_task())
        search_time = time.time() - start_time

        outs = []
        for result in search_results:
            equation = result.equation
            test_metrics, ood_metrics = None, None
            if getattr(problem, 'problem_type', 'explicit') == 'explicit':
                if problem.test_samples is not None and problem.test_samples.size > 0:
                    X_test, y_test = problem.test_samples[:, :2], problem.test_samples[:, 2]
                    lambda_fn = equation.lambda_format
                    if callable(lambda_fn):
                        try:
                            y_pred = lambda_fn(X_test)
                        except TypeError:
                            y_pred = lambda_fn(*X_test.T)
                        pc_true = np.hstack([X_test, y_test.reshape(-1, 1)])
                        pc_pred = np.hstack([X_test, y_pred.reshape(-1, 1)])
                        test_metrics = {"chamfer": chamfer_distance(pc_true, pc_pred), "hausdorff": hausdorff_distance(pc_true, pc_pred), "mse": np.mean((y_test - y_pred)**2)}

                if problem.ood_test_samples is not None and problem.ood_test_samples.size > 0:
                    X_ood, y_ood = problem.ood_test_samples[:, :2], problem.ood_test_samples[:, 2]
                    lambda_fn = equation.lambda_format
                    if callable(lambda_fn):
                        try:
                            y_pred_ood = lambda_fn(X_ood)
                        except TypeError:
                            y_pred_ood = lambda_fn(*X_ood.T)
                        pc_true_ood = np.hstack([X_ood, y_ood.reshape(-1, 1)])
                        pc_pred_ood = np.hstack([X_ood, y_pred_ood.reshape(-1, 1)])
                        ood_metrics = {"chamfer": chamfer_distance(pc_true_ood, pc_pred_ood), "hausdorff": hausdorff_distance(pc_true_ood, pc_pred_ood), "mse": np.mean((y_ood - y_pred_ood)**2)}

            elif problem.problem_type == 'parametric':
                if problem.test_samples is not None and problem.test_samples.size > 0:
                    uv_test, xyz_true = problem.test_samples[:, :2], problem.test_samples[:, 2:]
                    x_fn, y_fn, z_fn = equation.lambda_format
                    if all(callable(fn) for fn in [x_fn, y_fn, z_fn]):
                        xyz_pred = np.stack([x_fn(uv_test), y_fn(uv_test), z_fn(uv_test)], axis=-1)
                        test_metrics = {"chamfer": chamfer_distance(xyz_true, xyz_pred), "hausdorff": hausdorff_distance(xyz_true, xyz_pred)}

                if problem.ood_test_samples is not None and problem.ood_test_samples.size > 0:
                    uv_ood, xyz_true_ood = problem.ood_test_samples[:, :2], problem.ood_test_samples[:, 2:]
                    x_fn, y_fn, z_fn = equation.lambda_format
                    if all(callable(fn) for fn in [x_fn, y_fn, z_fn]):
                        xyz_pred_ood = np.stack([x_fn(uv_ood), y_fn(uv_ood), z_fn(uv_ood)], axis=-1)
                        ood_metrics = {"chamfer": chamfer_distance(xyz_true_ood, xyz_pred_ood), "hausdorff": hausdorff_distance(xyz_true_ood, xyz_pred_ood)}

            outs.append({
                "search_result": result,
                "search_time": search_time,
                "id_metrics": test_metrics,
                "ood_metrics": ood_metrics,
            })
        return outs

    def evaluate_problems(self, problems: List[Problem], searcher: BaseSearcher, output_dir, result_file_subfix=""):
        output_dir = Path(output_dir)
        output_file_path = output_dir / f"results{result_file_subfix}.jsonl"
        visited_eqids = self.load_visited_problems(output_dir)

        for problem in problems:
            if problem.equation_idx in visited_eqids:
                print(f"Skipping problem: {problem.equation_idx}")
                continue

            print(f"Finding equation for problem: {problem.equation_idx} (gt: {problem.gt_equation.expression})")
            
            outs = self.run_and_evaluate(searcher, problem)

            log_data = {
                'equation_id': problem.equation_idx,
                'gt_equation': problem.gt_equation.expression,
                'num_datapoints': len(problem.train_samples) if problem.train_samples is not None else 0,
                'num_eval_datapoints': len(problem.test_samples) if problem.test_samples is not None else 0,
            }
            eval_results = []
            for out in outs:
                eq_str_format = getattr(out['search_result'].equation, 'expression', None)
                eq_program_format = getattr(out['search_result'].equation, 'program_format', None)
                aux_data = getattr(out['search_result'], 'aux', {})

                eval_results.append({
                    'search_time': out['search_time'],
                    'discovered_equation': eq_str_format,
                    'discovered_program': eq_program_format,
                    'id_metrics': out['id_metrics'],
                    'ood_metrics': out['ood_metrics'],
                    **aux_data
                })
            log_data['eval_results'] = eval_results
            with open(output_file_path, mode='a') as f:
                f.write(json.dumps(log_data) + "\n")

    def load_visited_problems(self, output_dir):
        visited = []
        for result_file in Path(output_dir).glob("results*.jsonl"):
            with open(result_file, 'r') as f:
                for line in f:
                    try:
                        visited.append(json.loads(line)['equation_id'])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return list(set(visited))