# File: bench/datamodules.py
import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Any
from huggingface_hub import hf_hub_download

from data_classes import Equation, Problem, SEDTask

SURFACEBENCH_REPO_ID = "Shobhnik/SurfaceBench"
SURFACEBENCH_HDF5_FILENAME = "dataset.h5"

def _download(repo_id, filename):
    """Downloads a specific file from a Hugging Face Hub repo."""
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

class ExplicitProblem(Problem):
    """Problem class for explicit surfaces (x, y) -> z."""
    @property
    def train_samples(self): return self.samples.get("train_data")
    @property
    def test_samples(self): return self.samples.get("test_data")
    @property
    def ood_test_samples(self): return self.samples.get("ood_test")

class ParametricProblem(Problem):
    """Problem class for parametric surfaces (u, v) -> (x, y, z)."""
    @property
    def train_samples(self): return self.samples.get("train_data")
    @property
    def test_samples(self): return self.samples.get("test_data_eval")
    @property
    def ood_test_samples(self): return self.samples.get("ood_test_eval")
    def create_task(self) -> SEDTask:
        return SEDTask(
            name=self.equation_idx,
            symbols=self.gt_equation.symbols,
            symbol_descs=self.gt_equation.symbol_descs,
            symbol_properties=self.gt_equation.symbol_properties,
            samples=self.train_samples,
            desc=self.gt_equation.desc
        )

class SurfaceBenchDataModule:
    """A DataModule for loading any category from the SurfaceBench HDF5 file."""
    def __init__(self, category_name: str, hdf5_path: Optional[str] = None):
        self.category_name = category_name
        self.problems: List[Problem] = []

    def setup(self):
        dataset_path = Path(_download(repo_id=SURFACEBENCH_REPO_ID, filename=SURFACEBENCH_HDF5_FILENAME))
        
        with h5py.File(dataset_path, "r") as hf:
            if self.category_name == 'Parametric_Equations' and self.category_name in hf:
                category_group = hf[self.category_name]
                for eq_name in category_group.keys():
                    equation_group = category_group[eq_name]
                    train_data = {k: v[...].astype(np.float64) for k, v in equation_group['train_data'].items()}
                    samples = {
                        "train_data": train_data,
                        "test_data_eval": equation_group['test_data_eval'][...].astype(np.float64) if 'test_data_eval' in equation_group else None,
                        "ood_test_eval": equation_group['ood_test_eval'][...].astype(np.float64) if 'ood_test_eval' in equation_group else None
                    }                    
                    gt_equation = Equation(
                        symbols=['u', 'v', 'x', 'y', 'z'],
                        symbol_descs=["parameter u", "parameter v", "x coordinate", "y coordinate", "z coordinate"],
                        symbol_properties=['V', 'V', 'O', 'O', 'O'],
                        expression=f"[x,y,z] = f(u,v) for {eq_name}"
                    )
                    
                    problem_instance = ParametricProblem(
                        dataset_identifier=self.category_name,
                        equation_idx=eq_name,
                        gt_equation=gt_equation,
                        samples=samples,
                        problem_type="parametric"
                    )
                    self.problems.append(problem_instance)
            elif self.category_name in hf:
                category_group = hf[self.category_name]
                for eq_name in category_group.keys():
                    equation_group = category_group[eq_name]
                    samples = {k: v[...].astype(np.float64) for k, v in equation_group.items()}
                    sample_data = samples.get("train_data", samples.get("test_data"))
                    num_columns = sample_data.shape[1] if sample_data is not None else 0
                    
                    if num_columns == 3:
                        gt_equation = Equation(
                            symbols=['z', 'x', 'y'],
                            symbol_descs=["Surface height", "Input variable x", "Input variable y"],
                            symbol_properties=['O', 'V', 'V'],
                            expression=f"z = f(x, y) for {eq_name}"
                        )
                        problem_instance = ExplicitProblem(
                            dataset_identifier=self.category_name,
                            equation_idx=eq_name,
                            gt_equation=gt_equation,
                            samples=samples,
                            problem_type="explicit"
                        )
                        self.problems.append(problem_instance)
                    else:
                        continue
            else:
                raise ValueError(f"Category '{self.category_name}' not found in HDF5 file.")
        
        print(f"✅ Loaded {len(self.problems)} problems for category '{self.category_name}'.")
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}
    
    def name(self):
        return self.name

    def __getitem__(self, key):
        return self.problems[key]
    
    def __len__(self):
        return len(self.problems)

SURFACEBENCH_CATEGORIES = [
                           "Algebraic_Manifolds_of_Higher_Degree",
                           "Bio-Inspired_Morphological_Surfaces",
                           "Complex_Composite_Surfaces",
                           "Discrete_Symbolic_Grid_Surfaces",
                           "Hybrid_Multi-Modal_Symbolic_Surfaces",
                           "Non-Canonical_3D_Geometric_Surfaces",
                           "Piecewise_Regime_Surfaces",
                           "Procedural_Fractal_Surfaces",
                           "Symbolic-Numeric_Composite_Surfaces",
                           "Tensor_Field_Surfaces",
                           "Quantum_Inspired_Surfaces",
                           "Nonlinear_Dynamical_System_Surfaces",
                           "Surrogate_Distilled_Symbolic_Approximations",
                           "Stochastic_Process_Surfaces",
                           "Parametric_Equations"
                        ]

def get_datamodule(name, hdf5_path=None):
    if name == 'Parametric_Equations':
        return SurfaceBenchDataModule(category_name=name, hdf5_path=hdf5_path)
    elif name in SURFACEBENCH_CATEGORIES:
        return SurfaceBenchDataModule(category_name=name)
    else:
        raise ValueError(f"Unknown datamodule name: {name}")