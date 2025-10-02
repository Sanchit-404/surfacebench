# In bench/surfacebench_datamodule.py

from typing import List
from pathlib import Path
import numpy as np
import h5py
from huggingface_hub import hf_hub_download
from .data_classes import Equation, SynProblem 

REPO_ID = "shobhnik/surfacebench"
HDF5_FILENAME = "surface_bench_data.hdf5"

def _download(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

class SurfaceBenchDataModule:
    def __init__(self, category_name: str):
        self.category_name = category_name
        self.problems: List[SynProblem] = []
        self.name2id = {}

    def setup(self):
        dataset_path = Path(_download(repo_id=REPO_ID, filename=HDF5_FILENAME))
        
        with h5py.File(dataset_path, "r") as hf:
            category_group = hf[self.category_name]
            
            for eq_name in category_group.keys():
                equation_group = category_group[eq_name]
                
                samples = {k: v[...].astype(np.float64) for k, v in equation_group.items()}
                
                if 'test_data' in samples:
                    samples['id_test_data'] = samples.pop('test_data')
                
                if 'ood_test' in samples:
                    samples['ood_test_data'] = samples.pop('ood_test')

                gt_equation = Equation(
                    symbols=['z', 'x', 'y'],
                    expression=f"Ground truth for {eq_name}",
                    symbol_descs=["Surface height", "Input variable x", "Input variable y"],
                    symbol_properties=['O', 'V', 'V']
                )
                
                problem = SynProblem(
                    dataset_identifier=self.category_name,
                    equation_idx=eq_name,
                    gt_equation=gt_equation,
                    samples=samples,
                )
                self.problems.append(problem)

        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}
        print(f"✅ Loaded {len(self.problems)} problems for category '{self.category_name}'.")

    @property
    def name(self):
        return self.category_name

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.problems[self.name2id[key]]
        return self.problems[key]
    
    def __len__(self):
        return len(self.problems)


def get_all_datamodules() -> List[SurfaceBenchDataModule]:
    """
    Automatically discovers all categories in the HDF5 file and returns a list
    of datamodules, one for each category.
    """
    datamodules = []
    dataset_path = Path(_download(repo_id=REPO_ID, filename=HDF5_FILENAME))
    with h5py.File(dataset_path, 'r') as hf:
        all_categories = list(hf.keys())
        print(f"Discovered categories: {all_categories}")
    
    for category_name in all_categories:
        dm = SurfaceBenchDataModule(category_name=category_name)
        datamodules.append(dm)
        
    return datamodules