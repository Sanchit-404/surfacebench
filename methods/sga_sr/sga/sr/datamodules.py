from typing import Optional, Any

import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from .dataclasses import Equation, Problem

import warnings


FEYNMAN_REPO_ID = "../../../datasets/orig-feyn"
TRANSFORMED_FEYNMAN_REPO_ID = "../../../datasets/lsr-transform-feyn"

def _download(repo_id):
    return snapshot_download(repo_id=repo_id, repo_type="dataset")

class FeynmanDataModule:
    def __init__(self):
        # self._dataset_dir = Path(_download(repo_id=FEYNMAN_REPO_ID))
        self._dataset_dir = FEYNMAN_REPO_ID
        self._dataset_identifier = 'feynman'
    
    def setup(self, description_path=None):
        if description_path is None:
            with open(self._dataset_dir / 'equations.jsonl', 'r') as f:
                equations = [json.loads(line) for line in f.readlines()]
        else:
            with open(description_path, 'r') as f:
                equations = [json.loads(line) for line in f.readlines()]
        sample_dir = self._dataset_dir / "samples"
        self.problems = [Problem(dataset_identifier=self._dataset_identifier,
                                 equation_idx = e['name'],
                                 gt_equation=Equation(
                                    symbols=e['symbols'],
                                    symbol_descs=e['symbol_descs'],
                                    symbol_properties=e['symbol_properties'],
                                    expression=e['expression'],
                                 ),
                                 samples=np.load(sample_dir / (e['name'] + ".npz"))
        ) for e in equations]
    
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return "feynman_100000"

class TransformedFeynmanDataModule:
    def __init__(self):
        # self._dataset_dir = Path(_download(repo_id=TRANSFORMED_FEYNMAN_REPO_ID))
        self._dataset_dir = TRANSFORMED_FEYNMAN_REPO_ID
        self._dataset_identifier = 'transformed_feynman'
    
    def setup(self, description_path=None):
        if description_path is None:
            with open(self._dataset_dir / 'equations.jsonl', 'r') as f:
                equations = [json.loads(line) for line in f.readlines()]
        else:
            with open(description_path, 'r') as f:
                equations = [json.loads(line) for line in f.readlines()]
        sample_dir = self._dataset_dir / "samples"
        self.problems = [Problem(dataset_identifier=self._dataset_identifier,
                                 equation_idx = e['name'],
                                 gt_equation=Equation(
                                    symbols=e['symbols'],
                                    symbol_descs=e['symbol_descs'],
                                    symbol_properties=e['symbol_properties'],
                                    expression=e['expression'],
                                 ),
                                 samples=np.load(sample_dir / (e['name'] + ".npz"))
        ) for e in equations]

        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return "invfeynman_100000"

class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples['train_data']
    
    @property
    def test_samples(self):
        return self.samples['id_test_data']
    
    @property
    def ood_test_samples(self):
        return self.samples['ood_test_data']

class BaseSynthDataModule:
    def __init__(self, dataset_identifier, short_dataset_identifier, root, default_symbols = None, default_symbol_descs=None):
        self._dataset_dir = Path(root)
        self._dataset_identifier = dataset_identifier
        self._short_dataset_identifier = short_dataset_identifier
        self._default_symbols = default_symbols
        self._default_symbol_descs = default_symbol_descs
    
    def setup(self):
        with open(self._dataset_dir / "processed_dataset.json", 'r') as f:
            equations = json.load(f)

        self.problems = []
        problems = []
        for equation_index, (_, info) in enumerate(equations.items()):
            desc = info['description']
            params = list(info['params'].keys())
            expression = info['equation']

            train_data = info['train_data']
            symbols = list(train_data.keys())
            # print(symbols)
            symbols = [symbols[-1]] + symbols[:-1] # Output symbol first

            symbol_descs = [desc.split("describing ")[1].split("(")[0].strip()]
            
            # print(desc)
            # print(symbols, expression)
            try:
                for i, d in enumerate(desc.split("given data on ")[1].split(')')):
                    if "(" not in d:
                        continue
                    else:
                        assert symbols[i + 1] in d, f"{symbols[i + 1]}, {d} and {desc}"
                        symbol_descs.append(d.split("(")[0].strip(",").strip())
                assert len(symbols) == len(symbol_descs)
            except AssertionError:
                # warnings.warn("Fallback to the default symbol descriptions")
                symbols = [s for s in symbols if s in self._default_symbols]
                symbol_descs = [self._default_symbol_descs[self._default_symbols.index(s)] for s in symbols if s in self._default_symbols]
            
            samples = {}
            for k in ["train_data", "id_test_data", "ood_test_data"]:
                data = [info[k][s] for s in symbols]
                data = np.stack(data, axis=1)
                samples[k] = data

            if np.any(np.isnan(samples['train_data'])):
                continue
            if np.any(np.isnan(samples['id_test_data'])):
                continue

            p = SynProblem(dataset_identifier=self._dataset_identifier,
                        equation_idx = self._short_dataset_identifier + str(equation_index),
                        gt_equation=Equation(
                        symbols=symbols,
                        symbol_descs=symbol_descs,
                        symbol_properties=['O',] + ['V'] * (len(symbols) - 1),
                        expression=expression,
                        desc=desc),
                samples=samples)

            problems.append(p)

        self.problems = problems

    
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return self._dataset_identifier

class MatSciDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("MatSci", "MatSci", root)

# class ChemReactKineticsDataModule(BaseSynthDataModule):
#     def __init__(self, root):
#         super().__init__("ChemReactKinetics", "CRK", root,
#                          default_symbols=['dA_dt', 't', 'A'],
#                          default_symbol_descs=['Rate of change of concentration', 'Time', 'Concentration at time t'])
        
# class BioPopGrowthDataModule(BaseSynthDataModule):
#     def __init__(self, root):
#         super().__init__("BioPopGrowth", "BPG", root,
#                          default_symbols=['dP_dt', 't', 'P'],
#                          default_symbol_descs=['Population growth rate', 'Time', 'Population at time t'])
        
# class PhysOscilDataModule(BaseSynthDataModule):
#     def __init__(self, root):
#         super().__init__("PhysOscillator", "PO", root,
#                          default_symbols=['dv_dt', 'x', 't', 'v'],
#                          default_symbol_descs=['Acceleration', 'Position at time t', 'Time', 'Velocity at time t'])

class ChemReactKineticsDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("ChemReactKinetics", "CRK", root,
                         default_symbols=['dA_dt', 't', 'A'],
                         default_symbol_descs=['Rate of change of concentration in chemistry reaction kinetics', 'Time', 'Concentration at time t'])
        
class BioPopGrowthDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("BioPopGrowth", "BPG", root,
                         default_symbols=['dP_dt', 't', 'P'],
                         default_symbol_descs=['Population growth rate', 'Time', 'Population at time t'])
        
class PhysOscilDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("PhysOscillator", "PO", root,
                         default_symbols=['dv_dt', 'x', 't', 'v'],
                         default_symbol_descs=['Acceleration in Nonl-linear Harmonic Oscillator', 'Position at time t', 'Time', 'Velocity at time t'])

def get_datamodule(root_folder):
    if 'bio' in root_folder:
        return BioPopGrowthDataModule(root_folder)
    elif 'chem' in root_folder:
        return ChemReactKineticsDataModule(root_folder)
    elif 'mat' in root_folder:
        return MatSciDataModule(root_folder)
    elif 'phys' in root_folder:
        return PhysOscilDataModule(root_folder)
    elif 'feynman' == root_folder:
        return FeynmanDataModule()
    elif 'invfeynman' == root_folder:
        return TransformedFeynmanDataModule()
    else:
        raise ValueError