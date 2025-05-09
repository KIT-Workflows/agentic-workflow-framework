from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml, os, json

from ..interfaces.conda_path import get_conda_sh_path
from ..core.qr_doc_engine import tot_corpus_raw

current_path = os.path.dirname(os.path.abspath(__file__))

conda_path = get_conda_sh_path()
if conda_path:
    CONDA_PATH = conda_path.as_posix()
    use_slurm = False
else:
    CONDA_PATH = None
    print(f"conda path not found, assuming the slurm submission is active")
    use_slurm = True

default_qe_settings = {
    'use_slurm': use_slurm,
    'qe_env': 'qe',
    'n_cores': 8,
    'conda_path': CONDA_PATH,
    'activate_command': "source {CONDA_PATH} && conda activate {QE_ENV} && cd {MAIN_DIR} && echo $PWD",
    'qe_prefix': "export OMP_NUM_THREADS={n_cores} && mpirun -n {n_cores}",
    'module_name': 'qe',
    'sbatch_script_template': Path(f'{current_path}/qe_slurm_template.sh').read_text() 
}

@dataclass
class ProjectConfig:
    """Configuration for Quantum Espresso Project"""
    name: str = field(default='my_project')
    main_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    pseudopotentials: Path = field(default_factory=Path)
    project_signature: str = field(default='')
    participants: Optional[List[str]] = field(default_factory=list)
    metadata: Optional[Dict] = field(default_factory=dict)
    main_documentation: List[Dict] = field(default_factory=lambda: tot_corpus_raw, repr=False)
    MAXDocToken: int = 9000
    kw_model_name: str = 'mistralai/mixtral-8x22b-instruct'
    chat_model_name: str = 'meta-llama/llama-3.3-70b-instruct'
    formulas: List[Dict] = field(default_factory=list)
    num_retries_permodel: int = 3
    async_batch_size: int = 10
    sleep_time: int = 30
    gen_model_hierarchy: List[str] = field(default_factory=lambda: ['dbrx', 'meta405o', 'nous405o'])
    model_config: Dict = field(default_factory=lambda: {
        'parameter_evaluation': 'meta405o',
        'condition_finder': 'meta405o'
    })
    lm_api: str = field(default_factory=str, repr=False)
    qe_settings: Dict = field(default_factory=lambda: default_qe_settings)
    history: Dict = field(default_factory=dict)
        
    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ProjectConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
            
        # Convert string paths to Path objects
        for key in ['main_dir', 'output_dir', 'pseudopotentials']:
            if key in data:
                data[key] = Path(data[key])
                
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = dict(sorted(asdict(self).items(), 
                         key=lambda x: [*self.__dataclass_fields__].index(x[0])))
        
        # Convert Path objects to strings
        for key in ['main_dir', 'output_dir', 'pseudopotentials']:
            if key in data and data[key] is not None:
                data[key] = str(data[key])
        
        # remove main_documentation and lm_api from data
        data.pop('main_documentation', None)
        data.pop('lm_api', None)
        
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def add_to_history(self, key: str, value: Any) -> None:
        """Add any key-value pair to history.
        
        Args:
            key: Any string key to identify this history entry
            value: Any value to store
        """
        self.history[key] = value