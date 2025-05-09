import os
from pathlib import Path

def get_conda_sh_path():
    try:
        # Get the CONDA_EXE environment variable
        conda_exe = os.environ.get('CONDA_EXE')
        
        if conda_exe is None:
            raise ValueError("CONDA_EXE environment variable is not set")
        
        conda_dir = os.path.dirname(conda_exe)
        conda_base = os.path.dirname(conda_dir)
        
        # path to conda.sh
        conda_sh_path = os.path.join(conda_base, 'etc', 'profile.d', 'conda.sh')
        if not os.path.exists(conda_sh_path):
            raise FileNotFoundError(f"conda.sh not found at {conda_sh_path}")
        
        return Path(conda_sh_path)
    
    except Exception as e:
        print(f"Error: {e}")
        return None
