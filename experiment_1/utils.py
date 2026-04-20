from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


UTILS_PATH = Path(__file__).resolve().parent.parent / "utils.py"
SPEC = spec_from_file_location("_project_utils", UTILS_PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

for name in dir(MODULE):
    if not name.startswith("_"):
        globals()[name] = getattr(MODULE, name)

__all__ = [name for name in globals() if not name.startswith("_")]
