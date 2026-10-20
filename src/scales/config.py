"""
Machine-specific configuration and the output folder layout.

External binaries
    `glpsol` (GLPK, needed to solve the LP) and `neato` (Graphviz, needed for `SupplyChain.draw_dynamic`)
    are located in this order:
        1. a path set explicitly with `set_glpsol()` / `set_neato()`,
        2. the environment variables `SCALES_GLPSOL` / `SCALES_NEATO`,
        3. the system PATH (`shutil.which`).

Output folders
    Every stage of the pipeline writes its default output into its own folder below the output root,
    so a user can stop after any stage and still find the intermediate results:

        <root>/networks/backbones/          SupplyChainBackbone.draw()                     (stage 1)
        <root>/supply_chains/               ScbConverter.to_json()                         (stage 2)
        <root>/networks/supply_chains/      ScbConverter.draw(), SupplyChain.draw_static() (stage 2)
        <root>/networks/diagnostics/        node diagnostic drawings (static and dynamic)
        <root>/networks/dynamic/            SupplyChain.draw_dynamic()                     (stage 3)
        <root>/results/lp/                  AMPL .dat files kept with `save_temp`          (stage 3)
        <root>/results/diagnostics/         CSV tables and LP constraint reports           (stage 3)
        <root>/results/perturbation/        SupplyChainAnalyser data and plots             (stage 4)
        <root>/temp/                        scratch files used while solving

    The root is the current working directory, unless `SCALES_OUTPUT_DIR` or `set_output_root()` says otherwise.
    Explicit filenames passed to any method are always used as given.
"""

import os
import shutil
from pathlib import Path

_glpsol = None
_neato = None
_output_root = None

LP_MODEL_PATH = Path(__file__).parent / "model" / "lp" / "supplychain-lp-model.mod"


# external binaries
def set_glpsol(path: str) -> None:
    """Sets the path to the `glpsol` binary for this session."""
    global _glpsol
    _glpsol = path


def set_neato(path: str) -> None:
    """Sets the path to the `neato` binary for this session."""
    global _neato
    _neato = path


def glpsol() -> str:
    """Returns the path to `glpsol`, or raises a RuntimeError explaining how to configure it."""
    return _find_binary("glpsol", _glpsol, "SCALES_GLPSOL", "set_glpsol",
                        "Install GLPK (e.g. `brew install glpk` or `apt-get install glpk-utils`)")


def neato() -> str:
    """Returns the path to `neato`, or raises a RuntimeError explaining how to configure it."""
    return _find_binary("neato", _neato, "SCALES_NEATO", "set_neato",
                        "Install Graphviz (e.g. `brew install graphviz` or `apt-get install graphviz`)")


def has_glpsol() -> bool:
    """Returns True if `glpsol` can be located."""
    try:
        glpsol()
        return True
    except RuntimeError:
        return False


def _find_binary(name: str, explicit: str, env_var: str, setter: str, install_hint: str) -> str:
    candidate = explicit or os.environ.get(env_var) or shutil.which(name)
    if candidate and os.path.isfile(candidate):
        return candidate
    raise RuntimeError(
        f"Could not find the `{name}` binary. {install_hint}, or point SCALeS to it with the "
        f"environment variable {env_var} or `scales.config.{setter}(path)`."
    )


# output folders
def set_output_root(path) -> None:
    """Sets the directory below which all default output folders are created."""
    global _output_root
    _output_root = Path(path)


def output_root() -> Path:
    if _output_root is not None:
        return _output_root
    return Path(os.environ.get("SCALES_OUTPUT_DIR", os.getcwd()))


def _dir(*parts: str) -> Path:
    path = output_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def networks_dir(*sub: str) -> Path:
    """Folder for drawings of backbones and supply chain networks."""
    return _dir("networks", *sub)


def supply_chains_dir(*sub: str) -> Path:
    """Folder for the static supply chain JSON files produced by ScbConverter."""
    return _dir("supply_chains", *sub)


def results_dir(*sub: str) -> Path:
    """Folder for solver outputs, diagnostics and analysis results."""
    return _dir("results", *sub)


def temp_dir() -> Path:
    """Folder for scratch files used while solving."""
    return _dir("temp")
