"""Test-Time Prompt-Agnostic Decomposition (TPD)."""

from .akc import AKCResult, AdaptiveKoopmanControl
from .controller import TPD, TPDStepInfo
from .hur import HURDecomposition, HankelUpdateRouter, RoutedUpdate
from .prompt import PromptParameterAdapter, PromptSnapshot
from .subspace import ProjectedUpdateSubspace, SubspaceState

__all__ = [
    "TPD",
    "TPDStepInfo",
    "AdaptiveKoopmanControl",
    "AKCResult",
    "HankelUpdateRouter",
    "HURDecomposition",
    "RoutedUpdate",
    "PromptParameterAdapter",
    "PromptSnapshot",
    "ProjectedUpdateSubspace",
    "SubspaceState",
]

__version__ = "0.1.0"
