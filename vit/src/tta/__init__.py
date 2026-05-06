"""
TTA (Test-Time Adaptation) module for VPT.

Implements Koopman-Spectral Controller (C1), Hankel-Balanced Update
Operator (C2), and KDMD semantic anchors as a plug-in wrapper for
test-time prompt adaptation.
"""

from .koopman_controller import KoopmanController
from .hbuo import HBUO
from .tta_engine import TTAEngine
from .state_utils import PromptStateManager
from .kdmd import KernelLifter, ClassKoopmanAccumulator, KDMDPredictor
from .ktmv import KoopmanTrajectoryMultiView
from .prototypes import OnlinePrototypeBank
from .prompt_adapter import PromptParameterAdapter, PromptSnapshot
from .update_subspace import ProjectionState, UpdateSubspaceManager
from .akc_controller import (
    AKCDecision,
    AdaptiveKoopmanController,
    DiagonalKoopmanFit,
    FullKoopmanFit,
)
from .hur_controller import HankelUpdateRouter, HURSplit, RoutedUpdate
from .tpd_runtime import RuntimeCheckpoint, TPDStepResult, TPDRuntime
from .tpd_engine import TPDTTAEngine
