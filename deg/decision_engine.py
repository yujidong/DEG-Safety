"""
Phase IV: Decision Engine via Active Inference

Implements adaptive policy selection based on Expected Free Energy.
This is the final phase of the 4-phase DEG pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Action types with different risk profiles."""
    WRITE = "WRITE"  # Modifying system state (high risk)
    READ = "READ"    # Information retrieval (low risk)
    UNKNOWN = "UNKNOWN"


class Decision(Enum):
    """Policy decisions."""
    COLLAB = "COLLAB"      # Allow request
    DEFENSE = "DEFENSE"    # Block request


@dataclass
class PolicyDecision:
    """
    Decision output from the decision engine.

    Attributes:
        decision: COLLAB or DEFENSE
        vfe_score: Calculated free energy
        threshold: Applied threshold
        action_type: WRITE/READ classification
        violated_axiom: ID of most violated axiom (if any)
        reasoning: Human-readable explanation
    """
    decision: Decision
    vfe_score: float
    threshold: float
    action_type: Action
    violated_axiom: Optional[str]
    reasoning: str


class DecisionEngine(ABC):
    """
    Abstract base class for decision engines.

    Implements Active Inference: select policy π that minimizes
    Expected Free Energy E[F(z, π)].
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize decision engine.

        Args:
            config: Configuration dict with thresholds and lambda
        """
        self.thresholds = config.get("thresholds", {
            "WRITE": 3.0,
            "READ": 2.5,
            "default": 3.0
        })
        self.lambda_risk = config.get("lambda_risk", 2.0)
        self.action_multiplier = config.get("action_multiplier", 2.0)

    @abstractmethod
    def decide(
        self,
        vfe_score: float,
        action_type: Action,
        violated_axiom: Optional[str] = None
    ) -> PolicyDecision:
        """
        Make decision based on VFE and action type.

        Args:
            vfe_score: Calculated free energy
            action_type: WRITE/READ classification
            violated_axiom: Most violated axiom ID

        Returns:
            PolicyDecision with recommendation
        """
        pass


class AdaptiveDecisionEngine(DecisionEngine):
    """
    Adaptive decision engine with action-type-aware thresholds.

    WRITE actions face stricter thresholds than READ actions.
    This reflects the higher risk of system state modifications.
    """

    def decide(
        self,
        vfe_score: float,
        action_type: Action,
        violated_axiom: Optional[str] = None
    ) -> PolicyDecision:
        """
        Adaptive policy selection based on action type.

        Args:
            vfe_score: Calculated free energy
            action_type: WRITE/READ classification
            violated_axiom: Most violated axiom ID

        Returns:
            PolicyDecision with recommendation
        """
        # Select threshold based on action type
        if action_type == Action.WRITE:
            threshold = self.thresholds.get("WRITE", self.thresholds["default"])
        elif action_type == Action.READ:
            threshold = self.thresholds.get("READ", self.thresholds["default"])
        else:
            threshold = self.thresholds["default"]

        # Make decision (no multiplier - thresholds already account for action type)
        decision = Decision.DEFENSE if vfe_score > threshold else Decision.COLLAB

        # Generate reasoning
        reasoning = self._generate_reasoning(
            decision, vfe_score, threshold, action_type, violated_axiom
        )

        return PolicyDecision(
            decision=decision,
            vfe_score=vfe_score,
            threshold=threshold,
            action_type=action_type,
            violated_axiom=violated_axiom,
            reasoning=reasoning
        )

    def _generate_reasoning(
        self,
        decision: Decision,
        vfe_score: float,
        threshold: float,
        action_type: Union[Action, str],
        violated_axiom: Optional[str]
    ) -> str:
        """Generate human-readable reasoning."""
        # Handle both Action enum and string
        action_str = action_type.value if isinstance(action_type, Action) else action_type
        decision_str = decision.value if isinstance(decision, Decision) else decision

        lines = [
            f"Decision Analysis:",
            f"  Action Type: {action_str}",
            f"  VFE Score: {vfe_score:.4f}",
            f"  Applied Threshold: {threshold:.4f}",
            f"  Result: {decision_str}"
        ]

        if violated_axiom:
            lines.append(f"  Violated Axiom: {violated_axiom}")

        if decision == Decision.DEFENSE:
            lines.append("\n  BLOCKED: Free energy exceeds cognitive capacity threshold.")
        else:
            lines.append("\n  SAFE: Request within acceptable energy bounds.")

        return "\n".join(lines)


class SimpleDecisionEngine(DecisionEngine):
    """
    Simple decision engine with single unified threshold.

    Safety is determined by axioms, not action types.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simple decision engine with unified threshold.

        Args:
            config: Configuration dict with "threshold" key (unified for all actions)
        """
        # Support both "threshold" (unified) and "thresholds" (legacy format)
        if "threshold" in config:
            # New format: unified threshold
            self.threshold = config["threshold"]
            # For compatibility with parent class
            self.thresholds = {
                "WRITE": self.threshold,
                "READ": self.threshold,
                "default": self.threshold
            }
        else:
            # Legacy format: use parent's __init__
            super().__init__(config)
            self.threshold = self.thresholds.get("default", 0.7)

        self.lambda_risk = config.get("lambda_risk", 2.0)
        self.action_multiplier = config.get("action_multiplier", 2.0)

    def decide(
        self,
        vfe_score: float,
        action_type: Action,
        violated_axiom: Optional[str] = None
    ) -> PolicyDecision:
        """
        Simple threshold-based decision (UNIFIED for all actions).

        Args:
            vfe_score: Calculated free energy
            action_type: WRITE/READ classification (not used - unified threshold)
            violated_axiom: Most violated axiom ID

        Returns:
            PolicyDecision with recommendation
        """
        decision = Decision.DEFENSE if vfe_score > self.threshold else Decision.COLLAB

        reasoning = (
            f"Decision Analysis:\n"
            f"  Action Type: {action_type.value}\n"
            f"  VFE Score: {vfe_score:.4f}\n"
            f"  Unified Threshold: {self.threshold:.4f}\n"
            f"  Result: {decision.value}"
        )

        return PolicyDecision(
            decision=decision,
            vfe_score=vfe_score,
            threshold=self.threshold,
            action_type=action_type,
            violated_axiom=violated_axiom,
            reasoning=reasoning
        )
