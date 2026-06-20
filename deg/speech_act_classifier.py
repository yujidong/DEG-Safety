#!/usr/bin/env python3
"""
Speech Act Classifier for Pragmatics Layer

Theoretical Basis:
- Questions: High epistemic entropy (information-seeking)
- Commands: Low epistemic entropy (action-executing)
- Statements: Medium entropy (fact-stating)

This classifier uses a multi-signal approach:
1. Linguistic patterns (question words, imperative verbs)
2. Action ontology (READ vs WRITE operations)
3. Syntactic structure (interrogative, declarative, imperative)
"""

import re
from typing import Dict, List, Optional, Tuple


class SpeechActClassifier:
    """
    Lightweight speech act classification for pragmatics layer.

    Uses rule-based patterns and action ontology for fast, deterministic
    classification without requiring additional ML models.
    """

    # Question words (interrogative markers)
    QUESTION_WORDS = {
        "what", "how", "why", "where", "when", "who", "which", "whose",
        "can", "could", "would", "should", "may", "might",
        "do", "does", "did", "will", "won't", "shall",
        "query", "show", "tell", "list", "display"  # Information-seeking verbs
    }

    # Imperative verbs (command markers)
    IMPERATIVE_VERBS = {
        "disable", "enable", "override", "bypass", "make", "create", "delete",
        "modify", "change", "update", "remove", "add", "set", "adjust",
        "increase", "decrease", "stop", "start", "restart", "kill", "terminate",
        "execute", "run", "launch", "initiate", "trigger", "activate",
        "deactivate", "suppress", "ignore", "skip", "block", "allow", "permit",
        "grant", "revoke", "deny", "restrict", "limit", "expand", "extend",
        "clear", "reset", "flush", "validate", "verify", "check", "test"
    }

    # READ action types (information-seeking)
    READ_ACTIONS = {
        "read", "view", "show", "display", "list", "get", "retrieve",
        "fetch", "query", "search", "find", "check", "verify", "validate",
        "monitor", "track", "log", "audit", "inspect", "examine", "analyze",
        "report", "export", "download", "access", "look up", "tell me",
        "what's", "where is", "how do"
    }

    # WRITE action types (modifying)
    WRITE_ACTIONS = {
        "write", "update", "delete", "create", "modify", "change", "remove",
        "disable", "enable", "override", "bypass", "adjust", "configure",
        "set", "reset", "clear", "flush", "invalidate", "stop", "start",
        "restart", "kill", "block", "allow", "grant", "revoke", "restrict"
    }

    def __init__(self):
        """Initialize the speech act classifier."""
        # Compile regex patterns for efficiency
        self._question_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in self.QUESTION_WORDS) + r')\b',
            re.IGNORECASE
        )
        self._imperative_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in self.IMPERATIVE_VERBS) + r')\b',
            re.IGNORECASE
        )
        # Declarative starters (statement indicators)
        self._declarative_pattern = re.compile(
            r'^(the|a|an|all|every|each|this|that|these|those)\s',
            re.IGNORECASE
        )

    def classify(
        self,
        query: str,
        intent: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Classify query as QUESTION, COMMAND, or STATEMENT.

        Args:
            query: The query text to classify
            intent: Optional intent dict from intent extractor (may contain action info)

        Returns:
            Dict with:
                - speech_act: "QUESTION", "COMMAND", or "STATEMENT"
                - confidence: float 0.0-1.0
                - signals: dict of individual signal scores
        """
        if not query or not isinstance(query, str):
            return {
                "speech_act": "UNKNOWN",
                "confidence": 0.0,
                "signals": {}
            }

        query_lower = query.lower().strip()

        # Extract signals
        signals = self._extract_signals(query_lower, intent)

        # Combine signals to determine speech act
        speech_act, confidence = self._combine_signals(signals)

        return {
            "speech_act": speech_act,
            "confidence": confidence,
            "signals": signals
        }

    def _extract_signals(self, query: str, intent: Optional[Dict]) -> Dict[str, float]:
        """
        Extract individual classification signals.

        Args:
            query: Lowercase query text
            intent: Optional intent dict

        Returns:
            Dict of signal scores (0.0-1.0)
        """
        signals = {}

        # Signal 1: Linguistic pattern - question words
        has_question_word = bool(self._question_pattern.search(query))
        signals["question_word"] = 1.0 if has_question_word else 0.0

        # Signal 2: Linguistic pattern - imperative verbs
        has_imperative_verb = bool(self._imperative_pattern.search(query))
        signals["imperative_verb"] = 1.0 if has_imperative_verb else 0.0

        # Signal 3: Syntactic structure - question mark
        has_question_mark = query.endswith('?')
        signals["question_mark"] = 1.0 if has_question_mark else 0.0

        # Signal 4: Action ontology - READ vs WRITE
        signals["action_type"] = self._extract_action_signal(query, intent)

        # Signal 5: First word analysis (imperative vs interrogative)
        signals["first_word"] = self._analyze_first_word(query)

        # Signal 6: Query length (shorter = more likely command/question)
        word_count = len(query.split())
        if word_count <= 5:
            signals["length"] = 0.3  # Short queries often commands
        elif word_count <= 10:
            signals["length"] = 0.5
        else:
            signals["length"] = 0.7  # Longer queries often statements

        # Signal 7: Declarative pattern (starts with "The", "All", etc.)
        has_declarative_start = bool(self._declarative_pattern.match(query))
        signals["declarative_start"] = 1.0 if has_declarative_start else 0.0

        return signals

    def _extract_action_signal(self, query: str, intent: Optional[Dict]) -> float:
        """
        Extract action type signal from query or intent.

        Returns:
            0.0 for WRITE actions (command-biased)
            0.5 for UNKNOWN/neutral
            1.0 for READ actions (question-biased)
        """
        # First check intent if available
        if intent:
            action = intent.get("action", "").lower()

            # Check action type field
            action_type = intent.get("action_type", "").lower()
            if action_type == "read":
                return 1.0  # READ -> question-biased
            elif action_type == "write":
                return 0.0  # WRITE -> command-biased

            # Check action name against known READ/WRITE actions
            if any(read_action in action for read_action in self.READ_ACTIONS):
                return 1.0
            if any(write_action in action for write_action in self.WRITE_ACTIONS):
                return 0.0

        # Fall back to analyzing query text for action keywords
        for read_action in self.READ_ACTIONS:
            if read_action in query:
                return 1.0

        for write_action in self.WRITE_ACTIONS:
            if write_action in query:
                return 0.0

        # Neutral
        return 0.5

    def _analyze_first_word(self, query: str) -> float:
        """
        Analyze first word for syntactic structure.

        Returns:
            0.0 for imperative (command-biased)
            0.5 for neutral
            1.0 for interrogative (question-biased)
        """
        words = query.split()
        if not words:
            return 0.5

        first_word = words[0].lower()

        # Question words at start -> interrogative
        if first_word in self.QUESTION_WORDS:
            return 1.0

        # Imperative verbs at start -> imperative
        if first_word in self.IMPERATIVE_VERBS:
            return 0.0

        # Neutral
        return 0.5

    def _combine_signals(self, signals: Dict[str, float]) -> Tuple[str, float]:
        """
        Combine signals to determine final speech act.

        Uses weighted voting with bias reduction:
        - question_word: 0.30 weight
        - question_mark: 0.25 weight
        - imperative_verb: 0.30 weight
        - action_type: 0.15 weight (higher bias for commands)
        - first_word: 0.20 weight
        - Base statement score: 0.10 (only if no strong signals)

        Returns:
            Tuple of (speech_act_label, confidence)
        """
        # Weights for each signal (increased to reduce statement bias)
        weights = {
            "question_word": 0.30,
            "question_mark": 0.25,
            "imperative_verb": 0.30,
            "action_type": 0.20,
            "first_word": 0.15
        }

        # Check if action_type signal is strong (intent information available)
        has_strong_intent = signals["action_type"] in [0.0, 1.0]  # Clear READ or WRITE

        # Calculate question score (higher = more question-like)
        question_score = (
            signals["question_word"] * weights["question_word"] +
            signals["question_mark"] * weights["question_mark"] +
            signals["action_type"] * weights["action_type"] * (0.5 if not has_strong_intent else 1.5) +  # READ actions suggest questions, boost if strong intent
            (signals["first_word"] * weights["first_word"] if signals["first_word"] >= 0.7 else 0)
        )

        # Calculate command score (higher = more command-like)
        command_score = (
            signals["imperative_verb"] * weights["imperative_verb"] +
            (1.0 - signals["action_type"]) * weights["action_type"] * (0.8 if not has_strong_intent else 1.5) +  # WRITE actions strongly suggest commands, boost if strong intent
            ((1.0 - signals["first_word"]) * weights["first_word"] if signals["first_word"] <= 0.3 else 0) +
            (0.2 if signals["first_word"] == 0.0 else 0)  # Bonus for clear imperative first word
        )

        # Statement score (base level, increased only if no clear signals)
        # Statements have no question marks, no question words, no imperative verbs
        has_question_signals = signals["question_word"] > 0 or signals["question_mark"] > 0
        has_command_signals = signals["imperative_verb"] > 0
        has_declarative_signal = signals.get("declarative_start", 0) > 0

        if not has_question_signals and not has_command_signals:
            # No clear signals -> likely statement
            statement_score = 0.6
            if has_declarative_signal:
                statement_score = 0.85  # Strong statement indicator
            # Reduce statement score if we have strong intent information
            if has_strong_intent:
                statement_score = 0.2  # Intent should override default statement assumption
        else:
            # Has some signals -> lower statement probability
            statement_score = 0.1
            if has_declarative_signal:
                statement_score = 0.3  # Weaken other signals for declarative sentences

        # Normalize scores to sum to 1.0
        total = question_score + command_score + statement_score
        if total > 0:
            question_score /= total
            command_score /= total
            statement_score /= total

        # Determine speech act
        scores = {
            "QUESTION": question_score,
            "COMMAND": command_score,
            "STATEMENT": statement_score
        }

        speech_act = max(scores, key=scores.get)
        confidence = scores[speech_act]

        # Boost confidence for clear cases
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin > 0.3:  # Clear winner
                confidence = min(1.0, confidence * 1.5)

        # Clamp confidence to [0.1, 1.0] (min confidence 0.1)
        confidence = max(0.1, min(1.0, confidence))

        return speech_act, confidence

    def batch_classify(
        self,
        queries: List[str],
        intents: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Classify multiple queries.

        Args:
            queries: List of query texts
            intents: Optional list of intent dicts (same length as queries)

        Returns:
            List of classification results
        """
        if intents is None:
            intents = [None] * len(queries)

        return [
            self.classify(query, intent)
            for query, intent in zip(queries, intents)
        ]


# Convenience function for quick classification
def classify_speech_act(query: str, intent: Optional[Dict] = None) -> str:
    """
    Quick speech act classification (returns label only).

    Args:
        query: Query text to classify
        intent: Optional intent dict

    Returns:
        "QUESTION", "COMMAND", or "STATEMENT"
    """
    classifier = SpeechActClassifier()
    result = classifier.classify(query, intent)
    return result["speech_act"]
