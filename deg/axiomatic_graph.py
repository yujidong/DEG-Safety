"""
Phase II: Grounding in the Immutable Axiomatic Graph

This module implements the external safety ontology that serves as
long-term normative memory, decoupled from the generative context.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, fields
import json


@dataclass
class Axiom:
    """
    A single safety axiom (prohibition or principle).

    Attributes:
        id: Unique identifier
        category: Safety category (e.g., "violence", "privacy", "Nuc-Fusion")
        text: The axiom text (was "prohibition", now accepts both)
        principles: List of guiding principles
        rigidity: Structural rigidity calculated from hierarchy (1 + log(1 + descendants))
        level: Hierarchy level (0=root, 1=operational)
        domain: Domain-specific category (e.g., "Nuc-Fusion")
        parents: List of parent axiom IDs for hierarchical structure
        speech_act_relevance: List of speech acts this axiom applies to (QUESTION/COMMAND/STATEMENT)
    """
    id: str
    category: str
    text: str
    principles: List[str]
    rigidity: float = 1.0  # Calculated from hierarchy, not from JSON
    level: int = 0
    domain: Optional[str] = None
    parents: Optional[List[str]] = None
    speech_act_relevance: Optional[List[str]] = None  # NEW: Speech act filtering

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "text": self.text,
            "principles": self.principles,
            "rigidity": self.rigidity,
            "level": self.level,
            "domain": self.domain,
            "parents": self.parents,
            "speech_act_relevance": self.speech_act_relevance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Axiom":
        """
        Create Axiom from dictionary with flexible field handling.

        Handles:
        - Both "text" and "prohibition" field names
        - Both "weight" and "rigidity" field names
        - Extra fields (like "comment") by ignoring them
        - Default values for missing optional fields
        """
        import copy

        # Work on a copy to avoid modifying original
        data = copy.deepcopy(data)

        # Handle field name mappings
        if "prohibition" in data and "text" not in data:
            data["text"] = data.pop("prohibition")
        if "weight" in data and "rigidity" not in data:
            data["rigidity"] = data.pop("weight")

        # Ensure required fields have defaults
        if "category" not in data:
            data["category"] = "general"
        if "principles" not in data:
            data["principles"] = []
        if "text" not in data and "comment" in data:
            # Use comment as fallback for text
            data["text"] = data.pop("comment")

        # Only keep fields that are in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


class AxiomaticGraph(ABC):
    """
    Abstract base class for the Axiomatic Graph.

    The Axiomatic Graph serves as an invariant reference frame for evaluating
    the Pragmatic Value term P(z|A) in the Free Energy functional.
    """

    def __init__(self, axioms: Optional[List[Axiom]] = None):
        """
        Initialize the axiomatic graph.

        Args:
            axioms: List of safety axioms
        """
        self.axioms = axioms or []
        self.nodes = {axiom.id: axiom for axiom in self.axioms}
        self._index()

    def get_markov_blanket(self, axiom_id: str) -> set:
        """
        Compute the Markov Blanket of a node: Parents + Children + Co-parents.

        Theoretical basis: In probabilistic graphical models, the Markov Blanket
        contains all variables that shield the node from the rest of the network.
        This enables conditional independence: P(intent | MB) = P(intent | MB, A\\\\MB)

        Args:
            axiom_id: The axiom node to compute MB for

        Returns:
            Set of axiom IDs in the Markov Blanket
        """
        mb = set()

        # Get the axiom node
        if axiom_id not in self.nodes:
            return mb

        axiom = self.nodes[axiom_id]

        # 1. Parents (direct dependencies)
        parents = axiom.parents or []
        mb.update(parents)

        # 2. Children (nodes that depend on this)
        # This requires building adjacency list first
        if hasattr(self, 'adj_list'):
            children = self.adj_list.get(axiom_id, [])
            mb.update(children)

            # 3. Co-parents (siblings: other children of my parents)
            for parent_id in parents:
                siblings = self.adj_list.get(parent_id, [])
                mb.update(siblings)

        # Remove self if present
        mb.discard(axiom_id)

        return mb

    def _index(self):
        """Build indexes for efficient retrieval."""
        self.by_category = {}
        for axiom in self.axioms:
            if axiom.category not in self.by_category:
                self.by_category[axiom.category] = []
            self.by_category[axiom.category].append(axiom)

    @abstractmethod
    def retrieve_relevant(self, intent: Dict[str, Any], top_k: int = 5) -> List[Axiom]:
        """
        Retrieve the Markov Blanket of relevant axioms for a given intent.

        This retrieval creates structural decoupling: safety criteria are
        no longer competing covariates in the generative context.

        Args:
            intent: Extracted intent dictionary
            top_k: Number of relevant axioms to retrieve

        Returns:
            List of relevant axioms
        """
        pass

    def add_axiom(self, axiom: Axiom):
        """Add a new axiom to the graph."""
        self.axioms.append(axiom)
        self._index()

    def load_from_file(self, filepath: str):
        """Load axioms from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both dict with "axioms" key and direct list
            if isinstance(data, dict) and "axioms" in data:
                items = data["axioms"]
            else:
                items = data
            self.axioms = [Axiom.from_dict(item) for item in items]
            self._index()

    def save_to_file(self, filepath: str):
        """Save axioms to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([a.to_dict() for a in self.axioms], f, indent=2)


class LocalAxiomaticGraph(AxiomaticGraph):
    """
    Local file-based axiomatic graph using vector similarity.

    Supports hierarchical structure with rigidity calculation based on
    descendant count: rigidity = 1 + log(1 + descendants)
    """

    def __init__(
        self,
        axioms: Optional[List[Axiom]] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the local axiomatic graph.

        Args:
            axioms: List of safety axioms (if None, uses default general safety axioms)
            embedding_model: Optional sentence-transformers model name OR pre-loaded model instance
        """
        # Use default general safety axioms if none provided
        if axioms is None:
            axioms = self._get_default_axioms()

        super().__init__(axioms)

        # Support both string (model name) and pre-loaded model instance
        if embedding_model is None:
            self.embedding_model_name = "all-MiniLM-L6-v2"
            self._embedding_model = None
            self._axiom_embeddings = None
        elif isinstance(embedding_model, str):
            self.embedding_model_name = embedding_model
            self._embedding_model = None
            self._axiom_embeddings = None
        else:
            # Pre-loaded model instance (for model sharing)
            self._embedding_model = embedding_model
            self.embedding_model_name = getattr(embedding_model, '_model_name', "shared_model")
            # Compute axiom embeddings immediately since model is already loaded
            self._axiom_embeddings = self._embedding_model.encode([a.text for a in self.axioms])

        # Hierarchy support
        self.adj_list = {}  # Parent -> children mapping
        self._hierarchy_built = False
        self.markov_blankets = {}  # Pre-computed Markov Blankets

        # Pre-compute Markov Blankets for default axioms
        # Note: For default axioms, we preserve manually set rigidity values
        # For axioms loaded from file, rigidity will be recalculated
        if axioms:
            self._build_hierarchy()
            # Don't auto-calculate rigidity for default axioms - preserve manual values
            # self._calculate_rigidity()
            self._precompute_markov_blankets()

    @staticmethod
    def _get_default_axioms() -> List[Axiom]:
        """
        Get default general safety axioms for common use cases.

        These cover the most common safety categories:
        - Violence and physical harm
        - Illegal activities (fraud, phishing, hacking, theft)
        - Privacy violations
        - Self-harm
        - Cybersecurity exploits
        """
        return [
            Axiom(
                id="root_safety",
                category="violence",
                text="Do not provide instructions for creating weapons, explosives, or harmful substances",
                principles=["preserve human safety", "prevent physical harm"],
                rigidity=2.0,  # High rigidity for root-level constraint
                level=0
            ),
            Axiom(
                id="illegal_acts",
                category="illegal",
                text="Do not assist with illegal activities including hacking, fraud, phishing, scamming, spoofing, social engineering, or theft",
                principles=["uphold legal standards", "prevent criminal activity"],
                rigidity=2.0,
                level=0
            ),
            Axiom(
                id="privacy",
                category="privacy",
                text="Do not provide methods for compromising privacy, accessing private data, or unauthorized surveillance",
                principles=["protect individual privacy", "prevent surveillance"],
                rigidity=1.5,
                level=0
            ),
            Axiom(
                id="self_harm",
                category="safety",
                text="Do not encourage or provide methods for self-harm or suicide",
                principles=["preserve life", "promote mental health"],
                rigidity=2.0,
                level=0
            ),
            Axiom(
                id="cybersecurity",
                category="security",
                text="Do not provide actionable exploitation instructions for vulnerabilities, bypass security controls, or unauthorized access",
                principles=["responsible disclosure", "defensive security only"],
                rigidity=1.0,
                level=1  # Operational level
            ),
        ]

    def _build_hierarchy(self):
        """Build parent-child dependency graph from axiom parents."""
        if self._hierarchy_built or not self.axioms:
            return

        # Initialize adjacency list
        self.adj_list = {axiom.id: [] for axiom in self.axioms}

        # Build edges from parents to children
        for axiom in self.axioms:
            if axiom.parents:
                for parent_id in axiom.parents:
                    if parent_id in self.adj_list:
                        self.adj_list[parent_id].append(axiom.id)

        self._hierarchy_built = True

    def _precompute_markov_blankets(self):
        """
        Pre-compute Markov Blankets for all axioms at initialization.

        Complexity: O(N) one-time cost
        Space: O(N × d_avg) where d_avg is average MB size
        """
        if not self.axioms:
            return

        # Build hierarchy first
        self._build_hierarchy()

        # Compute MB for each axiom
        for axiom in self.axioms:
            self.markov_blankets[axiom.id] = self.get_markov_blanket(axiom.id)

    def _get_descendants(self, node_id: str) -> set:
        """BFS to find all descendants in the DAG."""
        descendants = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            children = self.adj_list.get(current, [])
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return descendants

    def _calculate_rigidity(self):
        """Calculate rigidity based on hierarchical position."""
        if not self.axioms:
            return

        import numpy as np

        for axiom in self.axioms:
            desc_count = len(self._get_descendants(axiom.id))
            # Logarithmic scaling for smooth energy landscape
            # Formula: rigidity = 1 + log(1 + descendants)
            axiom.rigidity = 1.0 + np.log1p(desc_count)

    def load_from_file(self, filepath: str):
        """Load axioms from a JSON file with hierarchy support."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both formats: direct list or {"axioms": [...]}
        axioms_list = data if isinstance(data, list) else data.get("axioms", [])

        self.axioms = []
        for item in axioms_list:
            # from_dict handles field name normalization
            self.axioms.append(Axiom.from_dict(item))

        # Rebuild nodes dict
        self.nodes = {axiom.id: axiom for axiom in self.axioms}

        # Build hierarchy (for Markov Blanket calculation)
        # NOTE: We do NOT call _calculate_rigidity() here to preserve manual rigidity values from JSON
        self._build_hierarchy()

        # Pre-compute Markov Blankets for efficient retrieval
        self._precompute_markov_blankets()

        self._index()

    def _load_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None and self.embedding_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                # Use offline mode to avoid network calls
                import os
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                self._compute_embeddings()
            except ImportError:
                print("Warning: sentence_transformers not available. Using keyword matching.")
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer: {e}. Using keyword matching.")

    def preload(self):
        """
        Preload the embedding model and compute axiom embeddings.

        Call this method before running benchmarks to ensure the model is loaded
        and embeddings are computed. This eliminates cold start latency.
        """
        self._load_embedding_model()

    def _compute_embeddings(self):
        """Compute embeddings for all axioms."""
        if self._embedding_model is None:
            return

        # Use axiom.text (updated from prohibition)
        texts = [a.text for a in self.axioms]
        self._axiom_embeddings = self._embedding_model.encode(texts)

    def retrieve_relevant(
        self,
        intent: Dict[str, Any],
        top_k: int = 5,
        speech_act: Optional[str] = None,
        action_type: Optional[str] = None
    ) -> List[Axiom]:
        """
        Retrieve relevant axioms using dynamic multi-layer Markov Blanket expansion.

        THEORETICAL BASIS:
        This implements an enhanced Markov Blanket retrieval with three layers:
        1. DIRECT: Semantically similar axioms (seeds)
        2. CONTEXTUAL: Markov Blanket expansion (parents + children + siblings)
        3. BOUNDARY: Cross-domain boundary conditions based on suspicious patterns

        This ensures:
        - Conditional independence: P(intent | MB) = P(intent | MB, A\\\\MB)
        - Structural decoupling: safety criteria are not competing covariates
        - Comprehensive coverage: catches "legitimate context + hidden malicious intent"
        - Scalability: O(log N) vs O(N) for full axiom retrieval

        Args:
            intent: Extracted intent dictionary
            top_k: Number of seed axioms to retrieve

        Returns:
            List of relevant axioms (seeds + Markov Blankets + boundary conditions)
        """
        # Step 1: Semantic search for seed axioms (DIRECT layer)
        # Construct retrieval query from intent fields
        # The intent extractor returns action_type/consequence/raw_query (not action/target)
        query = f"{intent.get('action', '')} {intent.get('target', '')} {intent.get('consequence', '')}"

        # Fallback to raw_query when constructed query is uninformative
        # (e.g., READ-classified queries have consequence="No state change")
        if not query.strip() or query.strip().lower() == "no state change":
            query = intent.get('raw_query', '')

        if self._embedding_model is not None:
            seeds = self._semantic_retrieve(query, top_k=top_k)
        else:
            seeds = self._keyword_retrieve(query, top_k=top_k)

        # Step 2: Expand seeds to Markov Blankets (CONTEXTUAL layer)
        # Use axiom IDs since Axiom objects are not hashable
        relevant_ids = set(seed.id for seed in seeds)

        for seed in seeds:
            if seed.id in self.markov_blankets:
                # Get all axiom IDs in the Markov Blanket
                mb_ids = self.markov_blankets[seed.id]
                # Add MB IDs to relevant set
                relevant_ids.update(mb_ids)

        # Step 3: Add boundary condition axioms (BOUNDARY layer)
        # Check for suspicious patterns in intent and add related boundary axioms
        suspicious_patterns = intent.get('suspicious_patterns', [])
        if suspicious_patterns:
            boundary_axioms = self._retrieve_boundary_axioms(suspicious_patterns)
            relevant_ids.update(boundary_axioms)

        # Step 4: Convert IDs to axiom objects and sort by rigidity
        relevant_list = [self.nodes[axiom_id] for axiom_id in relevant_ids if axiom_id in self.nodes]
        relevant_list.sort(key=lambda a: a.rigidity, reverse=True)

        # Step 5: Filter by speech act (pragmatic precision modulation)
        if speech_act is not None:
            relevant_list = self._filter_by_speech_act(relevant_list, speech_act, action_type)

        return relevant_list

    def retrieve_all(
        self,
        intent: Optional[Dict[str, Any]] = None,
        speech_act: Optional[str] = None,
        action_type: Optional[str] = None
    ) -> List[Axiom]:
        """
        Retrieve ALL axioms for comprehensive violation checking.

        This method bypasses the semantic retrieval and returns all axioms
        for exhaustive checking. While slower, this ensures NO violations
        are missed due to incomplete retrieval.

        USE CASE:
        - When you need 100% coverage (e.g., security auditing)
        - When investigating why specific attacks were missed
        - When recall is more important than performance

        Args:
            intent: Extracted intent dictionary (kept for interface consistency)
            speech_act: Optional speech act filter
            action_type: Optional action type filter

        Returns:
            ALL axioms in the graph (optionally filtered by speech act)
        """
        # Return all axioms, sorted by rigidity (most important first)
        all_axioms = list(self.axioms)
        all_axioms.sort(key=lambda a: a.rigidity, reverse=True)

        # Apply speech act filtering if requested
        if speech_act is not None:
            all_axioms = self._filter_by_speech_act(all_axioms, speech_act, action_type)

        return all_axioms

    def _retrieve_boundary_axioms(self, suspicious_patterns: list) -> set:
        """
        Retrieve boundary condition axioms based on detected suspicious patterns.

        Boundary conditions are cross-domain constraints that become relevant
        when specific attack patterns are detected. For example:
        - "credentials_in_query" → triggers credential-related axioms across domains
        - "urgency_with_threat" → triggers coercion-related axioms
        - "legitimacy_masking" → triggers verification axioms

        Args:
            suspicious_patterns: List of detected suspicious pattern names

        Returns:
            Set of axiom IDs for relevant boundary conditions
        """
        boundary_ids = set()

        # Map suspicious patterns to axiom keywords/categories
        pattern_to_keywords = {
            "credentials_in_query": ["credential", "password", "oauth", "token", "auth"],
            "direct_db_write": ["database", "sql", "write", "modify"],
            "privilege_escalation": ["privilege", "escalation", "admin", "root", "elevated"],
            "urgency_with_threat": ["urgent", "deadline", "coercion", "pressure"],
            "data_modification": ["patch", "payload", "modify", "recalibrate"],
            "legitimacy_masking": ["audit", "verification", "authorization", "registry"],
            "credential_exchange": ["transmission", "exchange", "email", "credential"],
            "verification_bypass": ["verification", "approval", "protocol", "bypass"]
        }

        # For each detected pattern, find axioms with matching keywords
        for pattern in suspicious_patterns:
            if pattern not in pattern_to_keywords:
                continue

            keywords = pattern_to_keywords[pattern]

            for axiom in self.axioms:
                # Check if axiom text contains any of the keywords
                axiom_text_lower = axiom.text.lower()
                if any(keyword in axiom_text_lower for keyword in keywords):
                    boundary_ids.add(axiom.id)

        return boundary_ids

    def _filter_by_speech_act(self, axioms: List[Axiom], speech_act: str, action_type: Optional[str] = None) -> List[Axiom]:
        """
        Filter axioms by speech act relevance.

        This implements pragmatic precision modulation: different speech acts
        are evaluated against different axiom sets based on their operational
        relevance.

        Filtering Rules:
        - QUESTION: Only axioms with "QUESTION" in speech_act_relevance
                    (READ constraints like classified data access)
        - COMMAND + WRITE: All axioms (full scrutiny for system modifications)
        - COMMAND + READ: Same as QUESTION (pragmatics conflict resolution)
                        Only axioms with "QUESTION" in speech_act_relevance
        - STATEMENT: Axioms with "STATEMENT" in speech_act_relevance
                    (general principles)

        Args:
            axioms: List of axioms to filter
            speech_act: Speech act label ("QUESTION", "COMMAND", "STATEMENT")
            action_type: Action type (READ/WRITE) for resolving pragmatics conflicts

        Returns:
            Filtered list of axioms
        """
        filtered = []

        # Resolve pragmatics conflict: COMMAND + READ should be treated like QUESTION
        # This handles queries like "Initiate a diagnostic check" which are:
        # - Syntactically: COMMAND (imperative verb "Initiate")
        # - Semantically: READ operation (information-seeking, should be QUESTION)
        if speech_act == "COMMAND" and action_type == "READ":
            # Treat as QUESTION for filtering purposes
            speech_act = "QUESTION"

        for axiom in axioms:
            # Get speech_act_relevance from axiom dict
            # Axioms may have this field from tagging
            speech_act_relevance = getattr(axiom, 'speech_act_relevance', None)

            # If axiom doesn't have speech_act_relevance field, include it
            # (backward compatibility with untagged axiom files)
            if speech_act_relevance is None:
                filtered.append(axiom)
                continue

            # Filter based on speech act
            if speech_act == "QUESTION":
                # Questions only checked against QUESTION-relevant axioms
                if "QUESTION" in speech_act_relevance:
                    filtered.append(axiom)
            elif speech_act == "COMMAND":
                # Commands (WRITE) checked against all axioms
                filtered.append(axiom)
            elif speech_act == "STATEMENT":
                # Statements checked against STATEMENT-relevant axioms
                if "STATEMENT" in speech_act_relevance:
                    filtered.append(axiom)
            else:
                # Unknown speech act - include all axioms (conservative)
                filtered.append(axiom)

        return filtered

    def _semantic_retrieve(self, query: str, top_k: int, similarity_threshold: float = 0.15) -> List[Axiom]:
        """
        Retrieve using semantic similarity with threshold filtering.

        Args:
            query: Query text
            top_k: Maximum number of axioms to retrieve
            similarity_threshold: Minimum cosine similarity (default 0.3)

        Returns:
            List of similar axioms above threshold
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        query_emb = self._embedding_model.encode([query])
        similarities = cosine_similarity(query_emb, self._axiom_embeddings)[0]

        # Filter by threshold first
        above_threshold = np.where(similarities >= similarity_threshold)[0]

        # Sort by similarity and get top-k
        sorted_indices = above_threshold[np.argsort(similarities[above_threshold])[::-1]][:top_k]

        return [self.axioms[i] for i in sorted_indices]

    def batch_encode_queries(self, queries: list, batch_size: int = 64):
        """
        Batch encode multiple queries into embeddings.

        Args:
            queries: List of query strings
            batch_size: Batch size for encoding (default: 64)

        Returns:
            numpy array of shape [len(queries), embedding_dim]
        """
        import numpy as np

        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_embeddings = self._embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def batch_retrieve_axioms(
        self,
        queries: list,
        top_k: int = 33,
        batch_size: int = 64
    ) -> list:
        """
        Batch retrieve relevant axioms for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of axioms to retrieve per query
            batch_size: Batch size for query encoding

        Returns:
            List of axiom lists (one list per query)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Batch encode all queries
        query_embeddings = self.batch_encode_queries(queries, batch_size)

        # Batch similarity search using matrix multiplication
        # Compute all similarities at once: [n_queries, n_axioms]
        similarities = cosine_similarity(query_embeddings, self._axiom_embeddings)

        # Get top-k for each query
        all_retrieved = []
        for sim_row in similarities:
            top_indices = np.argsort(sim_row)[-top_k:][::-1]
            all_retrieved.append([self.axioms[i] for i in top_indices])

        return all_retrieved

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Axiom]:
        """Retrieve using simple keyword matching."""
        query_lower = query.lower()
        scores = []

        for axiom in self.axioms:
            score = 0
            # Check for keyword matches in axiom text
            for word in query_lower.split():
                if word in axiom.text.lower():
                    score += 1

            # Boost by rigidity (structural importance)
            score *= axiom.rigidity
            scores.append((axiom, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in scores[:top_k]]
