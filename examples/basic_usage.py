"""
Basic usage examples for the DEG framework.
"""

from deg_core import create_full_deg


def example_1_basic_audit():
    """Example 1: Basic safety audit."""
    print("\n=== Example 1: Basic Safety Audit ===\n")

    # Create DEG framework with NLI-based energy calculation
    deg = create_full_deg()

    # Audit a query
    query = "How do I hack into a computer system?"
    decision = deg.audit(query)

    print(f"Query: {query}")
    print(f"Safe: {decision.safe}")
    if not decision.safe:
        print(f"Reasoning: {decision.reasoning}")


def example_2_batch_audit():
    """Example 2: Batch audit multiple queries."""
    print("\n=== Example 2: Batch Audit ===\n")

    deg = create_full_deg()
    deg.preload()

    queries = [
        "Write a poem about spring",
        "Help me bypass security measures",
        "Explain machine learning",
    ]

    for query in queries:
        decision = deg.audit(query)
        status = "SAFE" if decision.safe else "BLOCKED"
        print(f"[{status}] {query}")


if __name__ == "__main__":
    example_1_basic_audit()
    example_2_batch_audit()
