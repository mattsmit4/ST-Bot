"""
Query Analyzer Module - Technical Requirement Detection

Analyzes user queries for explicit technical requirements that should
override device inference or generic defaults.

Design Principles:
- Priority-based: Explicit mentions > Technical requirements > Inferred defaults
- Extensible: Easy to add new technical terms without changing code
- Centralized: All technical knowledge in one place
- Transparent: Returns confidence scores and reasoning
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import re


@dataclass
class QueryRequirement:
    """
    A technical requirement detected in the query.

    Attributes:
        requirement_type: Type of requirement (connector, feature, etc.)
        value: The specific value (e.g., "DisplayPort", "4K")
        priority: Priority level (1=highest, 3=lowest)
        confidence: Confidence score 0.0-1.0
        reason: Why this was detected
        matched_term: The actual term that was matched in the query
    """
    requirement_type: str
    value: str
    priority: int  # 1 = explicit, 2 = technical, 3 = inferred
    confidence: float
    reason: str
    matched_term: str


class QueryAnalyzer:
    """
    Analyzes queries for explicit technical requirements.

    Priority Levels:
    1. EXPLICIT - User explicitly mentioned connector/spec ("USB-C to HDMI", "DP 1.4")
    2. TECHNICAL - Technical requirement implies connector ("daisy-chain" → DisplayPort)
    3. INFERRED - Device-based inference ("MacBook" → USB-C)

    Example:
        >>> analyzer = QueryAnalyzer()
        >>> reqs = analyzer.analyze("I need DP 1.4 cables for daisy-chaining")
        >>> print(reqs[0].value)  # "DisplayPort"
        >>> print(reqs[0].priority)  # 1 (explicit mention)
    """

    def __init__(self):
        """Initialize with technical term database."""

        # Priority 1: Explicit connector mentions
        # Format: {pattern: (connector_type, confidence)}
        self.explicit_connectors = {
            # DisplayPort variants
            r'\bdp\b': ('DisplayPort', 0.95),
            r'\bdisplayport\b': ('DisplayPort', 1.0),
            r'\bdisplay port\b': ('DisplayPort', 1.0),
            r'\bdp 1\.4\b': ('DisplayPort', 1.0),
            r'\bdp 2\.0\b': ('DisplayPort', 1.0),
            r'\bdp 2\.1\b': ('DisplayPort', 1.0),

            # USB-C variants
            r'\busb-c\b': ('USB-C', 1.0),
            r'\busb c\b': ('USB-C', 1.0),
            r'\btype-c\b': ('USB-C', 0.95),
            r'\btype c\b': ('USB-C', 0.95),

            # HDMI variants
            r'\bhdmi\b': ('HDMI', 1.0),
            r'\bhdmi 2\.0\b': ('HDMI', 1.0),
            r'\bhdmi 2\.1\b': ('HDMI', 1.0),

            # Thunderbolt
            r'\bthunderbolt\b': ('Thunderbolt', 1.0),
            r'\btb3\b': ('Thunderbolt', 0.9),
            r'\btb4\b': ('Thunderbolt', 0.9),

            # Others
            r'\bdvi\b': ('DVI', 1.0),
            r'\bvga\b': ('VGA', 1.0),
        }

        # Priority 2: Technical requirements that imply connector
        # Format: {pattern: (connector_type, reason, confidence)}
        self.technical_requirements = {
            # DisplayPort-specific features
            r'\bdaisy.?chain': ('DisplayPort', 'Daisy-chaining requires DisplayPort MST', 1.0),
            r'\bmst\b': ('DisplayPort', 'MST (Multi-Stream Transport) is DisplayPort feature', 1.0),
            r'\bmulti.?stream': ('DisplayPort', 'Multi-stream is DisplayPort feature', 0.95),

            # Thunderbolt-specific features
            r'\b40\s*gbps\b': ('Thunderbolt', '40 Gbps is Thunderbolt speed', 0.9),
            r'\bpcie\b': ('Thunderbolt', 'PCIe tunneling requires Thunderbolt', 0.85),

            # Power delivery (typically USB-C)
            r'\bpower delivery\b': ('USB-C', 'Power Delivery is USB-C feature', 0.8),
            r'\bpd charging\b': ('USB-C', 'PD charging requires USB-C', 0.8),

            # High bandwidth (usually DisplayPort or HDMI 2.1)
            r'\b8k\b': ('DisplayPort', '8K typically requires DisplayPort or HDMI 2.1', 0.7),
        }

        # Priority 2: Feature requirements
        # Format: {pattern: (feature_name, confidence)}
        self.feature_requirements = {
            r'\b4k\b': ('4K', 0.95),
            r'\b4k@60\b': ('4K', 1.0),
            r'\b4k 60hz\b': ('4K', 1.0),
            r'\b8k\b': ('8K', 0.95),
            r'\bhdr\b': ('HDR', 0.9),
            r'\barc\b': ('ARC', 0.85),
            r'\bearc\b': ('eARC', 0.9),
        }

    def analyze(self, query: str) -> List[QueryRequirement]:
        """
        Analyze query for technical requirements.

        Args:
            query: User's natural language query

        Returns:
            List of QueryRequirement sorted by priority (highest first)

        Example:
            >>> reqs = analyzer.analyze("I need DP 1.4 for daisy-chaining")
            >>> len(reqs)  # 2 requirements found
            >>> reqs[0].priority  # 1 (explicit "DP 1.4")
            >>> reqs[1].priority  # 2 (technical "daisy-chaining")
        """
        query_lower = query.lower()
        requirements = []

        # 1. Check for explicit connector mentions (Priority 1)
        for pattern, (connector, confidence) in self.explicit_connectors.items():
            match = re.search(pattern, query_lower)
            if match:
                requirements.append(QueryRequirement(
                    requirement_type='connector',
                    value=connector,
                    priority=1,  # Highest priority
                    confidence=confidence,
                    reason=f"Explicit mention of {connector}",
                    matched_term=match.group()
                ))

        # 2. Check for technical requirements (Priority 2)
        for pattern, (connector, reason, confidence) in self.technical_requirements.items():
            match = re.search(pattern, query_lower)
            if match:
                requirements.append(QueryRequirement(
                    requirement_type='connector',
                    value=connector,
                    priority=2,  # Medium priority
                    confidence=confidence,
                    reason=reason,
                    matched_term=match.group()
                ))

        # 3. Check for feature requirements (Priority 2)
        for pattern, (feature, confidence) in self.feature_requirements.items():
            match = re.search(pattern, query_lower)
            if match:
                requirements.append(QueryRequirement(
                    requirement_type='feature',
                    value=feature,
                    priority=2,
                    confidence=confidence,
                    reason=f"Explicit mention of {feature} feature",
                    matched_term=match.group()
                ))

        # Sort by priority (1 = highest) then confidence
        requirements.sort(key=lambda r: (r.priority, -r.confidence))

        return requirements

    def get_connector_requirement(self, query: str) -> Optional[QueryRequirement]:
        """
        Get the highest-priority connector requirement.

        Args:
            query: User query

        Returns:
            Highest priority connector requirement, or None
        """
        reqs = self.analyze(query)
        connector_reqs = [r for r in reqs if r.requirement_type == 'connector']
        return connector_reqs[0] if connector_reqs else None

    def get_feature_requirements(self, query: str) -> List[str]:
        """
        Get all feature requirements.

        Args:
            query: User query

        Returns:
            List of feature names (e.g., ['4K', 'HDR'])
        """
        reqs = self.analyze(query)
        return [r.value for r in reqs if r.requirement_type == 'feature']

    def add_technical_term(
        self,
        pattern: str,
        connector: str,
        reason: str,
        confidence: float = 1.0
    ):
        """
        Add a new technical term pattern (for extending the system).

        Args:
            pattern: Regex pattern to match
            connector: Connector type this implies
            reason: Explanation of why
            confidence: Confidence score 0.0-1.0

        Example:
            >>> analyzer.add_technical_term(
            ...     r'\bhdmi alt mode\b',
            ...     'USB-C',
            ...     'HDMI Alt Mode requires USB-C',
            ...     0.95
            ... )
        """
        self.technical_requirements[pattern] = (connector, reason, confidence)


# Example usage and testing
if __name__ == "__main__":
    analyzer = QueryAnalyzer()

    # Test cases
    test_queries = [
        "I need DP 1.4 cables for daisy-chaining monitors",
        "USB-C to HDMI cable for my MacBook",
        "Do you have cables that support MST?",
        "I need a 4K cable for my monitor",
        "Thunderbolt cable with 40 Gbps",
        "Connect my MacBook to my monitor",  # No explicit requirements
    ]

    print("Query Analyzer Test Results:")
    print("=" * 70)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        reqs = analyzer.analyze(query)

        if reqs:
            print(f"  Found {len(reqs)} requirement(s):")
            for req in reqs:
                print(f"    - {req.value} (Priority {req.priority}, Confidence {req.confidence:.2f})")
                print(f"      Reason: {req.reason}")
                print(f"      Matched: '{req.matched_term}'")
        else:
            print("  No explicit requirements found")
