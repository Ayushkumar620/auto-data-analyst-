"""Universal Multi-Modality Analytical Engines.

Supports:
1. TextModalityEngine: NLP sentiment, keyword extraction, TF-IDF n-grams, text statistics.
2. RelationalModalityEngine: Foreign key discovery, schema graphs, cross-table entity joins.
3. HierarchicalJSONEngine: Dynamic nesting/unnesting and hierarchical JSON flattening.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# 1. Text & Unstructured NLP Modality Engine
# ------------------------------------------------------------------------------

@dataclass
class TextAnalysisReport:
    """NLP analysis results for free-form text columns."""
    column_name: str
    total_documents: int
    avg_word_count: float
    vocabulary_size: int
    lexical_diversity: float  # Type-Token Ratio (TTR)
    sentiment_distribution: Dict[str, float]  # positive, neutral, negative %
    top_keywords: List[Tuple[str, int]]
    top_bigrams: List[Tuple[str, int]]
    sample_snippets: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "total_documents": self.total_documents,
            "avg_word_count": round(self.avg_word_count, 2),
            "vocabulary_size": self.vocabulary_size,
            "lexical_diversity": round(self.lexical_diversity, 4),
            "sentiment_distribution": {k: round(v, 3) for k, v in self.sentiment_distribution.items()},
            "top_keywords": [{"keyword": k, "count": c} for k, c in self.top_keywords],
            "top_bigrams": [{"bigram": b, "count": c} for b, c in self.top_bigrams],
            "sample_snippets": self.sample_snippets[:3],
        }


class TextModalityEngine:
    """Extracts NLP intelligence and sentiment patterns from text columns."""

    # Simple deterministic sentiment lexicons
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "love", "best", "positive", "happy",
        "satisfied", "superb", "fast", "helpful", "perfect", "boost", "growth", "profit",
        "gain", "efficient", "innovative", "reliable", "wonderful", "outstanding",
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "poor", "awful", "hate", "worst", "negative", "unhappy",
        "frustrated", "slow", "broken", "bug", "error", "drop", "decline", "loss",
        "fail", "failure", "inefficient", "issue", "problem", "defect", "complaint",
    }
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "of", "from", "as", "is", "was", "are", "were", "it", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "my", "your", "their",
    }

    @classmethod
    def is_text_column(cls, series: pd.Series) -> bool:
        """Check if column contains unstructured natural language text."""
        clean = series.dropna().astype(str)
        if clean.empty or len(clean) < 3:
            return False
        # Average character length > 25 and word count > 3
        avg_len = clean.str.len().mean()
        avg_words = clean.str.split().str.len().mean()
        return bool(avg_len >= 25 and avg_words >= 3)

    @classmethod
    def analyze_text_column(cls, series: pd.Series, column_name: str = "text") -> TextAnalysisReport:
        """Run full NLP profiling on a text series."""
        clean = series.dropna().astype(str)
        total_docs = len(clean)

        if total_docs == 0:
            return TextAnalysisReport(
                column_name=column_name, total_documents=0, avg_word_count=0.0,
                vocabulary_size=0, lexical_diversity=0.0, sentiment_distribution={},
                top_keywords=[], top_bigrams=[], sample_snippets=[],
            )

        all_tokens: List[str] = []
        doc_word_counts: List[int] = []
        sentiments: List[str] = []

        for doc in clean:
            words = re.findall(r"\b[a-zA-Z]{2,}\b", doc.lower())
            doc_word_counts.append(len(words))
            all_tokens.extend(words)

            # Sentiment calculation
            pos_count = sum(1 for w in words if w in cls.POSITIVE_WORDS)
            neg_count = sum(1 for w in words if w in cls.NEGATIVE_WORDS)
            if pos_count > neg_count:
                sentiments.append("positive")
            elif neg_count > pos_count:
                sentiments.append("negative")
            else:
                sentiments.append("neutral")

        # Vocabulary & Diversity
        unique_tokens = set(all_tokens)
        vocab_size = len(unique_tokens)
        ttr = (vocab_size / len(all_tokens)) if all_tokens else 0.0

        # Sentiment distribution
        sent_counts = collections.Counter(sentiments)
        sent_dist = {k: sent_counts.get(k, 0) / max(total_docs, 1) for k in ("positive", "neutral", "negative")}

        # Top keywords (excluding stopwords)
        filtered_tokens = [w for w in all_tokens if w not in cls.STOPWORDS]
        top_kw = collections.Counter(filtered_tokens).most_common(10)

        # Top Bigrams
        bigrams = [
            f"{filtered_tokens[i]} {filtered_tokens[i+1]}"
            for i in range(len(filtered_tokens) - 1)
        ]
        top_bg = collections.Counter(bigrams).most_common(5)

        return TextAnalysisReport(
            column_name=column_name,
            total_documents=total_docs,
            avg_word_count=float(np.mean(doc_word_counts)),
            vocabulary_size=vocab_size,
            lexical_diversity=float(ttr),
            sentiment_distribution=sent_dist,
            top_keywords=top_kw,
            top_bigrams=top_bg,
            sample_snippets=list(clean.head(3)),
        )


# ------------------------------------------------------------------------------
# 2. Relational Multi-Table Modality Engine
# ------------------------------------------------------------------------------

@dataclass
class ForeignKeyRelationship:
    """Discovered foreign key link between two tables."""
    parent_table: str
    parent_key: str
    child_table: str
    child_key: str
    cardinality: str  # one_to_many, one_to_one, many_to_many
    match_percentage: float


class RelationalModalityEngine:
    """Discovers relationships and joins across multi-table datasets."""

    @classmethod
    def discover_relationships(
        cls,
        tables: Dict[str, pd.DataFrame],
    ) -> List[ForeignKeyRelationship]:
        """Detect primary-to-foreign key candidate links between tables."""
        relationships: List[ForeignKeyRelationship] = []
        table_names = list(tables.keys())

        for i in range(len(table_names)):
            for j in range(len(table_names)):
                if i == j:
                    continue
                t1_name, t2_name = table_names[i], table_names[j]
                df1, df2 = tables[t1_name], tables[t2_name]

                # Look for common column names or id suffixes
                for col1 in df1.columns:
                    col1_clean = col1.lower()
                    for col2 in df2.columns:
                        col2_clean = col2.lower()

                        is_match_name = (
                            col1_clean == col2_clean
                            or col1_clean == f"{t2_name.lower()}_id"
                            or col2_clean == f"{t1_name.lower()}_id"
                            or col1_clean == "id" and col2_clean.endswith("_id")
                        )

                        if is_match_name:
                            s1 = set(df1[col1].dropna())
                            s2 = set(df2[col2].dropna())
                            if s1 and s2:
                                intersection = s1.intersection(s2)
                                match_pct = len(intersection) / max(len(s2), 1)
                                if match_pct >= 0.70:
                                    is_unique1 = df1[col1].is_unique
                                    card = "one_to_many" if is_unique1 else "many_to_many"
                                    relationships.append(
                                        ForeignKeyRelationship(
                                            parent_table=t1_name,
                                            parent_key=col1,
                                            child_table=t2_name,
                                            child_key=col2,
                                            cardinality=card,
                                            match_percentage=round(match_pct * 100, 2),
                                        )
                                    )

        return relationships

    @classmethod
    def auto_join_tables(
        cls,
        tables: Dict[str, pd.DataFrame],
        relationships: Optional[List[ForeignKeyRelationship]] = None,
    ) -> pd.DataFrame:
        """Merge related tables into a unified analytical DataFrame."""
        if not tables:
            return pd.DataFrame()
        if len(tables) == 1:
            return next(iter(tables.values()))

        if relationships is None:
            relationships = cls.discover_relationships(tables)

        if not relationships:
            # Fallback: return primary/largest table
            return max(tables.values(), key=len)

        # Merge on primary relationship
        rel = relationships[0]
        p_df = tables[rel.parent_table]
        c_df = tables[rel.child_table]

        merged = pd.merge(
            p_df,
            c_df,
            left_on=rel.parent_key,
            right_on=rel.child_key,
            how="left",
            suffixes=(f"_{rel.parent_table}", f"_{rel.child_table}"),
        )
        return merged


# ------------------------------------------------------------------------------
# 3. Hierarchical Nested JSON Engine
# ------------------------------------------------------------------------------

class HierarchicalJSONEngine:
    """Recursively flattens complex nested JSON into flat relational tabular data."""

    @classmethod
    def flatten_nested_data(
        cls,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        max_level: int = 5,
    ) -> pd.DataFrame:
        """Flatten nested dictionaries and unnest arrays."""
        if isinstance(data, dict):
            # If root dictionary contains a list of entities
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return pd.json_normalize(v, max_level=max_level)
            data = [data]

        df = pd.json_normalize(data, max_level=max_level)

        # Explode 1-level list columns if any
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                try:
                    df = df.explode(col)
                except Exception:
                    pass

        return df
