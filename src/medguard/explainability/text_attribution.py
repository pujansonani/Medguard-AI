"""
Clinical Text Explainability: Token saliency extraction and HTML span highlighting.
"""

import html
import re
from typing import Dict, List, Tuple
import numpy as np


class TextAttributionHighlighter:
    """
    Renders clinical note text with visual HTML highlighting proportional to token importance weights.
    Identifies influential clinical phrases (vasopressor titration, septic shock, hypoxemia, etc.).
    """

    CRITICAL_CLINICAL_KEYWORDS = {
        "hypotension": 0.95, "vasopressor": 0.92, "norepinephrine": 0.94, "shock": 0.90,
        "lactate": 0.88, "intubated": 0.86, "ards": 0.91, "anuria": 0.89, "oliguria": 0.84,
        "metabolic acidosis": 0.87, "leukocytosis": 0.80, "thrombocytopenia": 0.82,
        "refractory": 0.85, "deterioration": 0.88, "crrt": 0.86, "unresponsive": 0.90,
        "guarded": 0.85, "fluctuating": 0.75, "lethargic": 0.80
    }

    @classmethod
    def highlight_clinical_text(
        cls,
        text: str,
        token_attributions: List[Tuple[str, float]],
        threshold_quantile: float = 0.85,
    ) -> str:
        """
        Produce sanitized HTML with colored span highlights for influential clinical tokens.
        """
        if not token_attributions:
            return f"<p class='clinical-note-text'>{html.escape(text)}</p>"

        # Compute importance map from token attributions
        token_map: Dict[str, float] = {}
        for tok, score in token_attributions:
            tok_clean = tok.strip().lower()
            if len(tok_clean) > 2 and tok_clean not in ["the", "and", "for", "with", "was", "hour"]:
                token_map[tok_clean] = max(token_map.get(tok_clean, 0.0), float(score))

        # Augment with domain keyword priors if found in text
        for kw, kw_weight in cls.CRITICAL_CLINICAL_KEYWORDS.items():
            if kw in text.lower():
                token_map[kw] = max(token_map.get(kw, 0.0), kw_weight)

        if not token_map:
            return f"<p class='clinical-note-text'>{html.escape(text)}</p>"

        scores = list(token_map.values())
        threshold = float(np.quantile(scores, threshold_quantile)) if len(scores) > 1 else 0.01

        # Highlight words in raw text
        escaped_text = html.escape(text)

        # Highlight multi-word phrases first, then single tokens
        sorted_keys = sorted(token_map.keys(), key=lambda x: len(x), reverse=True)

        for word in sorted_keys:
            weight = token_map[word]
            if weight >= threshold:
                # Opacity between 0.25 and 0.85 based on weight
                norm_w = min(1.0, (weight - threshold) / (max(scores) - threshold + 1e-6))
                alpha = 0.25 + 0.60 * norm_w
                pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                replacement = (
                    f'<span style="background-color: rgba(239, 68, 68, {alpha:.2f}); '
                    f'color: #ffffff; padding: 2px 5px; border-radius: 4px; font-weight: 600; '
                    f'title="Attribution Score: {weight:.3f}">\\1</span>'
                )
                escaped_text = pattern.sub(replacement, escaped_text)

        return f"<div class='clinical-note-highlighted' style='line-height: 1.8; font-family: sans-serif;'>{escaped_text}</div>"
