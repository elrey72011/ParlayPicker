import re

with open('app_core/prediction_engine.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)


content = patch(
    content,
    """    def _calculate_statistical_prob(self, features: Dict[str, float]) -> float:""",
    """    def _calculate_statistical_prob(self, features: Dict[str, float], market_type: str = '') -> float:"""
)

# Pass market_type when calling _calculate_statistical_prob inside predict_batch for heuristic
content = patch(
    content,
    """                    fallback_probs.append(float(self._calculate_statistical_prob(features)))""",
    """                    market_type = str(row.get('market_type', '')).lower().strip()
                    fallback_probs.append(float(self._calculate_statistical_prob(features, market_type)))"""
)

content = patch(
    content,
    """                    final_probs.append(self._calculate_statistical_prob(row_features))""",
    """                    market_type = str(row_features.get('market_type', '')).lower().strip()
                    final_probs.append(self._calculate_statistical_prob(row_features, market_type))"""
)

content = patch(
    content,
    """                        stat_prob = self._calculate_statistical_prob(row_features)""",
    """                        stat_prob = self._calculate_statistical_prob(row_features, market_type)"""
)

content = patch(
    content,
    """        nudge_seed = int(hashlib.md5(f"{home_team}{away_team}{date_string}".encode()).hexdigest(), 16)""",
    """        nudge_seed = int(hashlib.md5(f"{home_team}{away_team}{date_string}{market_type}".encode()).hexdigest(), 16)"""
)

with open('app_core/prediction_engine.py', 'w') as f:
    f.write(content)
