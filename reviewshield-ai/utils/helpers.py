"""
ReviewShield AI - Utility Functions and Constants
"""

def compute_authenticity_score(ml_result, spam_result, sentiment_result):
    """
    Compute a final authenticity score 0-100.
    Higher = more authentic/genuine.
    """
    # Base score from ML confidence
    if ml_result['is_fake']:
        base_score = 100 - ml_result['confidence']
    else:
        base_score = ml_result['genuine_probability']

    # Apply spam penalty
    spam_penalty = spam_result['penalty']
    base_score = max(0, base_score - spam_penalty * 0.5)

    # Clip to range
    final_score = max(0, min(100, base_score))
    return round(final_score, 1)


def get_risk_level(score):
    """Return risk label based on authenticity score."""
    if score >= 75:
        return ('LOW RISK', '🟢', '#22c55e')
    elif score >= 50:
        return ('MEDIUM RISK', '🟡', '#f59e0b')
    elif score >= 25:
        return ('HIGH RISK', '🟠', '#f97316')
    else:
        return ('CRITICAL RISK', '🔴', '#ef4444')


def get_verdict(is_fake, confidence):
    """Get human-readable verdict."""
    if is_fake:
        if confidence >= 85:
            return "Very likely FAKE", "🚨"
        elif confidence >= 70:
            return "Likely FAKE", "⚠️"
        else:
            return "Possibly FAKE", "❓"
    else:
        if confidence >= 85:
            return "Very likely GENUINE", "✅"
        elif confidence >= 70:
            return "Likely GENUINE", "👍"
        else:
            return "Possibly GENUINE", "🤔"
