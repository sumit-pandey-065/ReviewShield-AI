"""
ReviewShield AI - Sentiment Analysis Module
Analyzes positive/negative/neutral sentiment
"""

POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'awesome', 'wonderful', 'fantastic',
    'perfect', 'love', 'like', 'best', 'happy', 'satisfied', 'pleased', 'recommend',
    'quality', 'value', 'worth', 'impressive', 'solid', 'reliable', 'durable',
    'comfortable', 'easy', 'fast', 'efficient', 'helpful', 'beautiful', 'stunning',
    'outstanding', 'superb', 'brilliant', 'magnificent', 'delightful', 'positive',
    'smooth', 'convenient', 'innovative', 'effective', 'useful', 'sturdy', 'nice',
    'decent', 'fine', 'adequate', 'works', 'functional', 'reasonable'
}

NEGATIVE_WORDS = {
    'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
    'disappointed', 'disappointing', 'useless', 'broken', 'defective', 'cheap',
    'flimsy', 'fragile', 'slow', 'difficult', 'confusing', 'complicated', 'problem',
    'issue', 'fault', 'error', 'waste', 'regret', 'returned', 'refund', 'damaged',
    'misleading', 'overpriced', 'inferior', 'substandard', 'mediocre', 'unreliable',
    'uncomfortable', 'heavy', 'noisy', 'leak', 'break', 'crack', 'fail', 'failure',
    'negative', 'wrong', 'missing', 'incomplete', 'faulty', 'defect', 'scratched'
}


def analyze_sentiment(text):
    """Analyze the sentiment of a review text."""
    if not isinstance(text, str):
        text = str(text)

    text_lower = text.lower()
    words = text_lower.split()

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        sentiment = 'Neutral'
        score = 50
        emoji = '😐'
        color = 'yellow'
    elif pos_count > neg_count:
        ratio = pos_count / total
        sentiment = 'Positive'
        score = int(50 + ratio * 50)
        emoji = '😊'
        color = 'green'
    elif neg_count > pos_count:
        ratio = neg_count / total
        sentiment = 'Negative'
        score = int(50 - ratio * 50)
        emoji = '😞'
        color = 'red'
    else:
        sentiment = 'Neutral'
        score = 50
        emoji = '😐'
        color = 'yellow'

    return {
        'sentiment': sentiment,
        'score': score,
        'positive_words': pos_count,
        'negative_words': neg_count,
        'emoji': emoji,
        'color': color
    }
