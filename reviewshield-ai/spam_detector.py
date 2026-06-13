"""
ReviewShield AI - Spam Pattern Detector
Detects suspicious patterns in reviews
"""

import re


PROMOTIONAL_PHRASES = [
    'buy now', 'order today', 'limited time', 'special offer', 'best deal',
    'click here', 'don\'t miss', 'act now', 'free shipping', 'discount code',
    'promo code', 'sale price', 'best price', 'flash sale', 'sponsored',
    'paid review', 'received for free', 'in exchange for', 'affiliate',
    'get yours', 'limited stock', 'must buy', 'life changing', 'changed my life'
]

SPAM_WORDS = [
    'amazing', 'incredible', 'fantastic', 'perfect', 'best ever', 'greatest',
    'revolutionary', 'unbelievable', 'outstanding', 'exceptional', 'phenomenal'
]


def detect_spam_patterns(text):
    """Analyze text for spam/fake indicators. Returns dict of findings."""
    if not isinstance(text, str):
        text = str(text)

    indicators = []
    score_penalty = 0

    text_lower = text.lower()

    # 1. Excessive exclamation marks
    excl_count = text.count('!')
    if excl_count >= 3:
        indicators.append(f"Excessive exclamation marks ({excl_count} found)")
        score_penalty += min(excl_count * 4, 20)

    # 2. ALL CAPS usage
    words = text.split()
    if len(words) > 0:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        caps_ratio = len(caps_words) / len(words)
        if caps_ratio > 0.25:
            indicators.append(f"Excessive ALL CAPS usage ({int(caps_ratio*100)}% of words)")
            score_penalty += min(int(caps_ratio * 40), 20)

    # 3. Promotional phrases
    found_promos = []
    for phrase in PROMOTIONAL_PHRASES:
        if phrase in text_lower:
            found_promos.append(phrase)
    if found_promos:
        indicators.append(f"Promotional language detected: {', '.join(found_promos[:3])}")
        score_penalty += min(len(found_promos) * 8, 25)

    # 4. Repeated words
    word_list = text_lower.split()
    word_freq = {}
    for w in word_list:
        if len(w) > 3:
            word_freq[w] = word_freq.get(w, 0) + 1
    repeated = {w: c for w, c in word_freq.items() if c >= 3}
    if repeated:
        top_repeated = sorted(repeated.items(), key=lambda x: -x[1])[:3]
        indicators.append(f"Repeated words: {', '.join([f'{w}({c}x)' for w, c in top_repeated])}")
        score_penalty += min(len(repeated) * 5, 15)

    # 5. Very short review (suspicious)
    word_count = len(words)
    if word_count < 5:
        indicators.append("Unusually short review")
        score_penalty += 10

    # 6. Spam superlatives
    found_spam_words = [w for w in SPAM_WORDS if w in text_lower]
    if len(found_spam_words) >= 3:
        indicators.append(f"Overuse of superlatives: {', '.join(found_spam_words[:3])}")
        score_penalty += min(len(found_spam_words) * 4, 15)

    # 7. Excessive punctuation overall
    punct_count = sum(1 for c in text if c in '!?*#@$%')
    if punct_count > 5:
        indicators.append(f"Excessive special characters ({punct_count} found)")
        score_penalty += min(punct_count * 2, 10)

    return {
        'indicators': indicators,
        'penalty': min(score_penalty, 60),
        'spam_count': len(indicators)
    }
