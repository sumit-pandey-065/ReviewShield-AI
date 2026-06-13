"""
ReviewShield AI - Text Preprocessing Module
Handles all text cleaning and normalization (NLTK-free version)
"""

import re
import string

# Basic English stopwords (no nltk needed)
STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be',
    'been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how','all',
    'both','each','few','more','most','other','some','such','no','nor','not','only',
    'own','same','so','than','too','very','s','t','can','will','just','don','should',
    'now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn',
    'hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn'
}

def simple_stem(word):
    """Very basic suffix stripping stemmer."""
    if len(word) <= 4:
        return word
    for suffix in ['ing', 'tion', 'ness', 'ment', 'able', 'ible', 'ful', 'less',
                   'ous', 'ive', 'ize', 'ise', 'ed', 'er', 'ly', 'es', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def clean_text(text):
    """Full preprocessing pipeline for a review text."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [simple_stem(t) for t in tokens]
    return ' '.join(tokens)

def get_raw_features(text):
    if not isinstance(text, str):
        text = str(text)
    return {
        'original_text': text,
        'char_count': len(text),
        'word_count': len(text.split()),
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'exclamation_count': text.count('!'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }
