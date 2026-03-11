"""
General functions for OCR apps
"""

import re
from collections import Counter

def calculate_bow_accuracy_new(prediction, ground_truth):
    """Measures Information Extraction accuracy using a bag-of-words approach."""
    
    # 1. Standardize: Lowercase and strip punctuation/symbols
    def clean_text(text):
        # We remove non-alphanumeric characters but keep spaces
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    pred_words = clean_text(prediction)
    gt_words = clean_text(ground_truth)
    
    if not gt_words: 
        return 0.0, set()

    # 2. Use Counter for "Multi-Set" logic (tracks word frequency)
    pred_counts = Counter(pred_words)
    gt_counts = Counter(gt_words)
    
    # 3. Calculate "Hits" (Intersection of frequencies)
    # This ensures if 'Total' appears 3 times, the model must find it 3 times.
    hits = pred_counts & gt_counts
    total_hits = sum(hits.values())
    
    accuracy = total_hits / len(gt_words)
    
    # We return the accuracy and the list of unique words found for the UI
    return accuracy, set(hits.keys())






# BOW: Information Extraction Accuracy (ignores order) vs WER: Sequential Accuracy
def calculate_bow_accuracy(prediction, ground_truth):
    """Measures Information Extraction accuracy (ignores word order)."""
    # Convert to sets (removes duplicates and ignores order)
    pred_set = set(prediction.lower().split())
    gt_set = set(ground_truth.lower().split())
    # Calculate how many truth words were found
    intersection = pred_set.intersection(gt_set)
    if not gt_set: return 0.0
    return (len(intersection) / len(gt_set)), intersection