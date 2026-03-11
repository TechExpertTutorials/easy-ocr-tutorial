"""
helper functions for easyocr app
"""

import cv2

def visualize_results(image_path, results):
    """
    Draws the red box and the text, but skips the probability score.
    """
    img = cv2.imread(image_path)
    # optional resize to increase character size
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if img is None: return None

    # Dynamic scaling for different image sizes
    font_scale = max(0.4, min(img.shape[1] / 1600, 0.7))
    thickness = max(1, int(img.shape[1] / 1200))

    for (bbox, text, _) in results:
        # Convert all corners to integers
        top_left = tuple(map(int, bbox[0]))
        top_right = tuple(map(int, bbox[1]))
        bottom_right = tuple(map(int, bbox[2]))

        # 1. Draw the Red Box (BGR: 0, 0, 255)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
        
        # 2. Draw the TEXT ONLY (no probability)
        # Positioned 10px to the right of the box
        text_pos = (top_right[0] + 10, top_right[1] + 15)
        
        cv2.putText(img, text, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return img