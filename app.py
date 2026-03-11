"""
conda create -n easyocr python=3.11 -y
conda activate easyocr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install easyocr
pip install "Pillow<11.0.0" opencv-python  # Ensures compatibility with the ANTIALIAS deprecation
"""

import easyocr
import cv2
import torch
from my_timer import my_timer
from utils import calculate_bow_accuracy_new
from easyocr_utils import visualize_results

# Initialize Reader ONCE globally (or inside the main block) 
# to save VRAM and initialization time.
READER = None

@my_timer
def get_reader(languages=['en']):
    global READER
    if READER is None:
        # load models into memory and enable gpu if detected
        READER = easyocr.Reader(languages, gpu=torch.cuda.is_available())
    return READER

@my_timer
def process_image(image_path):
    # get ocr engine from memory
    reader = get_reader()
    # 3. Run OCR
    # parameters to help find small words - add_margin helps with all files
    return reader.readtext(image_path, text_threshold=0.4, add_margin=0.2)

def save_to_txt(base_filename, results):
    """
    Extracts the text and confidence and saves them to a clean .txt file.
    """
    output_filename = f"ocr_results_{base_filename.split('.')[0]}.txt"
    text_results = []
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"OCR Results for: {base_filename}\n")
        f.write("-" * 30 + "\n")
        
        for (_, text, prob) in results:
            # Saving in [0.985] TEXT format
            f.write(f"[{prob:.3f}] {text}\n")
            text_results.append(text)
            
    print(f"Results successfully saved to: {output_filename}")
    return text_results

# load text from ground truth file for comparision to model prediction
def get_text_from_file(file_path):
    """Reads the ground truth text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

if __name__ == "__main__":

    file_pairs = [
        {"image": "table1.png", "truth_file": "ground_truth/ground_truth_table1.txt", "prompt": "<image>\nParse all charts and tables. Extract data as HTML tables."}, 
        {"image": "invoice1.png", "truth_file": "ground_truth/ground_truth_invoice1.txt", "prompt": "<image>\nConvert this entire document to json"}, 
        {"image": "dl1.png", "truth_file": "ground_truth/ground_truth_dl1.txt", "prompt": "<image>\nConvert the document to markdown."}, 
        {"image": "hw_clean.jpg", "truth_file": "ground_truth/ground_truth_hw_clean.txt", "prompt": "<image>What text is in this image of class notes? Do not guess, do not correct spelling, do not normalize the grammar, and maintain the original line breaks. Keep the case for each letter (upper/lower). If a word is illegible, represent it with [illegible].\n"}, 
        {"image": "document.jpg", "truth_file": "ground_truth/ground_truth_document.txt", "prompt": "<image>\nConvert this entire document to markdown"}, 
    ]

    for pair in file_pairs:
        img_path = pair['image']
        txt_path = pair['truth_file']
        prompt = pair['prompt']
    
        # 1. Process
        print(f"Running OCR on {img_path}...")
        ground_truth = get_text_from_file(txt_path)
        prediction = process_image(img_path)

        prediction_text = save_to_txt(img_path, prediction)

        bow_acc, hit_list = calculate_bow_accuracy_new(" ".join(prediction_text), ground_truth)
        print(f"extracting from {img_path} - bag of words accuracy: {bow_acc}")

        # 2. Visualize (Box + Text Only)
        annotated_img = visualize_results(img_path, prediction)

        if annotated_img is not None:
            # Save output
            output_name = f"result_annotated_{img_path.split('.')[0]}.png"
            cv2.imwrite(output_name, annotated_img)
            print(f"Saved: {output_name}")
            
            # Show result
            cv2.imshow('Clean Box + Text View', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()