import os
import shutil
import json
import base64
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from shapely.geometry import Polygon
import re

# --- Configuration ---
HUNYUAN_API_URL = "http://localhost:8000/v1"
TABLE_MODEL_PATH = "/home/ubuntu/chen/ocr/table/none_line"
# For wired table, if there is a model path, set it here. 
# For now, we only implemented lineless table in previous turns. 
# If 'Table' means wired table, we might need another model or logic.
# The user asked for "Hunyuanocr, Table and NoTable".
# 'Table' might refer to wired table structure recognition or just table recognition in general.
# Assuming 'NoTable' means Lineless Table (LORE) as requested previously.
# Assuming 'Table' might mean standard table recognition or wired.
# Let's assume 'Table' logic is placeholder or uses LORE too if not specified, 
# but usually 'Table' implies standard wired tables. 
# Since we don't have a wired table model path specified in previous context, 
# I will use LORE for 'NoTable' and maybe just HunyuanOCR for 'Hunyuanocr' (pure OCR).
# If user wants 'Table' (wired), I'll check if we have a model or use LORE as fallback/generic.
# Actually, let's keep it simple:
# - Hunyuanocr: Pure OCR
# - Table: Wired Table (placeholder or specific model if known, otherwise warn)
# - NoTable: Lineless Table (LORE)

# Initialize OpenAI client
client = OpenAI(
    api_key="EMPTY",
    base_url=HUNYUAN_API_URL,
    timeout=3600
)

# Global model cache
_lore_pipeline = None
_wired_pipeline = None
WIRED_TABLE_MODEL_PATH = "/home/ubuntu/chen/ocr/table/line"

def get_lore_pipeline():
    global _lore_pipeline
    if _lore_pipeline is None:
        print(f"Loading LORE model from {TABLE_MODEL_PATH}...")
        try:
            _lore_pipeline = pipeline(Tasks.lineless_table_recognition, model=TABLE_MODEL_PATH)
        except Exception as e:
            print(f"Failed to load local LORE model: {e}")
            try:
                _lore_pipeline = pipeline(Tasks.lineless_table_recognition, model='iic/cv_resnet-transformer_table-structure-recognition_lore')
            except Exception as e2:
                print(f"Failed to load online LORE model: {e2}")
                _lore_pipeline = None
    return _lore_pipeline

def get_wired_pipeline():
    global _wired_pipeline
    if _wired_pipeline is None:
        print(f"Loading Wired Table model from {WIRED_TABLE_MODEL_PATH}...")
        try:
            # Explicitly use device='gpu' or 'cpu' if needed, but default auto is usually fine.
            # However, logs show ModelScope might hang on preprocessor config.
            # Let's try to catch it more gracefully or print more logs.
            _wired_pipeline = pipeline(Tasks.table_recognition, model=WIRED_TABLE_MODEL_PATH)
            print("Wired Table model loaded successfully.")
        except Exception as e:
            print(f"Failed to load local Wired Table model: {e}")
            try:
                _wired_pipeline = pipeline(Tasks.table_recognition, model='iic/cv_dla34_table-structure-recognition_cycle-centernet')
                print("Online Wired Table model loaded successfully.")
            except Exception as e2:
                 print(f"Failed to load online Wired Table model: {e2}")
                 _wired_pipeline = None
    return _wired_pipeline

# --- Helper Functions ---

def encode_image_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def parse_ocr_result(text, image_size):
    img_w, img_h = image_size
    parsed_data = []
    pattern = re.compile(r'\((-?\d+),(-?\d+)\),\((-?\d+),(-?\d+)\)')
    
    try:
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        data = json.loads(cleaned_text)
        items = data if isinstance(data, list) else data.get("data", [])
        
        for item in items:
            content = item.get("text", "")
            box = item.get("bbox", [])
            if len(box) == 4:
                x1, y1, x2, y2 = box
                x1 = int(x1 * img_w / 1000)
                y1 = int(y1 * img_h / 1000)
                x2 = int(x2 * img_w / 1000)
                y2 = int(y2 * img_h / 1000)
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                parsed_data.append({"text": content, "box": points})
        return parsed_data
    except:
        pass

    matches = list(pattern.finditer(text))
    start_index = 0
    for match in matches:
        end_index = match.start()
        content = text[start_index:end_index].strip()
        x1, y1, x2, y2 = map(int, match.groups())
        x1 = int(x1 * img_w / 1000)
        y1 = int(y1 * img_h / 1000)
        x2 = int(x2 * img_w / 1000)
        y2 = int(y2 * img_h / 1000)
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        parsed_data.append({"text": content, "box": points})
        start_index = match.end()
    return parsed_data

def get_hunyuan_ocr(image_path, prompt):
    base64_img = encode_image_file(image_path)
    image_size = get_image_size(image_path)
    
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    try:
        response = client.chat.completions.create(
            model="tencent/HunyuanOCR",
            messages=messages,
            temperature=0.0,
            top_p=0.95,
            stream=False,
            max_tokens=4096
        )
        content = response.choices[0].message.content.strip()
        return parse_ocr_result(content, image_size)
    except Exception as e:
        print(f"HunyuanOCR Error: {e}")
        return []

def get_lore_structure(image_path):
    pipeline = get_lore_pipeline()
    if pipeline is None:
        return []
    try:
        result = pipeline(image_path)
        cells = []
        if 'polygons' in result:
            for poly in result['polygons']:
                points = np.array(poly).reshape(-1, 2).tolist()
                cells.append(points)
        return cells
    except Exception as e:
        print(f"LORE Error: {e}")
        return []

def get_wired_structure(image_path):
    pipeline = get_wired_pipeline()
    if pipeline is None:
        return []
    try:
        # Use PIL image for pipeline as in merge_ocr_table.py
        img = Image.open(image_path)
        result = pipeline(img)
        cells = []
        if 'polygons' in result:
            for poly in result['polygons']:
                points = np.array(poly).reshape(-1, 2).tolist()
                cells.append(points)
        return cells
    except Exception as e:
        print(f"Wired Table Error: {e}")
        return []

def calculate_iou(box1, box2):
    try:
        poly1 = Polygon(box1)
        poly2 = Polygon(box2)
        if not poly1.is_valid: poly1 = poly1.buffer(0)
        if not poly2.is_valid: poly2 = poly2.buffer(0)
        intersection = poly1.intersection(poly2).area
        text_area = poly1.area
        if text_area == 0: return 0
        return intersection / text_area
    except:
        return 0

def merge_results(ocr_results, table_cells, iou_threshold=0.8):
    merged_data = []
    used_indices = set()
    cell_contents = {i: [] for i in range(len(table_cells))}
    
    for i, text_item in enumerate(ocr_results):
        text_box = text_item['box']
        best_iou = 0
        best_cell_idx = -1
        for j, cell_box in enumerate(table_cells):
            iou = calculate_iou(text_box, cell_box)
            if iou > best_iou:
                best_iou = iou
                best_cell_idx = j
        
        if best_iou >= iou_threshold:
            cell_contents[best_cell_idx].append(i)
            used_indices.add(i)
    
    for cell_idx, text_indices in cell_contents.items():
        if not text_indices: continue
        cell_texts = [ocr_results[i] for i in text_indices]
        cell_texts.sort(key=lambda x: (x['box'][0][1], x['box'][0][0]))
        merged_text = "".join([t['text'] for t in cell_texts])
        
        all_points = []
        for t in cell_texts: all_points.extend(t['box'])
        all_points = np.array(all_points)
        min_x, min_y = np.min(all_points[:, 0]), np.min(all_points[:, 1])
        max_x, max_y = np.max(all_points[:, 0]), np.max(all_points[:, 1])
        
        merged_box = [[int(min_x), int(min_y)], [int(max_x), int(min_y)], 
                      [int(max_x), int(max_y)], [int(min_x), int(max_y)]]
        
        merged_data.append({
            "text": merged_text,
            "box": merged_box,
            "is_table_cell": True,
            "cell_id": cell_idx
        })
        
    for i, text_item in enumerate(ocr_results):
        if i not in used_indices:
            text_item['is_table_cell'] = False
            merged_data.append(text_item)
            
    return merged_data

def visualize_results(image_path, results, output_path, table_cells=None):
    image = cv2.imread(image_path)
    if image is None: return

    if table_cells:
        for cell in table_cells:
            box = np.array(cell, dtype=np.int32)
            cv2.polylines(image, [box], isClosed=True, color=(255, 0, 0), thickness=1)

    for item in results:
        box = np.array(item['box'], dtype=np.int32)
        is_table = item.get('is_table_cell', False)
        color = (0, 255, 0) if is_table else (0, 0, 255)
        cv2.polylines(image, [box], isClosed=True, color=color, thickness=2)
        
    cv2.imwrite(output_path, image)

def process_image(image_path, mode, prompt, iou_threshold, output_vis_path):
    print(f"Starting OCR process for {image_path} in mode {mode}")
    # 1. HunyuanOCR
    ocr_results = get_hunyuan_ocr(image_path, prompt)
    print(f"HunyuanOCR completed, detected {len(ocr_results)} segments.")
    
    if mode == "Hunyuanocr":
        # Pure OCR
        visualize_results(image_path, ocr_results, output_vis_path)
        return ocr_results
        
    elif mode == "NoTable":
        # Lineless Table (LORE)
        print("Running LORE structure recognition...")
        table_cells = get_lore_structure(image_path)
        print(f"LORE structure recognition completed, detected {len(table_cells)} cells.")
        merged_results = merge_results(ocr_results, table_cells, iou_threshold)
        visualize_results(image_path, merged_results, output_vis_path, table_cells)
        return merged_results
        
    elif mode == "Table":
        # Wired Table
        print("Running Wired Table structure recognition...")
        table_cells = get_wired_structure(image_path)
        print(f"Wired Table structure recognition completed, detected {len(table_cells)} cells.")
        merged_results = merge_results(ocr_results, table_cells, iou_threshold)
        visualize_results(image_path, merged_results, output_vis_path, table_cells)
        return merged_results
        
    return ocr_results
