# data_processor.py

import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoProcessor, AutoModel 
from PIL import Image
import pandas as pd
from pathlib import Path
from config import MODEL_NAME, BASE_DIR, EXCEL_PATH
import numpy as np

# =================================================================
# 1. تهيئة النموذج (Initialization)
# =================================================================

def initialize_clip_model():
    """
    تحميل نموذج Vision Transformer والمعالج (AutoModel/AutoProcessor) وتحديد الجهاز (CPU/GPU).
    """
    print(f"Loading Specialized Model: {MODEL_NAME}...")
    try:
        model = AutoModel.from_pretrained(MODEL_NAME)
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded successfully on device: {device}")
        
        try:
            embedding_size = model.config.hidden_size 
            print(f"Expected Embedding Size: {embedding_size}")
        except AttributeError:
            print("Warning: Could not determine embedding size automatically. Assuming 768.")

        return model, processor, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# =================================================================
# 2. تحميل المسارات والصور
# =================================================================

def load_image_paths(base_dir):
    """
    تحليل هيكل المجلدات لاستخراج مسار كل صورة وفئة تصنيفها.
    """
    print(f"Scanning directory: {base_dir}")
    data_list = []
    base_path = Path(base_dir)
    
    for classification_folder_path in base_path.iterdir():
        
        if classification_folder_path.is_dir():
            
            classification_folder_name = classification_folder_path.name
            print(f"-> Found classification folder: {classification_folder_name}")
            
            for patient_folder in classification_folder_path.iterdir():
                
                if patient_folder.is_dir():
                    patient_id = patient_folder.name
                    
                    for image_file in patient_folder.glob("*.jpg"):
                        data_list.append({
                            "file_path": str(image_file),
                            "classification_grade": classification_folder_name,
                            "patient_id": patient_id
                        })
                        
    print(f"Found {len(data_list)} images for processing.")
    return pd.DataFrame(data_list)

# =================================================================
# 3. استخراج الـ Embeddings
# =================================================================

def extract_embeddings(df, model, processor, device):
    """
    تمرير الصور عبر نموذج Swin Transformer واستخراج متجهات الملامح (Embeddings).
    """
    embeddings = []
    batch_size = 32
    
    try:
        embedding_size = model.config.hidden_size
    except AttributeError:
        embedding_size = 768 

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        image_paths = batch_df['file_path'].tolist()
        
        try:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                # استخراج متجهات الـ Embedding: متوسط الـ Hidden State
                image_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
            # التعديل الهام: استخدام list() لضمان التوافق مع DataFrame
            embeddings.extend(list(image_features)) 
            
        except Exception as e:
            print(f"Error processing batch starting at index {i} ({image_paths[0]}): {e}")
            embeddings.extend([np.zeros(embedding_size)] * len(batch_df)) 
            
    df['embedding'] = embeddings
    return df

# =================================================================
# 4. دمج بيانات Excel
# =================================================================

def merge_excel_data(df):
    """
    قراءة ملف Excel ودمجه مع DataFrame الخاص بالصور.
    """
    try:
        excel_df = pd.read_excel(EXCEL_PATH)
        
        patient_id_col = next((col for col in excel_df.columns if 'patient' in col.lower()), None)
        
        if patient_id_col:
             excel_df['patient_id'] = excel_df[patient_id_col].astype(str)
             excel_df = excel_df.drop(columns=[patient_id_col], errors='ignore')

        else:
            print("Warning: Could not find 'patient_id' column in Excel file. Skipping merge.")
            return df.copy()

        print(f"Loaded {len(excel_df)} rows from {EXCEL_PATH}.")
        merged_df = pd.merge(df, excel_df, on='patient_id', how='left')
        print("DataFrames merged successfully.")
        return merged_df
        
    except FileNotFoundError:
        print(f"Warning: Excel file not found at {EXCEL_PATH}. Proceeding without extra data.")
        return df.copy()
    except Exception as e:
        print(f"Error reading or merging Excel data: {e}")
        return df.copy()