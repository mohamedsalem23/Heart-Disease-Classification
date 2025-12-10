# classifier.py

import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import data_processor # يجب استيراد data_processor للحصول على وظيفة extract_embeddings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from PIL import Image
from transformers import AutoProcessor, AutoModel # استيراد النماذج لضمان إمكانية التهيئة داخل الوظيفة المساعدة

# =================================================================
# 1. تدريب نموذج Multi-Modal (البصري والسريري)
# =================================================================

def initialize_clip_model():
    # استدعاء تهيئة النموذج من data_processor
    return data_processor.initialize_clip_model()

def train_multi_modal_models(all_embeddings_df):
    """
    تدريب نموذجين SVM: واحد على Embeddings البصرية وآخر على الأعمدة السريرية.
    """
    
    print("\n--- Starting Multi-Modal Model Training (Visual & Clinical) ---")
    
    # الأعمدة السريرية المحدثة بناءً على ملف Excel الخاص بك:
    POTENTIAL_FEATURE_COLS = [
        'Age', 'BSA', 'BMI', 'Hypertention', 'Smoking', 'LAV', 'LAVI'
    ]
    
    # تصفية الأعمدة التي توجد فعلاً في البيانات
    FEATURE_COLS = [col for col in POTENTIAL_FEATURE_COLS if col in all_embeddings_df.columns]
    
    if not FEATURE_COLS:
        print(f"⚠️ Warning: No clinical feature columns found. Available columns: {list(all_embeddings_df.columns)}")
        FEATURE_COLS = []
    
    # بناء قائمة dropna المناسبة
    cols_to_check = ['embedding', 'classification_grade'] + FEATURE_COLS
    df = all_embeddings_df.dropna(subset=cols_to_check, how='any')

    if df.empty or len(df['classification_grade'].unique()) < 2:
        print("Error: Not enough complete data or classes to train Multi-Modal models.")
        return None, None, None, FEATURE_COLS

    y = df['classification_grade'].values
    
    # --- أ. تدريب النموذج البصري (Visual SVM) ---
    X_visual = np.stack(df['embedding'].values)
    print(f"Training Visual SVM on {X_visual.shape[0]} samples.")
    visual_svm = SVC(kernel='linear', random_state=42, probability=True)
    visual_svm.fit(X_visual, y)
    print("✅ Visual SVM trained successfully.")
    
    # --- ب. تدريب النموذج السريري (Clinical SVM) ---
    X_clinical = df[FEATURE_COLS].values
    
    # توحيد البيانات السريرية (Normalization)
    scaler = StandardScaler()
    X_clinical_scaled = scaler.fit_transform(X_clinical)
    
    print(f"Training Clinical SVM on {X_clinical_scaled.shape[0]} samples.")
    clinical_svm = SVC(kernel='linear', random_state=42, probability=True)
    clinical_svm.fit(X_clinical_scaled, y)
    print("✅ Clinical SVM trained successfully.")

    return visual_svm, clinical_svm, scaler, FEATURE_COLS

# =================================================================
# 2. وظيفة مساعدة لاستخلاص الـ Embedding فقط
# =================================================================

def classify_new_image_svm(image_path, model=None, processor=None, device=None, svm_model=None, return_embedding_only=False):
    """
    وظيفة مساعدة لاستخلاص الـ Embedding فقط لعملية Multi-modal.
    """
    
    if model is None:
        model, processor, device = initialize_clip_model()
        if model is None:
            return "Classification Failed: Model Load Error", None
            
    try:
        new_image_df = pd.DataFrame([{'file_path': image_path}])
        embedded_new_image_df = data_processor.extract_embeddings(new_image_df, model, processor, device)
        new_embedding = embedded_new_image_df['embedding'].iloc[0]
        
        if new_embedding is None or not isinstance(new_embedding, np.ndarray):
            return "Classification Failed: Image Processing Error (Embedding is None)", None
            
    except Exception as e:
        return f"Classification Failed: Image Processing Error ({e})", None

    if return_embedding_only:
        return new_embedding, None 
        
    return "Error: Function used incorrectly.", None

# =================================================================
# 3. تصنيف حالة جديدة باستخدام Weighted Decision
# =================================================================

def classify_new_case_multi_modal(
    image_paths, 
    clinical_data, 
    model, processor, device, 
    visual_svm, clinical_svm, scaler, feature_cols
):
    """
    تنفذ التصنيف بدمج الوزن (60% بصري و 40% سريري).
    """
    
    # 1. استخراج الـ Embedding البصري (متوسط الصور)
    all_embeddings = []
    for path in image_paths:
        embedding, _ = classify_new_image_svm(path, model, processor, device, return_embedding_only=True)
        if isinstance(embedding, np.ndarray):
            all_embeddings.append(embedding)
        else:
            print(f"Skipping failed embedding extraction for {path}")

    if not all_embeddings:
        return "Classification Failed: No valid image embeddings extracted.", 0
        
    average_embedding = np.mean(all_embeddings, axis=0).reshape(1, -1)
    
    # 2. حساب الاحتمالات
    classes = visual_svm.classes_
    
    # P_visual (Visual Probability)
    P_visual = visual_svm.predict_proba(average_embedding)[0]
    
    # إذا لم تكن هناك ميزات سريرية، استخدم النموذج البصري فقط
    if len(feature_cols) == 0 or clinical_svm is None:
        print("Warning: No clinical features available. Using visual model only.")
        P_final = P_visual
    else:
        # تحضير البيانات السريرية
        # معالجة القيم التي قد تكون نصوص بفواصل عشرية
        clinical_values = []
        for col in feature_cols:
            val = clinical_data.get(col, 0)
            # تحويل من string إلى float إذا لزم الأمر
            if isinstance(val, str):
                val = float(val.replace(',', '.'))
            clinical_values.append(val)
        
        clinical_vector = np.array(clinical_values).reshape(1, -1)
        clinical_vector_scaled = scaler.transform(clinical_vector)
        
        # P_clinical (Clinical Probability)
        P_clinical = clinical_svm.predict_proba(clinical_vector_scaled)[0]
        
        # دمج الاحتمالات بالوزن المرجح (60% بصري + 40% سريري)
        P_final = (0.60 * P_visual) + (0.40 * P_clinical)
    
    # 5. اتخاذ القرار النهائي
    final_index = np.argmax(P_final)
    final_classification = classes[final_index]
    confidence = P_final[final_index] * 100
    
    print(f"\n✅ Final Classification: **{final_classification}** (Confidence: {confidence:.2f}%)")

    return final_classification, confidence