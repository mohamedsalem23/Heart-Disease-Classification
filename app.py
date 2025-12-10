# app.py - Fixed Version

import os
import sys

# Suppress warnings early
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

# =================================================================
# 1. Page Configuration
# =================================================================

st.set_page_config(
    page_title="Heart Disease Classifier", 
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
try:
    import classifier
    from db_manager import load_embeddings_from_sqlite
except ImportError as e:
    st.error(f"❌ خطأ في استيراد الوحدات: {e}")
    st.stop()

# =================================================================
# 2. Load Resources (Models & Database)
# =================================================================

@st.cache_resource
def load_resources():
    """Load models and train SVM classifiers once."""
    try:
        # Load model
        model, processor, device = classifier.initialize_clip_model()
        all_embeddings_df = load_embeddings_from_sqlite()
        
        if model is None or all_embeddings_df is None or all_embeddings_df.empty:
            st.error("❌ فشل تحميل المصادر الأساسية")
            return None, None, None, None, None, None, None
        
        # Train multi-modal models
        visual_svm, clinical_svm, scaler, feature_cols = classifier.train_multi_modal_models(all_embeddings_df)
        
        if visual_svm is None:
            st.error("❌ فشل تدريب نماذج SVM")
            return model, processor, device, None, None, None, None
        
        return model, processor, device, visual_svm, clinical_svm, scaler, feature_cols
        
    except Exception as e:
        st.error(f"❌ خطأ في التحميل: {str(e)[:100]}")
        return None, None, None, None, None, None, None

# Load models
model, processor, device, visual_svm, clinical_svm, scaler, feature_cols = load_resources()

# =================================================================
# 3. Classification Function
# =================================================================

def run_classification(uploaded_files, clinical_data):
    """Run multi-modal classification."""
    
    if visual_svm is None or clinical_svm is None:
        return None, "فشل التصنيف: النماذج غير جاهزة"
    
    # Save images temporarily
    image_paths = []
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, f in enumerate(uploaded_files):
        if f is None:
            continue
        temp_path = os.path.join(temp_dir, f"temp_{i}_{f.name}")
        with open(temp_path, "wb") as fp:
            fp.write(f.getbuffer())
        image_paths.append(temp_path)
    
    if not image_paths:
        return None, "لم يتم رفع أي صور"
    
    # Run classification
    try:
        classification, confidence = classifier.classify_new_case_multi_modal(
            image_paths=image_paths,
            clinical_data=clinical_data,
            model=model,
            processor=processor,
            device=device,
            visual_svm=visual_svm,
            clinical_svm=clinical_svm,
            scaler=scaler,
            feature_cols=feature_cols
        )
        return (classification, confidence), None
    except Exception as e:
        return None, f"خطأ في التصنيف: {str(e)[:100]}"
    finally:
        # Cleanup
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

# =================================================================
# 4. UI Layout
# =================================================================

# Add doctor's name at the top
st.markdown("""
<div style='text-align: center; color: #e74c3c; font-size: 28px; font-weight: bold; margin-bottom: 20px;'>
    <strong>Dr. Mohammad El Tahlawi</strong>
</div>
""", unsafe_allow_html=True)

st.title("❤️ نظام تصنيف أمراض القلب بالذكاء الاصطناعي")
st.markdown("---")

col_input, col_result = st.columns([3, 2])

# ===== INPUT COLUMN =====
with col_input:
    st.header("📋 إدخال البيانات")
    
    # Image upload
    st.subheader("🖼️ صور الإيكو")
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        uploaded_file1 = st.file_uploader(
            "الصورة الأولى:",
            type=["jpg", "jpeg", "png"],
            key="file1"
        )
        if uploaded_file1:
            st.image(uploaded_file1, caption="الصورة الأولى", use_column_width=True)
    
    with img_col2:
        uploaded_file2 = st.file_uploader(
            "الصورة الثانية:",
            type=["jpg", "jpeg", "png"],
            key="file2"
        )
        if uploaded_file2:
            st.image(uploaded_file2, caption="الصورة الثانية", use_column_width=True)
    
    uploaded_files = [f for f in [uploaded_file1, uploaded_file2] if f is not None]
    
    # Clinical data
    st.subheader("🏥 البيانات السريرية")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input(
            "العمر (سنة):",
            min_value=1,
            max_value=120,
            value=65,
            step=1
        )
        bsa = st.number_input(
            "BSA (m²):",
            min_value=0.5,
            max_value=3.0,
            value=1.8,
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        bmi = st.number_input(
            "BMI (kg/m²):",
            min_value=15.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
            format="%.1f"
        )
        lav = st.number_input(
            "LAV (ml/m²):",
            min_value=5.0,
            max_value=200.0,
            value=30.0,
            step=1.0,
            format="%.1f"
        )
    
    with col3:
        lavi = st.number_input(
            "LAVI (g/m²):",
            min_value=5.0,
            max_value=200.0,
            value=100.0,
            step=1.0,
            format="%.1f"
        )
        hypertension = st.checkbox("ارتفاع ضغط الدم", value=True)
    
    smoking = st.checkbox("التدخين", value=False)
    
    # Prepare clinical data
    clinical_data = {
        'Age': float(age),
        'BSA': float(bsa),
        'BMI': float(bmi),
        'Hypertention': 1 if hypertension else 0,
        'Smoking': 1 if smoking else 0,
        'LAV': float(lav),
        'LAVI': float(lavi)
    }

# ===== RESULT COLUMN =====
with col_result:
    st.header("📊 النتيجة")
    
    if st.button("🚀 تصنيف الحالة", type="primary", use_container_width=True):
        
        if not uploaded_files:
            st.warning("⚠️ الرجاء رفع صورة واحدة على الأقل")
        elif visual_svm is None:
            st.info("⏳ جاري تحميل النماذج... يرجى الانتظار")
        else:
            with st.spinner("🔄 جاري المعالجة..."):
                result, error = run_classification(uploaded_files, clinical_data)
            
            st.markdown("---")
            
            if error:
                st.error(f"❌ {error}")
            elif result:
                classification, confidence = result
                
                st.success("✅ تم التصنيف بنجاح!")
                
                # Display result
                st.markdown(f"""
                ## النتيجة: **{classification.upper()}**
                ### مستوى الثقة: **{confidence:.1f}%**
                """)
                
                # Display used factors
                st.markdown("---")
                st.info(f"""
                **العوامل المستخدمة (40% من القرار):**
                
                • العمر: **{age}** سنة
                • BSA: **{bsa}** m²
                • BMI: **{bmi}** kg/m²
                • LAV: **{lav}** ml/m²
                • LAVI: **{lavi}** g/m²
                • ضغط الدم: **{'✓' if hypertension else '✗'}**
                • التدخين: **{'✓' if smoking else '✗'}**
                """)

# =================================================================
# 5. Sidebar Info
# =================================================================

with st.sidebar:
    st.header("ℹ️ معلومات")
    st.info("""
    **نظام تصنيف متعدد الأنماط:**
    - 60% من القرار: تحليل الصور
    - 40% من القرار: البيانات السريرية
    
    **تنبيه طبي:**
    هذا النظام لأغراض بحثية فقط.
    """)
    
    st.markdown("---")
    
    if st.checkbox("عرض معلومات تقنية"):
        st.write(f"""
        **حالة النموذج:**
        - Visual SVM: {'✓' if visual_svm is not None else '✗'}
        - Clinical SVM: {'✓' if clinical_svm is not None else '✗'}
        - الميزات المستخدمة: {feature_cols if feature_cols else 'لا توجد'}
        """)
