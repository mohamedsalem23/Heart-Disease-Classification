# db_manager.py

import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sqlite3
import pandas as pd
import pickle
from config import DB_NAME, TABLE_NAME
import numpy as np

# =================================================================
# 1. تخزين البيانات في SQLite
# =================================================================

def store_data_in_sqlite(df):
    """
    تخزين الـ DataFrame في قاعدة بيانات SQLite، مع تحويل الـ Embeddings إلى BLOB.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # تحويل الـ Embeddings (مصفوفات NumPy) إلى بيانات ثنائية (BLOB) باستخدام pickle
    df['embedding_blob'] = df['embedding'].apply(lambda x: pickle.dumps(x) if x is not None and isinstance(x, np.ndarray) else None)
    
    # تحديد الأعمدة التي سيتم تخزينها
    columns_to_store = ['file_path', 'classification_grade', 'patient_id', 'embedding_blob'] + \
                       [col for col in df.columns if col not in ['file_path', 'classification_grade', 'patient_id', 'embedding_blob', 'embedding']]
    
    # إنشاء جدول جديد بالهيكل الصحيح
    column_definitions = [
        "file_path TEXT PRIMARY KEY",
        "classification_grade TEXT",
        "patient_id TEXT",
        "embedding_blob BLOB"
    ]
    # إضافة الأعمدة الديناميكية الأخرى (من Excel) كـ TEXT
    for col in df.columns:
        if col not in ['file_path', 'classification_grade', 'patient_id', 'embedding_blob', 'embedding']:
            # استخدام علامات الاقتباس حول اسم العمود لحماية الأسماء التي قد تحتوي على مسافات
            column_definitions.append(f"'{col}' TEXT")
            
    create_table_query = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ({', '.join(column_definitions)})"
    
    try:
        cursor.execute(create_table_query)
        conn.commit()
        print(f"Table '{TABLE_NAME}' ensured/created successfully in {DB_NAME}.")
        
        # تخزين البيانات باستخدام to_sql
        df_to_store = df[columns_to_store]
        # يجب تمرير اسم الأعمدة المتوقعة
        df_to_store.columns = [col.replace("'", "") for col in columns_to_store]
        df_to_store.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        
        print(f"Successfully stored {len(df_to_store)} records.")
        
    except Exception as e:
        print(f"Error storing data in SQLite: {e}")
        
    finally:
        conn.close()

# =================================================================
# 2. استرجاع الـ Embeddings من SQLite
# =================================================================

def load_embeddings_from_sqlite():
    """
    قراءة جميع البيانات والـ Embeddings من قاعدة البيانات.
    """
    conn = sqlite3.connect(DB_NAME)
    
    try:
        # قراءة جميع الأعمدة
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        
        # فك تشفير الـ Embeddings من BLOB إلى مصفوفات NumPy
        df['embedding'] = df['embedding_blob'].apply(lambda x: pickle.loads(x) if x is not None else None)
        df = df.drop(columns=['embedding_blob'], errors='ignore')
        
        print(f"Successfully loaded {len(df)} records from database.")
        return df
        
    except pd.io.sql.DatabaseError:
        print("Database or table not found. Run the extraction process first.")
        return pd.DataFrame()
        
    finally:
        conn.close()