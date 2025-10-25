# ===============================
# train.py - Pelatihan Model
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

print("🚀 Memulai pelatihan model Decision Tree...")

# 1️⃣ Muat dataset
try:
    df = pd.read_csv('diabetes.csv')
    print("✅ Dataset berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: File 'diabetes.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

# 2️⃣ Definisikan fitur dan target
FEATURES = ['kehamilan', 'glukosa', 'tekanan_darah', 'ketebalan_kulit',
            'insulin', 'BMI', 'fungsi_silsilah_diabetes', 'usia']
X = df[FEATURES]
y = df['hasil']

# 3️⃣ Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4️⃣ Latih model Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Simpan model
MODEL_FILENAME = 'model_diabetes.pkl'
joblib.dump(model, MODEL_FILENAME)

print(f"✅ Model berhasil disimpan sebagai '{MODEL_FILENAME}'.")
