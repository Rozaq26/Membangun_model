import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Atur URI tracking dan nama eksperimen. Ini tetap diperlukan
# untuk memberitahu MLflow ke mana harus mengirimkan data.
mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment("Diabetes Modeling - SVM")

# Aktifkan autologging untuk scikit-learn
# Ini akan secara otomatis mencatat parameter, metrik, dan model.
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

# Muat dataset Anda
try:
    X_train = pd.read_csv("dataset_preprocessing/x_train.csv")
    X_test = pd.read_csv("dataset_preprocessing/x_test.csv")
    y_train = pd.read_csv("dataset_preprocessing/y_train.csv")
    y_test = pd.read_csv("dataset_preprocessing/y_test.csv")
except FileNotFoundError as e:
    print(f"Error: Pastikan file dataset ada di folder 'dataset_preprocessing/'.\n{e}")
    # Keluar jika file tidak ditemukan
    exit()

# Mulai MLflow run dengan nama yang spesifik
with mlflow.start_run(run_name="SVM_Model"):
    # Buat dan latih model
    model = SVC()
    # Panggilan .fit() ini akan secara otomatis memicu MLflow
    # untuk mencatat semua informasi yang relevan.
    model.fit(X_train, y_train.values.ravel())

    # Evaluasi model pada data uji
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Anda masih bisa mencetak metrik secara manual jika ingin melihatnya di konsol.
    # Metrik ini juga sudah dicatat secara otomatis oleh autolog.
    print(f"Akurasi model (default SVC): {accuracy:.4f}")
    print("\nMLflow autologging selesai.")
    print(f"Cek UI MLflow di {mlflow.get_tracking_uri()} untuk melihat hasilnya.")

