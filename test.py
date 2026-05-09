import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# MENGGUNAKAN UTILS (Pastikan Anda sudah memindahkan fungsi ke utils.py)
try:
    from utils import apply_spatial_filter, create_lenet, load_and_preprocess
except ImportError:
    print("Error: File 'utils.py' tidak ditemukan. Pastikan fungsi ada di utils.py")

# Optimasi log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Optimasi log TensorFlow agar terminal lebih bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import fungsi dari utils.py
from train import (
    load_and_preprocess, 
    create_lenet, 
    compile_model, 
    get_model_summary
)

# ==================== KONFIGURASI ====================
# Sesuaikan path ini dengan lokasi dataset Anda
DATA_DIR = "flowers_dataset"  # Ganti dengan path folder dataset Anda
MODEL_SAVE_PATH = "best_lenet_flower_model.h5"
PLOT_SAVE_PATH = "training_history.png"

# Hyperparameters
IMG_SIZE = (32, 32)          # Ukuran input gambar
BATCH_SIZE = 16              # Batch size (lebih kecil karena data sedikit)
EPOCHS = 30                  # Maksimum epoch
VALIDATION_SPLIT = 0.2       # 20% data untuk validasi
FILTER_TYPE = "gaussian"     # Jenis filter: 'gaussian', 'median', 'sharpen', atau None
LEARNING_RATE = 0.001
SAMPLES_PER_CLASS = 50       # <--- AMBIL 50 FOTO PER KELAS

# ==================== 1. LOAD DAN PREPROCESS DATA ====================
print("="*60)
print("PROYEK: IMPLEMENTASI LeNet UNTUK KLASIFIKASI 5 JENIS BUNGA")
print("DENGAN PREPROCESSING FILTERING SPASIAL")
print("="*60)
print(f"\nKonfigurasi:")
print(f"  - Dataset path: {DATA_DIR}")
print(f"  - Gambar per kelas: {SAMPLES_PER_CLASS} (total {SAMPLES_PER_CLASS*5} gambar)")
print(f"  - Ukuran gambar: {IMG_SIZE}")
print(f"  - Filter spasial: {FILTER_TYPE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")

# Cek apakah folder dataset ada
if not os.path.exists(DATA_DIR):
    print(f"\nERROR: Folder dataset '{DATA_DIR}' tidak ditemukan!")
    print("Silakan periksa path DATA_DIR dan pastikan folder 'flowers_dataset' ada.")
    sys.exit(1)

# Load data dengan preprocessing filtering (hanya 50 gambar per kelas)
print("\n" + "="*60)
print("TAHAP 1: Load dan Preprocessing Dataset (50 gambar per kelas)")
print("="*60)

X, y, class_names = load_and_preprocess(
    data_dir=DATA_DIR,
    img_size=IMG_SIZE,
    filter_type=FILTER_TYPE,
    samples_per_class=SAMPLES_PER_CLASS  # <--- Parameter baru
)

num_classes = len(class_names)
print(f"\nJumlah kelas: {num_classes}")
print(f"Label kelas: {class_names}")

# ==================== 2. SPLIT DATA TRAIN/VALIDATION ====================
print("\n" + "="*60)
print("TAHAP 2: Split Data Training dan Validasi")
print("="*60)

# One-hot encoding untuk label
y_categorical = to_categorical(y, num_classes=num_classes)

# Split data (stratify agar distribusi kelas seimbang)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, 
    test_size=VALIDATION_SPLIT, 
    random_state=42, 
    stratify=y
)

print(f"Training data: {X_train.shape[0]} gambar")
print(f"Validation data: {X_val.shape[0]} gambar")
print(f"Input shape: {X_train.shape[1:]}")
print(f"Output shape: {y_train.shape[1]} kelas")

# Hitung jumlah per kelas di training dan validation
train_labels = np.argmax(y_train, axis=1)
val_labels = np.argmax(y_val, axis=1)

print("\nDistribusi training:")
for i, class_name in enumerate(class_names):
    count = np.sum(train_labels == i)
    print(f"  {class_name}: {count} gambar")

print("\nDistribusi validation:")
for i, class_name in enumerate(class_names):
    count = np.sum(val_labels == i)
    print(f"  {class_name}: {count} gambar")

# ==================== 3. BANGUN MODEL LENET ====================
print("\n" + "="*60)
print("TAHAP 3: Membangun Arsitektur LeNet")
print("="*60)

# Buat model LeNet
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
model = create_lenet(input_shape=input_shape, num_classes=num_classes)

# Compile model
model = compile_model(model, learning_rate=LEARNING_RATE)

# Tampilkan summary model
print("\nArsitektur Model LeNet:")
print(get_model_summary(model))

# ==================== 4. SETUP CALLBACKS ====================
print("\n" + "="*60)
print("TAHAP 4: Setup Callbacks")
print("="*60)

# Early stopping: hentikan training jika validation loss tidak membaik
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,                # Lebih kecil karena data sedikit
    verbose=1,
    restore_best_weights=True
)

# Model checkpoint: simpan model terbaik berdasarkan validation accuracy
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Reduce learning rate jika loss plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=0.00001
)

callbacks = [early_stop, checkpoint, reduce_lr]

# ==================== 5. TRAINING MODEL ====================
print("\n" + "="*60)
print("TAHAP 5: Memulai Training Model")
print("="*60)
print(f"Mulai training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total data training: {len(X_train)} gambar")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {len(X_train) // BATCH_SIZE}")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print(f"Selesai training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== 6. EVALUASI MODEL ====================
print("\n" + "="*60)
print("TAHAP 6: Evaluasi Model")
print("="*60)

# Evaluasi pada data validasi
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Accuracy (%): {val_accuracy*100:.2f}%")

# ==================== 7. PLOT HISTORY TRAINING ====================
print("\n" + "="*60)
print("TAHAP 7: Visualisasi Hasil Training")
print("="*60)

# Buat figure dengan 2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', marker='o', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Tambahkan informasi akurasi akhir di judul
best_acc = max(history.history['val_accuracy'])
best_epoch = np.argmax(history.history['val_accuracy']) + 1
fig.suptitle(f'LeNet Training (50 gambar/kelas)\nBest Validation Accuracy: {best_acc:.2%} (Epoch {best_epoch})', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"Plot tersimpan di: {PLOT_SAVE_PATH}")
plt.show()

# ==================== 8. SIMPAN INFORMASI TRAINING ====================
print("\n" + "="*60)
print("TAHAP 8: Menyimpan Ringkasan Training")
print("="*60)

# Simpan ringkasan training ke file teks
summary_file = "training_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("RINGKASAN TRAINING LENET UNTUK KLASIFIKASI BUNGA\n")
    f.write(f"(Menggunakan {SAMPLES_PER_CLASS} gambar per kelas)\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Dataset: {DATA_DIR}\n")
    f.write(f"Filter spasial: {FILTER_TYPE}\n")
    f.write(f"Sampel per kelas: {SAMPLES_PER_CLASS}\n")
    f.write(f"Total gambar: {len(X)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Validation samples: {len(X_val)}\n")
    f.write(f"Input shape: {input_shape}\n")
    f.write(f"Jumlah kelas: {num_classes}\n")
    f.write(f"Kelas: {', '.join(class_names)}\n\n")
    
    f.write(f"Hyperparameters:\n")
    f.write(f"  - Batch size: {BATCH_SIZE}\n")
    f.write(f"  - Epochs: {EPOCHS}\n")
    f.write(f"  - Learning rate: {LEARNING_RATE}\n\n")
    
    f.write(f"Hasil terbaik:\n")
    f.write(f"  - Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
    f.write(f"  - Best epoch: {best_epoch}\n\n")
    
    f.write(f"Hasil akhir:\n")
    f.write(f"  - Validation Loss: {val_loss:.4f}\n")
    f.write(f"  - Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n\n")
    
    f.write(f"Model terbaik disimpan di: {MODEL_SAVE_PATH}\n")
    f.write(f"Plot training disimpan di: {PLOT_SAVE_PATH}\n")

print(f"Ringkasan training tersimpan di: {summary_file}")

# ==================== 9. TAMPILKAN PREDIKSI CONTOH ====================
print("\n" + "="*60)
print("TAHAP 9: Contoh Prediksi pada Validation Data")
print("="*60)

# Ambil 5 sampel dari validation set
n_samples = min(5, len(X_val))
sample_indices = np.random.choice(len(X_val), n_samples, replace=False)

for idx in sample_indices:
    img = X_val[idx]
    true_label = class_names[np.argmax(y_val[idx])]
    
    # Prediksi
    pred_prob = model.predict(img.reshape(1, *img.shape), verbose=0)
    pred_label = class_names[np.argmax(pred_prob)]
    confidence = np.max(pred_prob) * 100
    
    # Tampilkan gambar (denormalisasi untuk display)
    img_display = (img * 255).astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    plt.title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
              fontsize=10, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==================== 10. SELESAI ====================
print("\n" + "="*60)
print("TRAINING SELESAI!")
print("="*60)
print(f"\n✓ Model terbaik disimpan di: {MODEL_SAVE_PATH}")
print(f"✓ Plot history training di: {PLOT_SAVE_PATH}")
print(f"✓ Ringkasan training di: {summary_file}")
print(f"\nStatistik akhir:")
print(f"  - Total data training: {len(X_train)} gambar")
print(f"  - Total data validasi: {len(X_val)} gambar")
print(f"  - Best validation accuracy: {best_acc:.2%}")
print("\nLangkah selanjutnya:")
print("1. Jalankan file test.py untuk menguji model dengan gambar baru")
print("2. Gunakan model yang sudah disimpan untuk prediksi")
