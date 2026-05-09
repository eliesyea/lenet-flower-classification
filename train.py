import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import os
import random
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def apply_spatial_filter(image, filter_type='gaussian'):
    """
    Menerapkan filtering spasial pada gambar.
    
    Parameters:
    - image: gambar dalam format numpy array (BGR atau RGB)
    - filter_type: 'gaussian', 'median', atau 'sharpen'
    
    Returns:
    - filtered: gambar yang sudah difilter
    """
    if filter_type == 'gaussian':
        filtered = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'median':
        filtered = cv2.medianBlur(image, 5)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        filtered = cv2.filter2D(image, -1, kernel)
    else:
        filtered = image
    
    return filtered

def load_and_preprocess(data_dir, img_size=(32, 32), filter_type='gaussian', samples_per_class=None):
    """
    Membaca dataset, melakukan preprocessing, filtering, dan resize.
    
    Parameters:
    - data_dir: path ke folder dataset (berisi subfolder kelas)
    - img_size: tuple (height, width) untuk resize
    - filter_type: jenis filter yang akan diaplikasikan
    - samples_per_class: jumlah gambar yang diambil per kelas (None = ambil semua)
    
    Returns:
    - images: numpy array berisi gambar yang sudah diproses
    - labels: numpy array berisi label (integer)
    - class_names: list nama kelas sesuai urutan folder
    """
    images = []
    labels = []
    
    # Ambil semua nama kelas berdasarkan nama folder
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    if samples_per_class:
        print(f"Dataset ditemukan: {len(class_names)} kelas")
        print(f"Kelas: {class_names}")
        print(f"Mengambil {samples_per_class} gambar per kelas")
    else:
        print(f"Dataset ditemukan: {len(class_names)} kelas")
        print(f"Kelas: {class_names}")
        print(f"Mengambil SEMUA gambar")
    
    print("Memuat dan memproses gambar...")
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        # Ambil semua file gambar
        all_images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Jika samples_per_class ditentukan, ambil sejumlah itu
        if samples_per_class and samples_per_class < len(all_images):
            random.shuffle(all_images)
            selected_images = all_images[:samples_per_class]
        else:
            selected_images = all_images
        
        if samples_per_class:
            print(f"  Memproses kelas '{class_name}': mengambil {len(selected_images)} dari {len(all_images)} gambar")
        else:
            print(f"  Memproses kelas '{class_name}': {len(selected_images)} gambar")
        
        for img_file in selected_images:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"    Peringatan: Gagal membaca {img_file}")
                continue
            
            # Resize gambar
            img = cv2.resize(img, img_size)
            
            # Terapkan filtering spasial
            img = apply_spatial_filter(img, filter_type)
            
            # Normalisasi pixel ke range [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(class_map[class_name])
    
    # Konversi ke numpy array
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\nTotal gambar yang berhasil dimuat: {len(images)}")
    print(f"Shape gambar: {images.shape}")

    # Acak dataset
    images, labels = shuffle(images, labels, random_state=42)

    # Tampilkan distribusi label
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nDistribusi label:")

    # Tampilkan distribusi label
    unique, counts = np.unique(labels, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"  {class_names[class_idx]}: {count} gambar")
    
    return images, labels, class_names

def create_lenet(input_shape=(32, 32, 3), num_classes=5):
    """
    Membangun arsitektur LeNet-5 yang dimodifikasi untuk RGB.
    """
    model = Sequential([
        Conv2D(32, (5,5), activation='relu', padding='same'),
        MaxPooling2D((2,2)),

        Conv2D(64, (5,5), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.0001):
    """
    Mengcompile model dengan optimizer, loss, dan metrics.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_model_summary(model):
    """
    Mendapatkan ringkasan model dalam bentuk string.
    """
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    summary_str = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    return summary_str
