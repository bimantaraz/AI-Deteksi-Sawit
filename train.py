from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Menggunakan device: {device}")
    if device == 'cpu':
        print("PERINGATAN: Tidak ada GPU (CUDA) yang terdeteksi. Training akan berjalan di CPU dan akan sangat lambat.")

    # --- Konfigurasi Training ---
    MODEL_NAME = 'yolo11m.pt'
    DATA_CONFIG = 'data_sawit.yaml'
    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    # ---------------------------

    print(f"Memulai training dengan model: {MODEL_NAME}")
    print(f"Data config: {DATA_CONFIG}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Image Size: {IMAGE_SIZE}")

    try:
        model = YOLO(MODEL_NAME)

        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            name='train_sawit_AI'
        )
        
        print("\n--- Training Selesai ---")
        print(f"Hasil training disimpan di folder: {results.save_dir}")
        print(f"Model terbaik disimpan di: {results.save_dir}/weights/best.pt")

        print("\nMenjalankan validasi pada 'validation set'...")
        val_results = model.val()
        print(val_results.metrics.box.maps)

        print("\nMenjalankan prediksi pada 'test set'...")
        test_results = model.predict(source='./sample_dataset', save=True, name='hasil_test_sawit')
        print(f"Hasil prediksi 'test' disimpan di folder 'sample_dataset/hasil_test_sawit'")

    except Exception as e:
        print(f"Error saat training: {e}")

if __name__ == '__main__':
    main()