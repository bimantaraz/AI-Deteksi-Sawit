# AI Deteksi Sawit

![1029 (1)](https://github.com/user-attachments/assets/4096c137-f075-42d1-9690-a5a1d3007df7)

---

## 📊 Hasil Performa Model

Model ini ditraining selama 50 *epoch* dan mencapai **mAP50 sebesar 95.6%** pada *validation set*.

| Metrik | Nilai |
| :--- | :--- |
| mAP50 | 0.956 |
| mAP50-95 | 0.669 |
| Precision | 0.848 |
| Recall | 0.967 |

<img width="400" height="200" alt="results" src="https://github.com/user-attachments/assets/2913aca4-f43f-4d66-af28-91975993bf53" />
<br>
<img width="300" height="300" alt="results" src="https://github.com/user-attachments/assets/8bfb8f7b-c64b-4c6d-b9a8-f9fd5d60d7d4" />
<br>
<img width="400" height="1000" alt="results" src="https://github.com/user-attachments/assets/73590b34-6c85-458f-92de-182b69c56a79" />

---

## 🛠️ Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone https://github.com/bimantaraz/AI-Deteksi-Sawit.git
    cd AI-Deteksi-Sawit
    ```

2.  **Buat virtual environment**:
    ```bash
    python -m venv venv
    source venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Cara Penggunaan

Gunakan skrip `deteksi.py` untuk menjalankan deteksi pada video.

```bash
python deteksi.py
