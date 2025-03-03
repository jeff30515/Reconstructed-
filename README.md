# 重建影像 PSNR 作業

## 專案概述
本專案利用 PyTorch 實作一個自動編碼器 (Autoencoder) 模型，對 DRIVE 資料集中的影像進行重建，並以 PSNR（峰值訊噪比, Peak Signal-to-Noise Ratio）作為評估指標，衡量重建影像與原始影像之間的差異。模型訓練共 200 個 epochs，最終會將重建影像儲存至指定資料夾，並將每張影像的 PSNR 值存入 CSV 檔案中。

---

## 主要特色
- **自動編碼器模型**：採用 Encoder-Decoder 結構，透過多層卷積與反卷積層進行影像重建。
- **自訂資料集類別**：實作 `DRIVE_Dataset` 類別，方便讀取 DRIVE 資料集中的影像。
- **訓練流程**：使用 MSELoss 搭配 Adam 優化器進行模型訓練。
- **PSNR 計算**：利用自定義函數計算重建影像的 PSNR 值，評估重建品質。
- **結果儲存**：重建影像會儲存到指定資料夾，所有 PSNR 值將輸出至 CSV 檔案。

---

## 系統需求
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Pandas
- Pillow (PIL)

---

## 資料集
本專案使用的資料集為 DRIVE：
- **訓練資料**：位於 `.../archive/DRIVE/training/images` 目錄下，包含原始影像。
- **測試資料**：位於 `.../DIRVE_TestingSet` 目錄下（請確認實際路徑是否正確）。

請根據實際情況更新程式中資料集的路徑。

---

## 使用說明

### 1. 資料準備
- 更新程式中 `train_images_path` 與 `test_images_path` 的路徑，確保能正確讀取影像檔案。

### 2. 模型訓練
- 執行程式開始模型訓練：
  ```bash
  python your_script.py
  ```
- 程式將進行 200 個 epochs 的訓練，每個 epoch 結束時會輸出平均損失值。

### 3.模型評估與結果儲存
- 訓練完成後，程式會在測試資料上進行重建，計算每張影像的 PSNR 值。
- 重建影像將儲存至 `reconstructed_images` 資料夾中。
- 所有影像的 PSNR 值會存入 CSV 檔案 `reconstruction_psnr.csv`。

---

## 模型架構
自動編碼器模型主要包含：
- 下採樣階段 (Encoder)：透過多層 DoubleConv 模組及 MaxPool2d 降低影像空間解析度，提取特徵。
- 上採樣階段 (Decoder)：利用 ConvTranspose2d 進行上採樣，並結合對應層的特徵，重建影像。
- 輸出層：透過 1x1 卷積產生最終的重建影像（3 個通道）。

---

## 訓練細節
- 損失函數：使用 MSELoss，計算重建影像與原始影像之間的均方誤差。
- 優化器：採用 Adam 優化器，學習率設為 1e-3。
- 訓練 Epoch：共 200 個 epochs。

---

## 評估指標
- PSNR（峰值訊噪比）：
  - 透過自定義函數 compute_psnr 計算重建影像與原始影像之間的 PSNR 值。
  - PSNR 值越高，代表重建品質越好。

---

## 輸出結果
- 訓練過程：每個 epoch 輸出平均損失值。
- 測試階段：列印每張影像的 PSNR 值，並計算整體平均 PSNR。
- 影像儲存：重建影像將儲存於 `reconstructed_images` 資料夾中。
- CSV 檔案：所有 PSNR 指標存入 `reconstruction_psnr.csv`。
