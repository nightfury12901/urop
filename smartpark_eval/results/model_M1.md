# Model Architecture: M1 (Compact Target)

- **Model Class:** `SmartPark_M1_Compact_16_32_64`
- **Total Parameters:** 24,097
- **Trainable Parameters:** 23,873
- **Non-trainable Parameters:** 224
- **Calculated Weights FP32 Size:** 96,388 bytes (94.13 KB)

| Layer Index | Layer Name | Layer Type | Output Shape | Param # |
| :--- | :--- | :--- | :--- | ---: |
| 1 | `conv2d_1` | Conv2D | `(None, 64, 64, 16)` | 448 |
| 2 | `batch_norm_1` | BatchNormalization | `(None, 64, 64, 16)` | 64 |
| 3 | `max_pooling2d_1` | MaxPooling2D | `(None, 32, 32, 16)` | 0 |
| 4 | `conv2d_2` | Conv2D | `(None, 32, 32, 32)` | 4,640 |
| 5 | `batch_norm_2` | BatchNormalization | `(None, 32, 32, 32)` | 128 |
| 6 | `max_pooling2d_2` | MaxPooling2D | `(None, 16, 16, 32)` | 0 |
| 7 | `conv2d_3` | Conv2D | `(None, 16, 16, 64)` | 18,496 |
| 8 | `batch_norm_3` | BatchNormalization | `(None, 16, 16, 64)` | 256 |
| 9 | `max_pooling2d_3` | MaxPooling2D | `(None, 8, 8, 64)` | 0 |
| 10 | `global_avg_pool_1` | GlobalAveragePooling2D | `(None, 64)` | 0 |
| 11 | `dropout_1` | Dropout | `(None, 64)` | 0 |
| 12 | `output_dense` | Dense | `(None, 1)` | 65 |
