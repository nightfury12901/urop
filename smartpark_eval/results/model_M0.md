# Model Architecture: M0 (Paper Baseline)

- **Model Class:** `SmartPark_M0_Baseline`
- **Total Parameters:** 285,857
- **Trainable Parameters:** 285,857
- **Non-trainable Parameters:** 0
- **Calculated Weights FP32 Size:** 1,143,428 bytes (1116.63 KB)

| Layer Index | Layer Name | Layer Type | Output Shape | Param # |
| :--- | :--- | :--- | :--- | ---: |
| 1 | `conv2d_1` | Conv2D | `(None, 64, 64, 16)` | 448 |
| 2 | `max_pooling2d_1` | MaxPooling2D | `(None, 32, 32, 16)` | 0 |
| 3 | `conv2d_2` | Conv2D | `(None, 32, 32, 32)` | 4,640 |
| 4 | `max_pooling2d_2` | MaxPooling2D | `(None, 16, 16, 32)` | 0 |
| 5 | `conv2d_3` | Conv2D | `(None, 16, 16, 64)` | 18,496 |
| 6 | `max_pooling2d_3` | MaxPooling2D | `(None, 8, 8, 64)` | 0 |
| 7 | `flatten_1` | Flatten | `(None, 4096)` | 0 |
| 8 | `dense_1` | Dense | `(None, 64)` | 262,208 |
| 9 | `dropout_1` | Dropout | `(None, 64)` | 0 |
| 10 | `output_dense` | Dense | `(None, 1)` | 65 |
