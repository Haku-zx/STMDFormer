# STMDFormer

This repository provides the official PyTorch implementation of **STMDFormer (Spatiotemporal Memory Decoupled Transformer)** for traffic flow forecasting.

STMDFormer is a Transformer-based spatiotemporal forecasting model that integrates spatiotemporal embedding, spatial memory attention, and decoupled learning to capture long-term dependencies and heterogeneous spatiotemporal patterns in traffic networks.

---

## Requirements

The code is implemented in **Python 3.8+** and **PyTorch**.

### Basic environment
- Python >= 3.8  
- PyTorch >= 1.9  
- NumPy  
- Pandas  
- SciPy  

You can install the dependencies using:

```bash
pip install numpy pandas scipy
pip install torch torchvision torchaudio
````

> GPU with CUDA is recommended for training, but CPU execution is also supported for testing.

---

## Project Structure

```text
STMDFormer/
│
├── model.py                     # STMDFormer model definition
├── train.py                     # Training and evaluation script
├── util.py                      # Data loading and evaluation utilities
├── engine.py                    # Training engine
│
├── generate_train_data_flow.py  # Generate training data (flow)
├── generate_train_data_speed.py # Generate training data (speed)
│
├── data/                        # Processed datasets (not included by default)
├── checkpoints/                 # Saved model checkpoints
└── README.md
```

---

## Data Preparation

This implementation follows the standard traffic forecasting data format used in PeMS datasets.

1. Prepare raw traffic data (e.g., PeMS03 / PeMS04 / PeMS07 / PeMS08).
2. Generate training, validation, and test sets using:

```bash
python generate_train_data_flow.py
```

or

```bash
python generate_train_data_speed.py
```

The processed data will be saved in `.npz` format and used directly for training.

> Note: Large raw datasets are **not included** in this repository.

---

## Training

You can train STMDFormer using the following command:

```bash
python train.py \
  --device cuda \
  --data PEMS07 \
  --num_nodes 207 \
  --seq_length 12 \
  --input_dim 3 \
  --nhid 64 \
  --edim 32 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --epochs 100
```

Key arguments:

* `--data`: dataset name
* `--num_nodes`: number of sensors
* `--seq_length`: prediction horizon
* `--batch_size`: batch size
* `--learning_rate`: learning rate

During training, model checkpoints will be saved automatically.

---

## Evaluation

After training, the script automatically evaluates the best model on the test set and reports:

* MAE
* RMSE
* MAPE

for each prediction horizon as well as the overall average performance.

---

## Model Description

STMDFormer consists of:

* **Spatiotemporal Embedding Layer**: fuses spatial topology and periodic temporal information.
* **Spatial Memory Attention**: captures long-term historical traffic patterns via a learnable memory bank.
* **Decoupled Learning Module**: separates spatial and temporal representations through normalization.
* **Transformer Encoder**: models heterogeneous spatiotemporal dependencies.

---

## Citation

If you find this code useful for your research, please consider citing the corresponding paper:

```bibtex
@article{STMDFormer,
  title={Spatiotemporal Memory Decoupled Transformer for Traffic Flow Forecasting},
  author={},
  journal={},
  year={}
}
```

---

## Acknowledgements

This implementation is based on PyTorch and follows common practices in spatiotemporal graph learning and traffic forecasting research.

---

## Contact

If you have any questions or issues, feel free to open an issue in this repository.

```
