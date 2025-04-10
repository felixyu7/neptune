# Neptune: An Efficient Point Transformer for Ultrarelativistic Neutrino Events

Neptune (a**N** **E**fficient **P**oint **T**ransformer for **U**ltrarelativistic **N**eutrino **E**vents) is a transformer-based point cloud processing model specifically designed for neutrino event reconstruction in IceCube.

## Features

- Point cloud tokenization for irregular neutrino detector data
- Transformer-based architecture
- Compatible with both Prometheus and IceCube parquet datasets

## Usage

```python
# Example usage with training
python run.py -c configs/train_neptune.cfg
```

## Requirements

- PyTorch
- PyTorch Lightning
- NumPy
- Awkward Array
- PyArrow
- PyYAML
- SciPy
- fpsample
