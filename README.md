# Neptune: An Efficient Point Transformer for Ultrarelativistic Neutrino Events

Neptune (aN Efficient Point Transformer for Ultrarelativistic Neutrino Events) is a point cloud processing model specifically designed for neutrino event reconstruction in IceCube.

## Features

- Point cloud tokenization for irregular neutrino detector data
- Transformer-based architecture for effective feature extraction
- Compatible with both Prometheus and IceCube parquet datasets
- Flexible configuration system for easy model customization

## Usage

```python
# Example usage with training
python run.py -c configs/train_neptune.cfg
```

## Requirements

- PyTorch
- PyTorch Lightning
- NumPy
- Awkward Arrays
- PyArrow 