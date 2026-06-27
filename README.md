# MCNO: Monte Carlo-Type Neural Operator

Official implementation of **Monte Carlo-Type Neural Operator (MCNO)** for learning solution operators of parametric PDEs.

<p align="center">
  <a href="https://arxiv.org/abs/2511.18930"><img src="https://img.shields.io/badge/arXiv-2511.18930-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.1109/ICMLA61862.2025.00196"><img src="https://img.shields.io/badge/ICMLA%202025-10.1109/ICMLA61862.2025.00196-blue.svg" alt="ICMLA 2025"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

## Abstract

The Monte Carlo-type Neural Operator (MCNO) introduces a lightweight architecture for learning solution operators of parametric PDEs by directly approximating the kernel integral using a Monte Carlo approach. Unlike spectral or graph-based neural operators, MCNO avoids spectral or translation-invariance assumptions and instead represents the kernel as a learnable tensor over a fixed set of randomly sampled points. This design enables the model to generalize across multiple grid resolutions without requiring fixed global basis functions or repeated sampling during training. On standard 1D PDE benchmarks, MCNO achieves competitive accuracy with relatively low computational overhead, offering a simple and practical alternative to spectral and graph-based neural operators.

## Method

MCNO replaces the Fourier integral operator in FNO with a Monte Carlo-based kernel integration layer (`AdaptiveMCConv1d`):

1. **Random sampling** — A fixed set of random grid indices is sampled once at initialization
2. **Learnable kernel** — The kernel is parameterized as a learnable tensor over these sample points
3. **MC integration** — The integral operator is approximated via Monte Carlo summation
4. **Resolution invariance** — The architecture handles varying input resolutions through interpolation

## Installation

```bash
git clone https://github.com/pj1911/MCNO.git
cd MCNO
pip install -r requirements.txt
```

Requires Python >= 3.8, PyTorch >= 1.10, and CUDA (recommended).

## Data

Download the benchmark datasets and place them in the `data/` directory. See [data/README.md](data/README.md) for details.

- **KdV equation**: `kdv_train_test.mat` — generation code available at [mwt-operator/Data/kDV](https://github.com/gaurav71531/mwt-operator/tree/master/Data/kDV)
- **Burgers equation**: `Burgers_R10.zip` or `Burgers_v1000.zip` — available from the same [Google Drive folder](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)

## Usage

```bash
# Train on KdV equation (default)
python train.py --data_path data/kdv_train_test.mat

# See all options
python train.py --help
```

A step-by-step walkthrough is also available in [notebooks/demo.ipynb](notebooks/demo.ipynb).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{choutri2025mcno,
  title={Monte Carlo-Type Neural Operator for One-Dimensional Differential Equations},
  author={Choutri, Salah Eddine and Chauhan, Prajwal and Mazhar, Othmane and Jabari, Saif Eddin},
  booktitle={2025 International Conference on Machine Learning and Applications (ICMLA)},
  pages={1307--1312},
  year={2025},
  organization={IEEE}
}
```

```bibtex
@article{choutri2025mcno_arxiv,
  title={Monte Carlo-Type Neural Operator for One-Dimensional Differential Equations},
  author={Choutri, Salah Eddine and Chauhan, Prajwal and Mazhar, Othmane and Jabari, Saif Eddin},
  journal={arXiv preprint arXiv:2511.18930},
  year={2025},
  note={Machine Learning and the Physical Sciences Workshop, NeurIPS 2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- Data utilities and loss functions adapted from the [Fourier Neural Operator](https://github.com/neuraloperator/neuraloperator) codebase by [Zongyi Li et al.](https://arxiv.org/abs/2010.08895)
- Benchmark datasets from the [FNO data repository](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
