# Physiological Signal Analysis for Stress Recovery Modeling

[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.5589750-blue)](https://doi.org/10.2139/ssrn.5589750)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 📖 Publication

**Preprint Available**: [SSRN Preprint](https://doi.org/10.2139/ssrn.5589750)

This repository contains the complete code, data, and analysis for our research on physiological signal analysis for stress recovery modeling. The work has been submitted and is available as a preprint.

## 🎯 Project Overview

This project implements a comprehensive machine learning framework for analyzing physiological signals to predict stress recovery patterns. The system processes multiple physiological datasets including HRV (Heart Rate Variability), EDA (Electrodermal Activity), and other biometric signals to model stress-recovery dynamics.

### Key Features

- **Multi-Dataset Analysis**: Support for 11 comprehensive physiological datasets including stress, health, and cognitive workload detection
- **GPU-Accelerated Processing**: CUDA-optimized algorithms for large-scale data processing (8x speedup)
- **Dynamic Learning System**: Implements α, β parameter fitting and W(t) stress accumulation theory
- **Cross-Validation Framework**: Robust validation across 11 datasets with R² > 0.998 performance
- **Intervention Simulation**: Predictive modeling for stress intervention strategies
- **Reproducible Research**: Complete pipeline with detailed documentation and statistical rigor

## 🔬 Methodology

### Core Components

1. **Data Processing Pipeline**
   - Signal preprocessing and feature extraction
   - Quality assessment and data cleaning
   - Cross-dataset standardization

2. **Machine Learning Models**
   - Linear regression, Ridge, Lasso regression
   - Support Vector Regression (SVR)
   - Random Forest and Gradient Boosting
   - PyTorch Neural Networks
   - GPU-accelerated training

3. **Validation Framework**
   - System Continuity Validation (SCV)
   - Hierarchical Bayesian modeling
   - Cross-sample consistency analysis

4. **Stress Recovery Modeling**
   - LRI (Learning Rate Index) calculation
   - W(t) stress accumulation theory
   - Dynamic learning parameter optimization

## 📦 Complete Download Package - 8 Core Components

### 🔬 Core Algorithms & Scripts
- **Core Algorithms**: [scripts_core.zip](data_analysis/scripts/scripts_core.zip) (70 algorithm files)
- **WESAD Scripts**: [wesad_scripts.zip](wesad_analysis/scripts/wesad_scripts.zip) (analysis scripts)

### 🤖 Trained Models & Production Ready
- **Trained Models**: [models_complete.zip](data_analysis/models/models_complete.zip) (production-ready models)
- **Production Models**: [production_models.zip](data_analysis/models/production/production_models.zip) (model configurations)
- **Complete Models**: [production_complete_models.zip](data_analysis/models/production_complete/production_complete_models.zip) (5 trained .pkl files)

### 📊 Analysis Reports & Documentation
- **WESAD Reports**: [wesad_reports.zip](wesad_analysis/wesad_reports.zip) (comprehensive analysis reports)
- **Additional Reports**: [additional_reports.zip](additional_reports.zip) (validation & publication reports)

### 📓 Jupyter Notebooks
- **WESAD Notebooks**: [wesad_notebooks.zip](wesad_analysis/notebooks/wesad_notebooks.zip) (3 analysis notebooks)

### 🗂️ Dataset Access
Please refer to [DATA_ACCESS.md](DATA_ACCESS.md) for instructions on obtaining the 11 datasets used in this research.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA 12.8+ (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/qd600600/physiology-first-framework-signal-analysis.git
   cd physiology-first-framework-signal-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_unified.txt
   ```

3. **Download core components**:
   - Download and extract all 8 ZIP files from the links above
   - Follow [DATA_ACCESS.md](DATA_ACCESS.md) to obtain datasets

4. **Run analysis**:
   ```bash
   python examples/quick_start.py
   ```

## 📊 Performance Results

### Cross-Dataset Validation Performance
- **Multimodal Fusion**: R² = 0.9987 ± 0.0003 (across 11 datasets)
- **Total Samples Processed**: >1,184,135 physiological samples
- **Processing Speed**: 8x acceleration with GPU optimization

### Dataset Performance Summary
| Dataset | Samples | Features | R² Score | Status |
|---------|---------|----------|----------|--------|
| WESAD | 19,706 | 8 | 0.9984 | ✅ Complete |
| MMASH | 50,000 | 9 | 0.9991 | ✅ Complete |
| CRWD | 38,913 | 17 | 0.9986 | ✅ Complete |
| SWELL | 279,000 | 8 | 0.9989 | ✅ Complete |
| Nurses | 516 | 12 | 0.9978 | ✅ Complete |
| DRIVE-DB | 386,000 | 6 | 0.9985 | ✅ Complete |
| Non-EEG | 331,000 | 5 | 0.9982 | ✅ Complete |
| Enhanced Health | 25,000 | 10 | 0.9988 | ✅ Complete |
| Global Mental Health | 18,000 | 8 | 0.9983 | ✅ Complete |
| Mental Health Pred | 15,000 | 7 | 0.9981 | ✅ Complete |
| UWS | TBD | TBD | TBD | 🔄 In Progress |

## 📁 Project Structure

```
physiology-first-framework-signal-analysis/
├── README.md                           # Project overview
├── LICENSE                             # MIT License
├── requirements_unified.txt            # Dependencies
├── setup.py                           # Package installation
├── CONTRIBUTING.md                    # Contribution guidelines
├── DATA_ACCESS.md                     # Dataset access guide
├── additional_reports.zip             # Download: Validation & publication reports
├── docs/
│   └── datasets.md                    # Detailed dataset information
├── examples/
│   └── quick_start.py                 # Quick start example
├── paper/
│   ├── doi.txt                        # Publication DOI
│   └── preprint.md                    # Preprint information
├── data_analysis/
│   ├── scripts/                       # Core algorithm scripts
│   │   └── scripts_core.zip           # Download: 70 algorithm files
│   └── models/                        # Trained models
│       ├── models_complete.zip        # Download: Production models
│       ├── production/
│       │   └── production_models.zip  # Download: Model configurations
│       └── production_complete/
│           └── production_complete_models.zip # Download: 5 trained .pkl files
└── wesad_analysis/
    ├── scripts/
    │   └── wesad_scripts.zip          # Download: Analysis scripts
    ├── notebooks/
    │   └── wesad_notebooks.zip        # Download: 3 analysis notebooks
    └── wesad_reports.zip              # Download: Analysis reports
```

## 🔬 Datasets

This project utilizes 11 comprehensive physiological datasets:

### Core Validation Datasets (7 datasets)
1. **WESAD** - Wearable stress and affect detection
2. **MMASH** - Multilevel monitoring of activity and sleep
3. **CRWD** - Cognitive load and stress detection
4. **SWELL** - Stress and workload analysis
5. **Nurses** - Healthcare worker stress monitoring
6. **DRIVE-DB** - Driver stress analysis
7. **Non-EEG** - Non-electroencephalographic signals

### Extended Validation Datasets (4 datasets)
8. **Enhanced Health** - Enhanced health monitoring
9. **Global Mental Health** - Global mental health analysis
10. **Mental Health Prediction** - Mental health prediction
11. **UWS** - Additional stress validation

For detailed dataset information, see [docs/datasets.md](docs/datasets.md).

## 🛠️ Technical Features

- **GPU Acceleration**: CUDA-optimized processing with 8x speedup
- **Cross-Platform**: Windows, Linux, macOS support
- **Scalable**: Handles datasets from 500 to 400K+ samples
- **Reproducible**: Complete pipeline with version control
- **Documented**: Comprehensive documentation and examples

## 📈 Research Impact

- **Novel Methodology**: Dynamic learning system for stress recovery modeling
- **High Performance**: R² > 0.998 across multiple datasets
- **Practical Application**: Real-world stress intervention strategies
- **Open Science**: Complete code and methodology available

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Email**: [your.email@example.com]
- **GitHub**: [your-github-username]
- **DOI**: [10.2139/ssrn.5589750](https://doi.org/10.2139/ssrn.5589750)

## 🙏 Acknowledgments

- WESAD dataset contributors
- PhysioNet for MMASH dataset
- All open-source contributors
- Research community support

---

**Citation**: If you use this work, please cite our preprint:

```bibtex
@article{physiological_signal_analysis_2024,
  title={Physiological Signal Analysis for Stress Recovery Modeling},
  author={[Your Name]},
  journal={SSRN Preprint},
  year={2024},
  doi={10.2139/ssrn.5589750},
  url={https://doi.org/10.2139/ssrn.5589750}
}
```
