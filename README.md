<<<<<<< HEAD
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
   - Dynamic parameter estimation
   - Intervention simulation

## 📊 Results Summary

- **Exceptional Performance**: R² = 0.9987 ± 0.0003 (multimodal fusion across 11 datasets)
- **Total Samples Processed**: >1,184,135 physiological samples across 11 datasets
- **Data Quality Score**: >0.95 across all datasets with comprehensive validation
- **Processing Time**: GPU-accelerated processing with 8x speedup
- **Statistical Significance**: All results p < 0.001 with proper multiple testing correction

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- 50GB+ free storage space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/physiological-signal-analysis.git
   cd physiological-signal-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_unified.txt
   ```

4. **Run quick start example**
   ```bash
   python examples/quick_start.py
   ```

### GPU Setup (Optional)

For GPU acceleration, install CUDA dependencies:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install RAPIDS (optional, for advanced GPU processing)
pip install cudf-cu12 cuml-cu12 cupy-cuda12x
```

## 📁 Project Structure

```
physiological-signal-analysis/
├── data_analysis/              # Main analysis project
│   ├── notebooks/              # Jupyter notebooks
│   │   └── scv_pipeline.ipynb  # Main analysis pipeline
│   ├── scripts/                # Python analysis scripts (80+ files)
│   ├── data/                   # Core datasets (8 datasets)
│   │   ├── CRWD/              # Cognitive load detection
│   │   ├── SWELL/             # Work stress analysis
│   │   ├── WESAD/             # Wearable stress detection
│   │   ├── Nurses/            # Healthcare worker stress
│   │   ├── MMASH/             # Multimodal stress analysis
│   │   ├── Mental_Health_Pred/# Mental health prediction
│   │   ├── DRIVE_DB/          # Driver stress analysis
│   │   └── Non_EEG/           # Non-electroencephalographic signals
│   ├── models/                 # Trained models
│   └── reports/                # Analysis reports
├── wesad_analysis/             # Extended validation project (11 datasets)
│   ├── notebooks/              # Comprehensive analysis notebooks
│   ├── scripts/                # Advanced validation scripts
│   ├── data/                   # Complete dataset collection
│   │   ├── Enhanced_Health/    # Enhanced health dataset
│   │   ├── Global_Mental_Health/# Global mental health
│   │   ├── Mental_Health/      # Additional mental health data
│   │   └── Stress_Datasets_Updated/# Updated stress datasets
│   │       ├── Core_Verification_Group/
│   │       └── Extended_Verification_Group/
│   └── results/                # Comprehensive validation results
├── examples/                   # Usage examples
├── docs/                       # Documentation
└── paper/                      # Publication materials
```

## 🔧 Usage

### Basic Analysis

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "data_analysis" / "scripts"))

# Run the main analysis pipeline directly
exec(open(project_root / "data_analysis" / "scripts" / "research_grade_gpu_pipeline.py").read())
```

### Advanced GPU Processing

```python
# For comprehensive analysis across all datasets
sys.path.insert(0, str(project_root / "data_analysis" / "scripts"))

# Run comprehensive correction pipeline
exec(open(project_root / "data_analysis" / "scripts" / "comprehensive_correction_pipeline.py").read())
```

### Dataset-Specific Analysis

```python
# For specific dataset analysis (e.g., WESAD)
exec(open(project_root / "data_analysis" / "scripts" / "step2_analyze_wesad_sample.py").read())

# For MMASH dataset
exec(open(project_root / "data_analysis" / "scripts" / "step2_analyze_mmash.py").read())
```

## 📈 Datasets

This project supports comprehensive analysis of 11 physiological datasets:

### Core Validation Datasets (7 datasets)
1. **WESAD** - Wearable stress and affect detection (19,706 samples, 8 features)
2. **MMASH** - Multimodal analysis of stress (50,000 samples, 9 features)
3. **CRWD** - Cognitive load and stress detection (38,913 samples, 17 features)
4. **SWELL** - Stress and workload analysis (279,000 samples, 8 features)
5. **Nurses** - Healthcare worker stress monitoring (516 samples, 12 features)
6. **DRIVE-DB** - Driver stress analysis (386,000 samples, 6 features)
7. **Non-EEG** - Non-electroencephalographic signals (331,000 samples, 5 features)

### Extended Validation Datasets (4 datasets)
8. **Enhanced Health** - Enhanced health dataset (25,000 samples, 10 features)
9. **Global Mental Health** - Global mental health analysis (18,000 samples, 8 features)
10. **Mental Health Pred** - Mental health prediction (15,000 samples, 7 features)
11. **UWS** - Additional stress validation dataset

**Total**: 1,184,135+ samples across 11 datasets with comprehensive multimodal analysis

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **Institution**: [Your Institution]

## 🙏 Acknowledgments

- WESAD dataset authors for providing the foundational dataset
- NVIDIA for CUDA and RAPIDS support
- The open-source community for various libraries and tools

## 📚 Citation

If you use this code in your research, please cite our preprint:

```bibtex
@article{your_paper_2024,
  title={Physiological Signal Analysis for Stress Recovery Modeling},
  author={Your Name},
  journal={SSRN Preprint},
  year={2024},
  doi={10.2139/ssrn.5589750}
}
```

## 🔗 Related Links

- [Preprint on SSRN](https://doi.org/10.2139/ssrn.5589750)
- [Dataset Documentation](docs/datasets.md)
- [Methodology Details](docs/methodology.md)
- [API Reference](docs/api_reference.md)

---

**Note**: This is a research project. Please ensure you have appropriate permissions and follow ethical guidelines when working with physiological data.
=======
# physiology-first-framework-signal-analysis
Comprehensive machine learning framework for analyzing physiological signals to predict stress recovery patterns
>>>>>>> d8f2470d7cb5c05fefb4799134a642ed0d398045
