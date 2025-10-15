# A Physiology-First, Execution-Efficiency Framework for Psychiatric Disorders: Retraining Intact Learning Trajectories

[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.5589750-blue)](https://doi.org/10.2139/ssrn.5589750)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 📖 Publication

**Preprint Available**: [SSRN Preprint](https://doi.org/10.2139/ssrn.5589750)

**Title**: A Physiology-First, Execution-Efficiency Framework for Psychiatric Disorders: Retraining Intact Learning Trajectories

**Author**: PENG LI, Independent Researcher, mr.perfect601601@gmail.com

This repository contains the complete code, data, and analysis for our research on physiological signal analysis for stress recovery modeling. The work has been submitted and is available as a preprint.

## 🌐 Language and Encoding Information

### 📋 Multilingual Content Notice

This repository contains content in both English and Chinese. Some files may display Chinese characters that could appear as encoding issues depending on your system settings.

### 🔧 Encoding Solutions

#### For Windows Users:
- Ensure your system locale supports UTF-8 encoding
- Use a text editor that supports UTF-8 (VS Code, Notepad++)
- Set terminal encoding to UTF-8 if viewing Chinese content

#### For Linux/Mac Users:
- Most modern systems handle UTF-8 by default
- If issues occur, set `LANG=en_US.UTF-8` or `LANG=zh_CN.UTF-8`

#### For GitHub Web Interface:
- Chinese characters should display correctly in modern browsers
- If garbled, try refreshing the page or using a different browser

### 📚 Translation Resources

#### Automated Translation Tools:
- **Google Translate**: Copy Chinese text and translate to your preferred language
- **DeepL**: More accurate for technical content
- **Browser Extensions**: Many browsers offer built-in translation

#### Manual Translation Priority:
1. **README files** - Essential for project understanding
2. **Configuration files** - Critical for setup
3. **Analysis reports** - Important for research details
4. **Code comments** - Helpful for implementation

#### Key Chinese Terms Translation:
- **生理信号** = Physiological Signals
- **压力恢复** = Stress Recovery  
- **机器学习** = Machine Learning
- **数据集** = Dataset
- **模型验证** = Model Validation
- **特征工程** = Feature Engineering

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

## 📦 Complete Download Package - 15 Core Components

### 🔬 Core Algorithms & Scripts
- **Core Algorithms**: [scripts_core.zip](data_analysis/scripts/scripts_core.zip) (70 algorithm files, 403KB)
- **WESAD Scripts**: [wesad_scripts.zip](wesad_analysis/scripts/wesad_scripts.zip) (analysis scripts, 94KB)

### 🤖 Trained Models & Production Ready
- **Trained Models**: [models_complete.zip](data_analysis/models/models_complete.zip) (production-ready models, 186KB)
- **Production Models**: [production_models.zip](data_analysis/models/production/production_models.zip) (model configurations, 182KB)
- **Complete Models**: [production_complete_models.zip](data_analysis/models/production_complete/production_complete_models.zip) (5 trained .pkl files, 3.4KB)

### 📊 Analysis Reports & Documentation
- **WESAD Reports**: [wesad_reports.zip](wesad_analysis/wesad_reports.zip) (comprehensive analysis reports, 64KB)
- **Additional Reports**: [additional_reports.zip](additional_reports.zip) (validation & publication reports, 7.7KB)
- **All WESAD Reports**: [all_wesad_reports.zip](wesad_analysis/all_wesad_reports.zip) (40 analysis reports, 64KB)

### 📓 Jupyter Notebooks
- **WESAD Notebooks**: [wesad_notebooks.zip](wesad_analysis/notebooks/wesad_notebooks.zip) (3 analysis notebooks, 1.7KB)

### 🔬 Advanced Analysis Results
- **Multimodal Fusion**: [multimodal_fusion_results.zip](wesad_analysis/multimodal_fusion_results/multimodal_fusion_results.zip) (17 files, 10.3MB)
- **Advanced Analysis**: [advanced_analysis_results.zip](wesad_analysis/advanced_analysis_results/advanced_analysis_results.zip) (8 files, 775KB)

### 📈 Theory Validation & Results
- **Theory Validation**: [theory_validation_results.zip](wesad_analysis/theory_validation_results/theory_validation_results.zip) (49 files, 19.5MB)
- **WESAD Results Part 1**: [wesad_results_part1.zip](wesad_analysis/results/wesad_results_part1.zip) (analysis results, 4.2MB)
- **WESAD Results Part 2**: [wesad_results_part2_files.zip](wesad_analysis/results/wesad_results_part2_files.zip) (GPU training files, 1.6MB)
- **WESAD Results Part 3**: [wesad_results_part2_weights.zip](wesad_analysis/results/wesad_results_part2_weights.zip) (model weights, 1.5MB)
- **WESAD Results Part 4**: [wesad_results_part5_phase_abc.zip](wesad_analysis/results/wesad_results_part5_phase_abc.zip) (phase analysis, 1.9MB)
- **WESAD Results Part 5**: [wesad_results_part4_logs.zip](wesad_analysis/results/wesad_results_part4_logs.zip) (training logs, 23.4MB)
- **WESAD Results Part 6**: [wesad_results_part3_pytorch.zip](wesad_analysis/results/wesad_results_part3_pytorch.zip) (PyTorch results, 23.9MB)
- **WESAD Results Part 7**: [wesad_results_part4_checkpoints.zip](wesad_analysis/results/wesad_results_part4_checkpoints.zip) (model checkpoints, 24.5MB)
- **WESAD Results Part 8**: [wesad_results_resume_best_model.zip](wesad_analysis/results/wesad_results_resume_best_model.zip) (best model, 501KB)
- **WESAD Results Part 9**: [wesad_results_resume_cache.zip](wesad_analysis/results/wesad_results_resume_cache.zip) (resume cache, 23.4MB)
- **WESAD Results Part 10**: [wesad_results_resume_epochs.zip](wesad_analysis/results/wesad_results_resume_epochs.zip) (epoch checkpoints, 24.0MB)
- **WESAD Results Part 11**: [wesad_results_cache_data.zip](wesad_analysis/results/wesad_results_cache_data.zip) (data cache, 24.0MB)

### 🗂️ Dataset Access
Please refer to [DATA_ACCESS.md](DATA_ACCESS.md) for instructions on obtaining the 11 datasets used in this research.

### 💾 Large Cache File (Available on Request)
- **GPU Training Cache**: `full_data_cache.npz` (58MB) - Contains preprocessed training data for GPU acceleration
  - **Location**: `wesad_analysis/results/LRI_Wt_GPU/checkpoints/full_data_cache.npz`
  - **Purpose**: Accelerates GPU training by caching preprocessed data tensors (X_train, X_val, y_train, y_val)
  - **Content**: Preprocessed data with shapes (1,425,603, 48, 3) for training and (322,324, 48, 3) for validation
  - **Availability**: Contact author for download link (file too large for GitHub web upload)
  - **Alternative**: File will be automatically generated when running training scripts
  - **Contact**: mr.perfect601601@gmail.com

## 📊 Datasets

This project uses 11 comprehensive physiological datasets for validation:

### 🌐 Publicly Available Datasets:
1. **WESAD** - [Kaggle Download](https://www.kaggle.com/datasets/robikscube/wesad-wearable-stress-affect-detection) (~500MB)
2. **MMASH** - [PhysioNet Download](https://physioNet.org/content/mmash/1.0.0/) (~200MB)
3. **SWELL** - [Kaggle Download](https://www.kaggle.com/datasets/swell-workload-analysis) (~1.2GB)
4. **DRIVE-DB** - [Kaggle Download](https://www.kaggle.com/datasets/drive-stress-analysis) (~1.5GB)

### 🔬 Research Datasets (Contact Author):
5. **CRWD** - Cognitive load and stress detection (~100MB)
6. **Nurses** - Healthcare worker stress monitoring (~1.1GB)
7. **Non-EEG** - Non-electroencephalographic signals (~100MB)
8. **Enhanced Health** - Enhanced health monitoring (~80MB)
9. **Global Mental Health** - Global mental health analysis (~60MB)
10. **Mental Health Prediction** - Mental health prediction (~40MB)
11. **UWS** - Additional stress validation (TBD)

### 📋 Dataset Organization
After downloading, organize datasets as follows:
```
data_analysis/data/
├── WESAD/
├── MMASH/
├── SWELL/
├── DRIVE_DB/
├── CRWD/
├── Nurses/
├── Non_EEG/
├── Enhanced_Health/
├── Global_Mental_Health/
├── Mental_Health_Pred/
└── UWS/
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA 12.8+ (for GPU acceleration)
- 8GB+ RAM recommended

### Installation
```bash
# Clone the repository
git clone https://github.com/qd600600/physiology-first-framework-signal-analysis.git
cd physiology-first-framework-signal-analysis

# Install dependencies
pip install -r requirements_unified.txt

# Run quick start example
python examples/quick_start.py
```

### Basic Usage
```python
from data_analysis.scripts import load_data, analyze_stress_recovery

# Load dataset
data = load_data('WESAD')

# Run analysis
results = analyze_stress_recovery(data)

# View results
print(results.summary())
```

## 📈 Performance Metrics

### Exceptional Results:
- **Multimodal Fusion**: R² = 0.9987 ± 0.0003 across 11 datasets
- **Cross-Dataset Validation**: >99.8% accuracy
- **GPU Acceleration**: 8x speedup over CPU processing
- **Total Samples**: >1,184,135 physiological samples processed

### Key Achievements:
- **Data Leakage Resolution**: Eliminated temporal data leakage in cross-validation
- **GPU Optimization**: CUDA-accelerated training pipeline
- **Robust Validation**: Comprehensive cross-dataset testing
- **Reproducible Results**: Complete documentation and version control

## 🛠️ System Requirements

### Minimum Requirements:
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.12+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Recommended for GPU Acceleration:
- **GPU**: NVIDIA RTX 3060 or better
- **CUDA**: 12.8+
- **RAM**: 32GB+
- **Storage**: SSD with 50GB+ free space

## 📚 Documentation

### 📖 Available Documentation:
- **Setup Guide**: [GITHUB_SETUP_GUIDE.md](GITHUB_SETUP_GUIDE.md)
- **Dataset Information**: [docs/datasets.md](docs/datasets.md)
- **Data Access**: [DATA_ACCESS.md](DATA_ACCESS.md)
- **Upload Instructions**: [UPLOAD_INSTRUCTIONS.md](UPLOAD_INSTRUCTIONS.md)
- **Contributing Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)

### 🔍 Troubleshooting:
- **Encoding Issues**: See Language and Encoding Information section above
- **GPU Setup**: Check CUDA installation and compatibility
- **Memory Issues**: Reduce batch size or use CPU-only mode
- **Dataset Access**: Contact author for research datasets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Author**: PENG LI  
**Institution**: Independent Researcher  
**Email**: mr.perfect601601@gmail.com  
**Preprint**: [SSRN Preprint](https://doi.org/10.2139/ssrn.5589750)

## 🙏 Acknowledgments

- WESAD dataset authors for providing comprehensive physiological data
- PhysioNet for hosting MMASH dataset
- Kaggle community for dataset accessibility
- CUDA developers for GPU acceleration support

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{li2024physiology,
  title={A Physiology-First, Execution-Efficiency Framework for Psychiatric Disorders: Retraining Intact Learning Trajectories},
  author={Li, Peng},
  journal={SSRN Preprint},
  year={2024},
  doi={10.2139/ssrn.5589750}
}
```

---

**Note**: This repository contains both English and Chinese content. Please refer to the Language and Encoding Information section for translation and encoding guidance.
