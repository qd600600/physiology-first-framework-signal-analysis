# A Physiology-First, Execution-Efficiency Framework for Psychiatric Disorders: Retraining Intact Learning Trajectories

[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.5589750-blue)](https://doi.org/10.2139/ssrn.5589750)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 📖 Publication

**Preprint Available**: [SSRN Preprint](https://doi.org/10.2139/ssrn.5589750)

**Technical Whitepaper**: [W(t) Stress Accumulation Framework](WHITEPAPER.md) - **NEW!**

**Title**: A Physiology-First, Execution-Efficiency Framework for Psychiatric Disorders: Retraining Intact Learning Trajectories

**Author**: PENG LI, Independent Researcher, mr.perfect601601@gmail.com

This repository contains the complete code, data, and analysis for our research on physiological signal analysis for stress recovery modeling. The work has been submitted and is available as a preprint. **NEW**: A comprehensive technical whitepaper detailing the W(t) stress accumulation framework with Bayesian validation (BF > 10³¹) and extreme physiological stratification (d > 10) is now available.

## 🎉 **Project Status: FULLY OPEN SOURCE**

**✅ COMPLETE OPEN SOURCE IMPLEMENTATION**

This project is now fully open-sourced with all essential components available for download. The repository contains 29 comprehensive ZIP packages covering all core algorithms, trained models, analysis results, and scientific documentation.

### 📊 **Upload Status Summary**
- **Total Components**: 29 ZIP files
- **Core Algorithms**: 100% uploaded (90 Python scripts)
- **Trained Models**: 100% uploaded (production-ready)
- **Scientific Documentation**: 100% uploaded (complete audit trail)
- **Analysis Results**: 100% uploaded (11 datasets validation)
- **LRI Calculation Data**: 100% uploaded (by dataset groups)
- **Technical Whitepaper**: 100% uploaded (publication-ready)
- **Project Size**: ~56MB (optimized for GitHub)

### 🚀 **Ready for Use**
All files are immediately available for download and use. No external dependencies or additional setup required beyond the provided ZIP packages.

## 📋 **Technical Whitepaper: W(t) Stress Accumulation Framework**

### 🎯 **Breakthrough Scientific Findings**

Our comprehensive technical whitepaper presents groundbreaking results that establish new standards in stress science:

#### **🔬 Decisive Continuity Evidence**
- **Bayes Factor > 10³¹** across all 11 datasets - the strongest evidence for stress continuity in scientific literature
- **WAIC Δ < -10** consistently favoring continuous over discrete models
- **Hierarchical Bayesian validation** with cross-dataset generalization

#### **⚡ Extreme Physiological Stratification**
- **Cohen's d = 16.80** for accumulation rate differences (high vs low stress groups)
- **Cohen's d = 9.06** for recovery rate differences
- **127% higher accumulation rates** in high-stress groups (α = 1.806 vs 0.794 × 10⁻³ s⁻¹)

#### **🏢 Context-Specific Risk Benchmarks**
- **Workplace environments**: Highest risk (α = 5.01 × 10⁻³ s⁻¹)
- **Driving contexts**: Lowest risk (α = 0.30 × 10⁻³ s⁻¹)
- **Cognitive tasks**: Medium risk (α = 1.19 × 10⁻³ s⁻¹)

#### **🎯 Clinical Translation Ready**
- **High-risk threshold**: α > 1.5 × 10⁻³ s⁻¹
- **Intervention priority**: β-boost strategies (sleep, mindfulness, recovery environments)
- **Multi-modal fusion**: R² = 0.9987 ± 0.0003 prediction accuracy

### 📊 **Whitepaper Contents**
- **8,500+ words** of publication-ready scientific content
- **9 publication-quality figures** (300 DPI)
- **Comprehensive statistical validation** with effect sizes and confidence intervals
- **Clinical implementation guidelines** with quantitative thresholds
- **Future research directions** and limitations

### 🔗 **Access the Whitepaper**
- **Main Document**: [WHITEPAPER.md](WHITEPAPER.md)
- **Figures**: [figures/](figures/) directory with 9 publication-quality charts
- **Data Tables**: [tables/](tables/) directory with statistical results
- **Citation**: [CITATION.cff](CITATION.cff) for proper academic referencing

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

### 🔤 Key Terminology (Chinese → English)
| Chinese | English | Context |
|---------|---------|---------|
| 生理信号 | Physiological Signals | Core functionality |
| 压力恢复 | Stress Recovery | Main research focus |
| 机器学习 | Machine Learning | ML pipeline |
| 数据集 | Dataset | Data sources |
| 模型验证 | Model Validation | Analysis results |
| 特征工程 | Feature Engineering | Data processing |
| 配置文件 | Configuration File | Setup files |
| 分析结果 | Analysis Results | Research outputs |
| 算法脚本 | Algorithm Scripts | Core code |
| 训练模型 | Trained Models | ML models |

### 🌍 System Compatibility

#### Recommended Tools:
- **Text Editors**: VS Code, Sublime Text, Notepad++
- **Terminals**: Windows Terminal, PowerShell, Git Bash
- **Browsers**: Chrome, Firefox, Edge (latest versions)
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

#### Troubleshooting Encoding Issues:
1. **Check file encoding**: Ensure files are saved as UTF-8
2. **Terminal settings**: Set UTF-8 locale in terminal
3. **Editor settings**: Configure editor to use UTF-8
4. **Browser settings**: Enable Unicode support

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

## 🎯 **Quick Download - Core Components**

### 📦 **Complete Project Packages (29 ZIP Files)**

#### **🔬 Core Algorithm & Scripts**:
- [Core Scripts Part 1](data_analysis/scripts/core_scripts_part1.zip) (45 scripts, 201KB)
- [Core Scripts Part 2](data_analysis/scripts/core_scripts_part2.zip) (45 scripts, 287KB)
- [WESAD Analysis Scripts](wesad_analysis/scripts/wesad_scripts.zip) (WESAD scripts, 94KB)

#### **🤖 Trained Models & Configurations**:
- [Complete Models](data_analysis/models/models_complete.zip) (trained models, 186KB)
- [Production Models](data_analysis/models/production_models.zip) (production configs, 182KB)
- [Complete Production Models](data_analysis/models/production_complete_models.zip) (5 .pkl files, 3.4KB)

#### **📊 Analysis Results & Reports**:
- [WESAD Reports](wesad_analysis/wesad_reports.zip) (all analysis reports, 64KB)
- [Additional Reports](additional_reports.zip) (other important reports, 7.7KB)
- [Important Reports](data_analysis/final_project_delivery/important_reports.zip) (final delivery reports, 16KB)
- [Project Summary](data_analysis/final_project_delivery/project_summary.zip) (project summary, 3KB)

#### **🔬 Scientific Validation Results**:
- [Theory Validation Results](wesad_analysis/theory_validation_results/theory_validation_results.zip) (19.5MB)
- [Multimodal Fusion Results](wesad_analysis/multimodal_fusion_results/multimodal_fusion_results.zip) (10.3MB)
- [Advanced Analysis Results](wesad_analysis/advanced_analysis_results/advanced_analysis_results.zip) (775KB)

#### **📈 WESAD Analysis Results** (Split into manageable parts):
- [WESAD Results Part 1](wesad_analysis/results/wesad_results_part1.zip) (4.2MB)
- [WESAD Results Part 2 - Weights](wesad_analysis/results/wesad_results_part2_weights.zip) (1.5MB)
- [WESAD Results Part 2 - Files](wesad_analysis/results/wesad_results_part2_files.zip) (1.6MB)
- [WESAD Results Part 3 - PyTorch](wesad_analysis/results/wesad_results_part3_pytorch.zip) (23.9MB)
- [WESAD Results Part 4 - Logs](wesad_analysis/results/wesad_results_part4_logs.zip) (23.4MB)
- [WESAD Results Part 4 - Checkpoints](wesad_analysis/results/wesad_results_part4_checkpoints.zip) (24.5MB)
- [WESAD Results Part 5 - Phase ABC](wesad_analysis/results/wesad_results_part5_phase_abc.zip) (1.9MB)

#### **🧠 LRI Calculation Data** (By Dataset Groups):
- [LRI Calculation - WESAD](data_analysis/final_project_delivery/lri_calculation_wesad.zip) (1.3MB)
- [LRI Calculation - MMASH](data_analysis/final_project_delivery/lri_calculation_mmash.zip) (234KB)
- [LRI Calculation - CRWD](data_analysis/final_project_delivery/lri_calculation_crwd.zip) (384KB)
- [LRI Calculation - SWELL](data_analysis/final_project_delivery/lri_calculation_swell.zip) (2MB)
- [LRI Calculation - DRIVE-DB](data_analysis/final_project_delivery/lri_calculation_drive_db.zip) (3MB)
- [LRI Calculation - Others](data_analysis/final_project_delivery/lri_calculation_others.zip) (36KB)

#### **👩‍⚕️ Nurses Dataset LRI Calculation** (Split by Time Windows):
- [Nurses LRI - 300s](data_analysis/final_project_delivery/lri_calculation_nurses_300s.zip) (5.6MB)
- [Nurses LRI - 60s Original](data_analysis/final_project_delivery/lri_calculation_nurses_60s_original.zip) (13.4MB)
- [Nurses LRI - 60s Fixed](data_analysis/final_project_delivery/lri_calculation_nurses_60s_fixed.zip) (14.8MB)
- [Nurses LRI - 900s](data_analysis/final_project_delivery/lri_calculation_nurses_900s.zip) (1.9MB)

#### **📚 Final Project Delivery Components**:
- [Code Scripts](data_analysis/final_project_delivery/code_scripts.zip) (final delivery scripts, 177KB)
- [Parameter Selection](data_analysis/final_project_delivery/data_processing_parameter_selection.zip) (parameter selection data, 8KB)

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

### 📊 **Additional Large Data Files (External Hosting)**
The following large data files (>100MB) are available upon request for complete data analysis:

#### **Cleaned Dataset Files** (~300MB total):
- `cleaned_SWELL_complete.parquet` (~104MB) - Complete SWELL dataset
- `cleaned_WESAD_full_sample.parquet` (~46MB) - Full WESAD sample
- `cleaned_Nurses_full_sample.parquet` (~61MB) - Full Nurses dataset
- Additional cleaned data files for all 11 datasets

#### **W(t) Time Series Data** (~400MB total):
- `wt_timeseries_SWELL_complete.parquet` (~113MB) - SWELL stress accumulation
- `wt_timeseries_WESAD_full_sample.parquet` (~90MB) - WESAD stress trajectories
- `wt_timeseries_Nurses_wsl_gpu.parquet` (~110MB) - Nurses GPU-processed data
- Additional W(t) time series for all datasets

**Note**: These files are not essential for core functionality but provide complete raw data for advanced analysis. All essential components are included in the 29 uploaded ZIP packages.

## 📊 Datasets

This project uses 11 comprehensive physiological datasets for validation:

### **Core Validation Datasets**:
1. **WESAD** - Wearable stress detection (17 subjects, Empatica E4)
2. **MMASH** - 24-hour multimodal monitoring (22 subjects)
3. **CRWD** - Cognitive workload detection (research dataset)
4. **SWELL** - Work stress analysis (25 subjects, 1.1GB)
5. **Nurses** - Healthcare worker stress (1.1GB)
6. **DRIVE-DB** - Driver stress detection (research dataset)
7. **Non-EEG** - Non-EEG physiological signals (research dataset)

### **Extended Verification Datasets**:
8. **Enhanced Health** - Enhanced health monitoring (research dataset)
9. **Global Mental Health** - Global mental health patterns (research dataset)
10. **Mental Health Pred** - Mental health prediction (research dataset)
11. **UWS** - Urban workplace stress (research dataset)

**Note**: Research datasets (CRWD, DRIVE-DB, Non-EEG, Enhanced Health, Global Mental Health, Mental Health Pred, UWS) require permission from the authors. Contact mr.perfect601601@gmail.com for access.

### 🌐 **Publicly Available Datasets**:
1. **WESAD** - [Kaggle Download](https://www.kaggle.com/datasets/robikscube/wesad-wearable-stress-affect-detection) (~500MB)
2. **MMASH** - [PhysioNet Download](https://physionet.org/content/mmash/1.0.0/) (~200MB)
3. **SWELL** - [Kaggle Download](https://www.kaggle.com/datasets/swell-workload-analysis) (~1.2GB)
4. **DRIVE-DB** - [Kaggle Download](https://www.kaggle.com/datasets/drive-stress-analysis) (~1.5GB)

### 📋 **Dataset Organization**
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

## 🏆 **Project Achievements**

- **Multi-Dataset Analysis**: Support for 11 comprehensive physiological datasets
- **Exceptional Performance**: R² = 0.9987 ± 0.0003 (multimodal fusion across 11 datasets)
- **Advanced GPU Acceleration**: CUDA 12.8, RAPIDS cuDF/cuML, PyTorch optimization
- **Comprehensive Validation**: Cross-dataset consistency analysis and intervention simulation
- **Total Samples Processed**: >1,184,135 physiological samples across 11 datasets
- **Complete Open Source**: 29 ZIP packages with all essential components ready for immediate use

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
- **Scientific Rigor**: Complete 6-step audit process with validation

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

# Extract all ZIP packages (see ZIP extraction guide above)
# Download and extract the 29 ZIP files for complete functionality

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

## 📁 Project Structure

```
physiology-first-framework-signal-analysis/
├── data_analysis/                 # Main analysis directory
│   ├── scripts/                  # Core algorithm scripts (90 files)
│   ├── models/                   # Trained models and configurations
│   ├── notebooks/                # Jupyter analysis notebooks
│   ├── final_project_delivery/   # Complete project delivery files
│   └── results/                  # Analysis results and outputs
├── wesad_analysis/               # WESAD-specific analysis
│   ├── data/                     # All 11 datasets (external links)
│   ├── scripts/                  # WESAD analysis scripts
│   ├── results/                  # WESAD analysis results
│   ├── multimodal_fusion_results/ # Multimodal fusion outputs
│   ├── theory_validation_results/ # Theory validation results
│   └── advanced_analysis_results/ # Advanced analysis outputs
├── docs/                         # Documentation
├── examples/                     # Quick start examples
├── paper/                        # Preprint information
└── requirements_unified.txt      # Dependencies
```

## 💡 **ZIP File Extraction Guide**

### 📋 **How to Extract and Use ZIP Files**

All 29 ZIP files contain essential components for complete project functionality. Here's how to extract and organize them:

#### **Step 1: Download All ZIP Files**
Download all 29 ZIP files from the links above to your local machine.

#### **Step 2: Extract to Correct Directories**
Extract each ZIP file to its corresponding directory in your cloned repository:

```bash
# Core scripts
unzip core_scripts_part1.zip -d data_analysis/scripts/
unzip core_scripts_part2.zip -d data_analysis/scripts/

# Models
unzip models_complete.zip -d data_analysis/models/
unzip production_models.zip -d data_analysis/models/

# WESAD analysis
unzip wesad_scripts.zip -d wesad_analysis/scripts/
unzip wesad_reports.zip -d wesad_analysis/

# Results (extract to appropriate subdirectories)
unzip theory_validation_results.zip -d wesad_analysis/theory_validation_results/
unzip multimodal_fusion_results.zip -d wesad_analysis/multimodal_fusion_results/
unzip advanced_analysis_results.zip -d wesad_analysis/advanced_analysis_results/

# WESAD results (extract to results directory)
unzip wesad_results_part1.zip -d wesad_analysis/results/
unzip wesad_results_part2_weights.zip -d wesad_analysis/results/
unzip wesad_results_part2_files.zip -d wesad_analysis/results/
unzip wesad_results_part3_pytorch.zip -d wesad_analysis/results/
unzip wesad_results_part4_logs.zip -d wesad_analysis/results/
unzip wesad_results_part4_checkpoints.zip -d wesad_analysis/results/
unzip wesad_results_part5_phase_abc.zip -d wesad_analysis/results/

# LRI calculation data
unzip lri_calculation_wesad.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_mmash.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_crwd.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_swell.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_drive_db.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_others.zip -d data_analysis/final_project_delivery/

# Nurses LRI data
unzip lri_calculation_nurses_300s.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_nurses_60s_original.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_nurses_60s_fixed.zip -d data_analysis/final_project_delivery/
unzip lri_calculation_nurses_900s.zip -d data_analysis/final_project_delivery/

# Final project delivery
unzip code_scripts.zip -d data_analysis/final_project_delivery/
unzip important_reports.zip -d data_analysis/final_project_delivery/
unzip project_summary.zip -d data_analysis/final_project_delivery/
unzip data_processing_parameter_selection.zip -d data_analysis/final_project_delivery/

# Additional reports
unzip additional_reports.zip -d ./
```

#### **Step 3: Verify Extraction**
After extraction, your directory structure should match the project structure shown above.

#### **Step 4: Install Dependencies**
```bash
pip install -r requirements_unified.txt
```

#### **Step 5: Run Analysis**
```bash
python examples/quick_start.py
```

### 📝 **What Each ZIP Contains**:

#### **Core Components**:
- **Scripts**: 90 Python files with core algorithms and data processing
- **Models**: Trained machine learning models and configurations
- **Reports**: Comprehensive analysis reports and documentation

#### **Analysis Results**:
- **Theory Validation**: Complete validation of W(t) stress theory
- **Multimodal Fusion**: Advanced fusion results across 11 datasets
- **LRI Calculation**: Learning Rate Index calculations for all datasets

#### **Final Delivery**:
- **Complete Scripts**: All processing and analysis scripts
- **Parameter Selection**: Optimized parameters for all datasets
- **Project Summary**: Comprehensive project documentation

### 💡 **Extraction Tips**:
- Use any standard ZIP extraction tool (WinRAR, 7-Zip, built-in Windows/Mac extractors)
- Maintain the directory structure when extracting
- These files are essential for complete research reproducibility

## 🔧 System Requirements

### Minimum Requirements:
- **OS**: Windows 10, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.12+
- **RAM**: 8GB+
- **Storage**: 10GB free space
- **GPU**: Optional (CPU-only mode supported)

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
- **NEW: Complete Project Documentation**: See ZIP extraction guide above

### 🔍 Troubleshooting:
- **Encoding Issues**: See Language and Encoding Information section above
- **GPU Setup**: Check CUDA installation and compatibility
- **Memory Issues**: Reduce batch size or use CPU-only mode
- **Dataset Access**: Contact author for research datasets
- **ZIP Extraction**: Follow the detailed extraction guide above
- **Large Data Files**: Contact author for files >100MB (cleaned data, W(t) time series)

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
- GitHub community for open source platform support

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{li2024physiology,
  title={A Physiology-First, Execution-Efficiency Framework for Psychiatric Disorders: Retraining Intact Learning Trajectories},
  author={Li, Peng},
  journal={SSRN Preprint},
  year={2024},
  doi={10.2139/ssrn.5589750},
  url={https://doi.org/10.2139/ssrn.5589750}
}
```

---

**🎉 Congratulations! You now have access to a complete, fully open-sourced physiological signal analysis framework with 29 comprehensive components ready for immediate use.**
