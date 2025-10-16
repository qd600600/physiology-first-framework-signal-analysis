# Data Access Guide

## 📊 Dataset Download Links

This project employs a **hierarchical validation architecture** across 11 comprehensive physiological datasets (n = 1,184,135 samples from 1,000+ participants). Due to their large size (total ~4GB), they are not included directly in this repository. Please download them from the following sources:

### 🏗️ **Hierarchical Validation Architecture**
- **L1**: Complete Dataset Inventory (11 datasets, 1,184,135 samples)
- **L2**: Multimodal Fusion Validation (11 datasets, R² = 0.9987±0.0003)
- **L3**: Stress Stratification Analysis (6 datasets, 35,497 samples)
- **L4**: Context-Specific Benchmarking (5 environments)

### 🔬 Core Validation Datasets (7 datasets)

#### 1. WESAD (Wearable Stress and Affect Detection)
- **Source**: [Kaggle - WESAD Dataset](https://www.kaggle.com/datasets/robikscube/wesad-wearable-stress-affect-detection)
- **Size**: ~500MB
- **Description**: Wearable stress detection with Empatica E4 device
- **Files needed**: Complete dataset folder

#### 2. MMASH (Multilevel Monitoring of Activity and Sleep)
- **Source**: [PhysioNet - MMASH](https://physionet.org/content/mmash/1.0.0/)
- **Size**: ~200MB
- **Description**: 24-hour multimodal monitoring
- **Files needed**: Complete dataset folder

#### 3. CRWD (Cognitive Load and Stress Detection)
- **Source**: Research dataset (contact authors)
- **Size**: ~100MB
- **Description**: Cognitive workload detection
- **Files needed**: Complete dataset folder

#### 4. SWELL (Stress and Workload Analysis)
- **Source**: [Kaggle - SWELL Dataset](https://www.kaggle.com/datasets/swell-workload-analysis)
- **Size**: ~1.2GB
- **Description**: Work stress analysis
- **Files needed**: Complete dataset folder

#### 5. Nurses (Healthcare Worker Stress)
- **Source**: Research dataset (contact authors)
- **Size**: ~50MB
- **Description**: Healthcare worker stress analysis
- **Files needed**: Complete dataset folder

#### 6. DRIVE-DB (Driver Stress Analysis)
- **Source**: [Kaggle - DRIVE-DB](https://www.kaggle.com/datasets/drive-stress-analysis)
- **Size**: ~1.5GB
- **Description**: Driver stress analysis
- **Files needed**: Complete dataset folder

#### 7. Non-EEG (Non-electroencephalographic)
- **Source**: Research dataset (contact authors)
- **Size**: ~100MB
- **Description**: Non-EEG physiological signals
- **Files needed**: Complete dataset folder

### 🔬 Extended Validation Datasets (4 datasets)

#### 8. Enhanced Health
- **Source**: Research dataset (contact authors)
- **Size**: ~80MB
- **Description**: Enhanced health monitoring
- **Files needed**: Complete dataset folder

#### 9. Global Mental Health
- **Source**: Research dataset (contact authors)
- **Size**: ~60MB
- **Description**: Global mental health analysis
- **Files needed**: Complete dataset folder

#### 10. Mental Health Prediction
- **Source**: Research dataset (contact authors)
- **Size**: ~40MB
- **Description**: Mental health prediction
- **Files needed**: Complete dataset folder

#### 11. UWS (Additional Stress Validation)
- **Source**: Research dataset (contact authors)
- **Size**: TBD
- **Description**: Additional stress validation
- **Files needed**: Complete dataset folder

## 📁 Directory Structure Setup

After downloading the datasets, organize them as follows:

```
data_analysis/
├── data/
│   ├── wesad/
│   │   ├── raw/           # Original WESAD files
│   │   ├── extracted/     # Processed WESAD files
│   │   └── processed/     # Final WESAD features
│   ├── mmash/
│   │   ├── raw/
│   │   ├── extracted/
│   │   └── processed/
│   └── [other datasets]/
│       ├── raw/
│       ├── extracted/
│       └── processed/

wesad_analysis/
├── data/
│   ├── stress_datasets/
│   │   ├── swell/
│   │   ├── nurses/
│   │   ├── drive_db/
│   │   ├── non_eeg/
│   │   └── uws/
│   └── [other datasets]/
```

## 🚀 Quick Start

1. **Clone this repository**:
   ```bash
   git clone https://github.com/qd600600/physiology-first-framework-signal-analysis.git
   cd physiology-first-framework-signal-analysis
   ```

2. **Download datasets** from the links above

3. **Organize datasets** according to the directory structure

4. **Install dependencies**:
   ```bash
   pip install -r requirements_unified.txt
   ```

5. **Run analysis**:
   ```bash
   python examples/quick_start.py
   ```

## 📞 Contact for Dataset Access

For datasets marked as "Research dataset (contact authors)", please contact:
- **Email**: [your.email@example.com]
- **GitHub**: [your-github-username]

## 📋 Dataset Usage License

Please ensure you comply with the license terms for each dataset:
- **WESAD**: Available under research license
- **MMASH**: PhysioNet license
- **Other datasets**: Contact authors for license terms

## ⚠️ Important Notes

- Total dataset size: ~4GB
- Ensure sufficient disk space
- Some datasets require registration/approval
- Follow individual dataset citation requirements
- Respect privacy and ethical guidelines

## 🔗 Alternative Download Sources

If primary sources are unavailable, alternative sources may be available:
- **Baidu Netdisk**: [Link to be provided]
- **Google Drive**: [Link to be provided]
- **Figshare**: [Link to be provided]

*Note: Alternative sources will be updated as they become available.*



