#!/usr/bin/env python3
"""
Quick Start Example for Physiological Signal Analysis

This script demonstrates the basic usage of the physiological signal analysis framework.
It shows how to load data, run basic analysis, and generate results.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function demonstrating basic usage."""
    
    print("🚀 Physiological Signal Analysis - Quick Start")
    print("=" * 50)
    
    # 1. Check if data directory exists
    data_dir = project_root / "data_analysis" / "data"
    if not data_dir.exists():
        print("⚠️  Data directory not found. Please ensure data files are available.")
        print(f"Expected location: {data_dir}")
        return
    
    # 2. Demonstrate basic data loading
    print("\n📊 Loading sample data...")
    try:
        # Look for sample data files
        sample_files = list(data_dir.glob("*.csv"))[:3]  # Get first 3 CSV files
        
        if sample_files:
            print(f"Found {len(sample_files)} data files:")
            for file in sample_files:
                print(f"  - {file.name}")
                
            # Load first file as example
            sample_data = pd.read_csv(sample_files[0])
            print(f"\n📈 Sample data shape: {sample_data.shape}")
            print(f"Columns: {list(sample_data.columns)}")
            
        else:
            print("No CSV files found in data directory")
            
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # 3. Demonstrate basic analysis functions
    print("\n🔬 Running basic analysis...")
    
    # Create sample physiological data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate HRV data
    hrv_data = np.random.normal(50, 15, n_samples)  # SDNN values
    time_points = np.arange(n_samples)
    
    # Calculate basic statistics
    mean_hrv = np.mean(hrv_data)
    std_hrv = np.std(hrv_data)
    
    print(f"📊 HRV Analysis Results:")
    print(f"  - Mean SDNN: {mean_hrv:.2f} ms")
    print(f"  - Std SDNN: {std_hrv:.2f} ms")
    print(f"  - Samples: {n_samples}")
    
    # 4. Generate simple visualization
    print("\n📈 Generating visualization...")
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot HRV time series
        plt.subplot(2, 1, 1)
        plt.plot(time_points[:100], hrv_data[:100], 'b-', alpha=0.7)
        plt.title('Sample HRV Time Series (First 100 points)')
        plt.xlabel('Time Points')
        plt.ylabel('SDNN (ms)')
        plt.grid(True, alpha=0.3)
        
        # Plot HRV distribution
        plt.subplot(2, 1, 2)
        plt.hist(hrv_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('HRV Distribution')
        plt.xlabel('SDNN (ms)')
        plt.ylabel('Frequency')
        plt.axvline(mean_hrv, color='red', linestyle='--', label=f'Mean: {mean_hrv:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = project_root / "examples" / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "quick_start_analysis.png", dpi=300, bbox_inches='tight')
        print(f"📁 Plot saved to: {output_dir / 'quick_start_analysis.png'}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error generating plot: {e}")
    
    # 5. Demonstrate model loading (if available)
    print("\n🤖 Checking for available models...")
    
    models_dir = project_root / "data_analysis" / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
        if model_files:
            print(f"Found {len(model_files)} trained models:")
            for model_file in model_files[:5]:  # Show first 5
                print(f"  - {model_file.name}")
        else:
            print("No trained models found")
    else:
        print("Models directory not found")
    
    # 6. Summary
    print("\n✅ Quick Start Complete!")
    print("\nNext Steps:")
    print("1. Check the documentation in docs/ folder")
    print("2. Run the full analysis pipeline")
    print("3. Explore the Jupyter notebooks")
    print("4. Check the paper preprint: https://doi.org/10.2139/ssrn.5589750")
    
    print(f"\n📚 Project Documentation:")
    print(f"  - README: {project_root / 'README.md'}")
    print(f"  - Contributing: {project_root / 'CONTRIBUTING.md'}")
    print(f"  - Paper: {project_root / 'paper' / 'preprint.md'}")

if __name__ == "__main__":
    main()
