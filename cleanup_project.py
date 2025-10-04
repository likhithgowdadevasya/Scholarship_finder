# cleanup_project.py - Organize your project files

import os
import shutil
from pathlib import Path

def cleanup_and_organize():
    """Clean up and organize project files"""
    
    print("🧹 CLEANING UP PROJECT STRUCTURE")
    print("=" * 50)
    
    # Create organized folder structure
    folders_to_create = [
        'production',           # Final production files
        'development',          # Development/testing scripts
        'archived',            # Old files to keep for reference
        'outputs',             # Generated files (images, CSVs)
        'documentation'        # Project documentation
    ]
    
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created folder: {folder}/")
    
    # Define file movements
    file_movements = {
        # Production files (what you need to run the app)
        'production': [
            'enhanced_app.py',
            'scholarship_model_quick.pkl',
            'clean_ml_data.csv',
        ],
        
        # Development scripts (used during development)
        'development': [
            'check_data.py',
            'data_analyzer.py',
            'ml_model.py',
            'improved_ml_model.py',
            'final_ml_model.py',
        ],
        
        # Archived (old versions)
        'archived': [
            'quick_app.py',
            'scholarship_app.py',
            'ml_ready_data.csv',
        ],
        
        # Generated outputs
        'outputs': [
            'model_evaluation.png',
            'scholarship_analysis.png',
        ]
    }
    
    # Move files
    for destination, files in file_movements.items():
        for file in files:
            if os.path.exists(file):
                try:
                    # Don't move if already in destination
                    if not file.startswith(destination):
                        shutil.copy2(file, destination)
                        print(f"📁 Copied {file} → {destination}/")
                except Exception as e:
                    print(f"⚠️ Could not move {file}: {e}")
    
    print("\n✅ Cleanup complete!")
    print("\n📋 RECOMMENDED PROJECT STRUCTURE:")
    print("""
    scholarship_recommender/
    ├── production/
    │   ├── app.py                    # Renamed from enhanced_app.py
    │   ├── scholarship_model.pkl     # ML model
    │   └── scholarships.db           # Database
    ├── development/
    │   ├── train_model.py            # Model training script
    │   └── data_processing.py        # Data processing utilities
    ├── data/
    │   └── raw/                      # Original Excel files
    ├── documentation/
    │   ├── README.md
    │   ├── USER_GUIDE.md
    │   └── PROJECT_REPORT.md
    ├── requirements.txt
    └── .gitignore
    """)
    
    # Create requirements.txt
    create_requirements()
    
    # Create README.md
    create_readme()
    
    # Create .gitignore
    create_gitignore()
    
    print("\n🎯 WHAT TO DO NOW:")
    print("1. Review the organized folders")
    print("2. Move essential files to production/")
    print("3. Update your run commands:")
    print("   streamlit run production/app.py")
    print("4. Delete development files you don't need")

def create_requirements():
    """Create requirements.txt"""
    requirements = """# Core dependencies
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2

# Visualization
plotly==5.17.0
matplotlib==3.7.2
seaborn==0.12.2

# Database
sqlite3

# Data processing
openpyxl==3.1.2
xlrd==2.0.1

# ML utilities
imbalanced-learn==0.11.0

# Optional
jupyter==1.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_readme():
    """Create README.md"""
    readme = """# Scholarship Finder & Recommender System

AI-Powered Scholarship Matching Using Machine Learning

## Project Overview

This system helps students find scholarships that match their profile using machine learning algorithms trained on real scholarship data.

### Features
- ML-powered scholarship recommendations
- Real-time database integration
- Personalized matching algorithm
- Web-based user interface
- Admin panel for scholarship management

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd scholarship_recommender
```

2. **Create virtual environment**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\\Scripts\\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run production/enhanced_app.py
```

Or if using database version:
```bash
streamlit run production/database_app.py
```

## Model Performance

- **ROC-AUC Score:** 0.776
- **PR-AUC Score:** 0.118
- **Training Data:** 221,183 student records
- **Scholarship Success Rate:** 5.4%

## Project Structure

```
scholarship_recommender/
├── production/          # Production-ready files
│   ├── enhanced_app.py  # Main application
│   ├── scholarship_model_quick.pkl  # Trained ML model
│   └── clean_ml_data.csv  # Clean dataset
├── development/         # Development scripts
│   ├── final_ml_model.py  # Model training
│   └── check_data.py      # Data processing
├── data/               # Raw datasets
├── documentation/      # Project documentation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Team Members

- Rahul K U (4SF23CS163)
- Nandan V Poojary (4SF23CS126)
- Rohit L (4SF23CS171)
- Likhith Gowda (4SF23CS092)

## Guide

Dr. Shreema B Shetty  
Department of CSE  
Sahyadri College of Engineering & Management

## Technology Stack

- **Backend:** Python, Scikit-learn, Pandas, NumPy
- **Frontend:** Streamlit
- **Database:** SQLite (optional)
- **ML Models:** Random Forest, Gradient Boosting, Logistic Regression
- **Visualization:** Plotly, Matplotlib, Seaborn

## Usage

### For Students:
1. Open the web application
2. Fill in your profile details (gender, community, income, etc.)
3. Click "Find Scholarships"
4. Review personalized recommendations
5. Apply through provided links

### For Administrators:
1. Access the admin panel in the sidebar
2. Add, update, or remove scholarships
3. View analytics and statistics
4. Export data for reports

## Future Enhancements

- Integration with live scholarship APIs
- Mobile application
- Email notifications for deadlines
- Multi-language support
- Advanced analytics dashboard

## License

This project is developed as part of VTU Mini Project requirements (2024-25).

## Contact

For queries related to this project, contact:
- Email: rahul.ku@sahyadri.edu.in
- Institution: Sahyadri College of Engineering & Management, Mangaluru
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("✅ Created README.md")

def create_gitignore():
    """Create .gitignore"""
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.xlsx
*.xls
!sample_data.csv

# Model files
*.pkl
*.joblib

# Database
*.db
*.sqlite

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Outputs
outputs/
*.png
*.jpg

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    
    print("✅ Created .gitignore")

if __name__ == "__main__":
    cleanup_and_organize()