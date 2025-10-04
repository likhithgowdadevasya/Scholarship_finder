# Scholarship Finder & Recommender System

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
source env/bin/activate  # On Windows: env\Scripts\activate
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
