import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ScholarshipDataAnalyzer:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.all_data = pd.DataFrame()
        self.excel_files = []
        
    def load_all_excel_files(self):
        """Load all 10 Excel files and combine them"""
        
        # Find all Excel files
        self.excel_files = list(self.data_folder.glob("*.xlsx")) + list(self.data_folder.glob("*.xls"))
        
        print(f"🔍 Found {len(self.excel_files)} Excel files:")
        for i, file in enumerate(self.excel_files, 1):
            print(f"  {i}. {file.name}")
        
        # Load and combine all files
        all_dataframes = []
        
        for file_path in self.excel_files:
            try:
                print(f"\n📊 Processing: {file_path.name}")
                
                # Read Excel file (try first sheet)
                df = pd.read_excel(file_path)
                print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Add source file info
                df['source_file'] = file_path.name
                
                all_dataframes.append(df)
                
            except Exception as e:
                print(f"❌ Error reading {file_path.name}: {e}")
        
        # Combine all data
        if all_dataframes:
            self.all_data = pd.concat(all_dataframes, ignore_index=True)
            print(f"\n✅ Combined dataset: {len(self.all_data)} total records")
            print(f"Columns: {list(self.all_data.columns)}")
        
        return self.all_data
    
    def clean_column_names(self):
        """Standardize column names"""
        
        # Clean column names
        self.all_data.columns = self.all_data.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Standard column mapping
        column_mapping = {
            'annual_income': 'income',
            'annual-per_income': 'income',
            'per_income': 'income',
            'outcome': 'target',  # This is your ML target variable
            'india': 'location',
            'exercise': 'exercise',
            'disability': 'has_disability',
            'sports': 'sports_participation',
            'community': 'community',
            'religion': 'religion',
            'gender': 'gender',
            'education': 'education_level'
        }
        
        # Apply mapping
        self.all_data.rename(columns=column_mapping, inplace=True)
        
        print("📝 Standardized column names:")
        for col in self.all_data.columns:
            print(f"   {col}")
    
    def explore_data_quality(self):
        """Analyze data quality and patterns"""
        
        print("\n📊 DATA QUALITY ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"Total records: {len(self.all_data)}")
        print(f"Total columns: {len(self.all_data.columns)}")
        
        # Missing values
        print("\n❓ Missing Values:")
        missing = self.all_data.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                percentage = (count / len(self.all_data)) * 100
                print(f"   {col}: {count} ({percentage:.1f}%)")
        
        # Target variable analysis (most important!)
        if 'target' in self.all_data.columns:
            print("\n🎯 TARGET VARIABLE ANALYSIS:")
            target_counts = self.all_data['target'].value_counts()
            print(f"   Class distribution:")
            for value, count in target_counts.items():
                percentage = (count / len(self.all_data)) * 100
                status = "Got scholarship" if value == 1 else "Didn't get"
                print(f"   {value} ({status}): {count} ({percentage:.1f}%)")
        
        # Categorical variables
        categorical_cols = ['gender', 'community', 'religion', 'has_disability', 
                          'sports_participation', 'education_level', 'location']
        
        print("\n🏷️ CATEGORICAL VARIABLES:")
        for col in categorical_cols:
            if col in self.all_data.columns:
                unique_vals = self.all_data[col].value_counts()
                print(f"\n   {col.upper()}:")
                for val, count in unique_vals.head(10).items():
                    percentage = (count / len(self.all_data)) * 100
                    print(f"     {val}: {count} ({percentage:.1f}%)")
        
        # Income analysis
        if 'income' in self.all_data.columns:
            print(f"\n💰 INCOME DISTRIBUTION:")
            income_counts = self.all_data['income'].value_counts()
            for income_bracket, count in income_counts.items():
                percentage = (count / len(self.all_data)) * 100
                print(f"   {income_bracket}: {count} ({percentage:.1f}%)")
    
    def analyze_success_patterns(self):
        """Analyze what factors lead to scholarship success"""
        
        if 'target' not in self.all_data.columns:
            print("❌ No target variable found. Cannot analyze success patterns.")
            return
        
        print("\n🔍 SUCCESS PATTERN ANALYSIS")
        print("=" * 50)
        
        # Success rate by different factors
        factors = ['gender', 'community', 'religion', 'has_disability', 
                  'sports_participation', 'income', 'education_level']
        
        for factor in factors:
            if factor in self.all_data.columns:
                print(f"\n📈 Success rate by {factor.upper()}:")
                
                success_by_factor = self.all_data.groupby(factor)['target'].agg([
                    'count', 'sum', 'mean'
                ]).round(3)
                success_by_factor.columns = ['Total_Applications', 'Scholarships_Won', 'Success_Rate']
                
                # Sort by success rate
                success_by_factor = success_by_factor.sort_values('Success_Rate', ascending=False)
                
                for index, row in success_by_factor.iterrows():
                    print(f"   {index}: {row['Scholarships_Won']}/{row['Total_Applications']} = {row['Success_Rate']:.1%}")
    
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        
        print("\n🤖 PREPARING ML FEATURES")
        print("=" * 40)
        
        # Create a copy for ML processing
        ml_data = self.all_data.copy()
        
        # Convert categorical variables to numerical
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['gender', 'community', 'religion', 'education_level', 'location']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in ml_data.columns:
                le = LabelEncoder()
                ml_data[f'{col}_encoded'] = le.fit_transform(ml_data[col].astype(str))
                label_encoders[col] = le
                print(f"✅ Encoded {col}: {le.classes_}")
        
        # Convert Yes/No to 1/0
        yes_no_cols = ['has_disability', 'sports_participation', 'exercise']
        for col in yes_no_cols:
            if col in ml_data.columns:
                ml_data[f'{col}_binary'] = ml_data[col].map({'Yes': 1, 'No': 0})
                print(f"✅ Converted {col} to binary")
        
        # Convert income brackets to numerical (approximate)
        income_mapping = {
            '90-100': 95000,      # 90-100k
            'Upto 1.5L': 75000,   # Up to 1.5 lakh
            '1.5L to 3L': 225000, # 1.5 to 3 lakh average
            '3L to 6L': 450000,   # 3 to 6 lakh average
            'Above 6L': 800000    # Above 6 lakh
        }
        
        if 'income' in ml_data.columns:
            ml_data['income_numerical'] = ml_data['income'].map(income_mapping)
            print(f"✅ Converted income to numerical values")
        
        # Feature selection for ML
        feature_cols = []
        for col in ml_data.columns:
            if col.endswith('_encoded') or col.endswith('_binary') or col == 'income_numerical':
                feature_cols.append(col)
        
        # Prepare final ML dataset
        if 'target' in ml_data.columns and feature_cols:
            X = ml_data[feature_cols]
            y = ml_data['target']
            
            print(f"\n🎯 ML Dataset Ready:")
            print(f"   Features: {len(feature_cols)}")
            print(f"   Feature names: {feature_cols}")
            print(f"   Samples: {len(X)}")
            print(f"   Target distribution: {y.value_counts().to_dict()}")
            
            # Save processed data
            ml_dataset = pd.concat([X, y], axis=1)
            ml_dataset.to_csv('ml_ready_data.csv', index=False)
            print(f"💾 Saved ML-ready data to: ml_ready_data.csv")
            
            return X, y, label_encoders
        else:
            print("❌ Could not prepare ML features")
            return None, None, None
    
    def create_visualizations(self):
        """Create visualization of the data"""
        
        if 'target' not in self.all_data.columns:
            print("No target variable for visualization")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. Target distribution
        plt.subplot(2, 3, 1)
        self.all_data['target'].value_counts().plot(kind='bar')
        plt.title('Scholarship Outcome Distribution')
        plt.xlabel('Outcome (0=No, 1=Yes)')
        plt.ylabel('Count')
        
        # 2. Success rate by gender
        if 'gender' in self.all_data.columns:
            plt.subplot(2, 3, 2)
            success_by_gender = self.all_data.groupby('gender')['target'].mean()
            success_by_gender.plot(kind='bar')
            plt.title('Success Rate by Gender')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
        
        # 3. Success rate by income
        if 'income' in self.all_data.columns:
            plt.subplot(2, 3, 3)
            success_by_income = self.all_data.groupby('income')['target'].mean()
            success_by_income.plot(kind='bar')
            plt.title('Success Rate by Income')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
        
        # 4. Success rate by disability
        if 'has_disability' in self.all_data.columns:
            plt.subplot(2, 3, 4)
            success_by_disability = self.all_data.groupby('has_disability')['target'].mean()
            success_by_disability.plot(kind='bar')
            plt.title('Success Rate by Disability Status')
            plt.ylabel('Success Rate')
        
        # 5. Success rate by sports
        if 'sports_participation' in self.all_data.columns:
            plt.subplot(2, 3, 5)
            success_by_sports = self.all_data.groupby('sports_participation')['target'].mean()
            success_by_sports.plot(kind='bar')
            plt.title('Success Rate by Sports Participation')
            plt.ylabel('Success Rate')
        
        # 6. Success rate by community
        if 'community' in self.all_data.columns:
            plt.subplot(2, 3, 6)
            success_by_community = self.all_data.groupby('community')['target'].mean()
            success_by_community.plot(kind='bar')
            plt.title('Success Rate by Community')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('scholarship_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualizations saved as 'scholarship_analysis.png'")

def main():
    """Main function to run the analysis"""
    
    print("🚀 SCHOLARSHIP DATA ANALYZER")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ScholarshipDataAnalyzer('data/')  # Update path to your Excel files
    
    # Step 1: Load all Excel files
    print("\n📁 STEP 1: Loading Excel files...")
    data = analyzer.load_all_excel_files()
    
    if data.empty:
        print("❌ No data loaded. Please check your file paths.")
        return
    
    # Step 2: Clean column names
    print("\n🧹 STEP 2: Cleaning data...")
    analyzer.clean_column_names()
    
    # Step 3: Explore data quality
    print("\n🔍 STEP 3: Analyzing data quality...")
    analyzer.explore_data_quality()
    
    # Step 4: Analyze success patterns
    print("\n📈 STEP 4: Analyzing success patterns...")
    analyzer.analyze_success_patterns()
    
    # Step 5: Prepare ML features
    print("\n🤖 STEP 5: Preparing ML features...")
    X, y, encoders = analyzer.prepare_ml_features()
    
    # Step 6: Create visualizations
    print("\n📊 STEP 6: Creating visualizations...")
    analyzer.create_visualizations()
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("\nNext steps:")
    print("1. Review the generated visualizations")
    print("2. Check the 'ml_ready_data.csv' file")
    print("3. Use this data to train your ML models")
    
    return analyzer, X, y, encoders

if __name__ == "__main__":
    analyzer, X, y, encoders = main()