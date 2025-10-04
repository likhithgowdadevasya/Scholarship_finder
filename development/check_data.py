import pandas as pd
import numpy as np

print("🔍 CHECKING DATA QUALITY")
print("=" * 40)

# Load data
df = pd.read_csv('ml_ready_data.csv')
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check target column
if 'target' in df.columns:
    print(f"\n🎯 TARGET COLUMN ANALYSIS:")
    print(f"Data type: {df['target'].dtype}")
    print(f"Unique values: {df['target'].unique()}")
    print(f"Value counts:")
    print(df['target'].value_counts(dropna=False))
    print(f"NaN count: {df['target'].isna().sum()}")
    
    # Clean target
    print(f"\n🧹 CLEANING TARGET:")
    df_clean = df.dropna(subset=['target'])
    print(f"Shape after removing NaN targets: {df_clean.shape}")
    
    # Check all columns for NaN
    print(f"\n❓ NaN VALUES IN ALL COLUMNS:")
    for col in df_clean.columns:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count}")
    
    # Save clean data
    df_clean.to_csv('clean_ml_data.csv', index=False)
    print(f"\n💾 Saved clean data to: clean_ml_data.csv")
    
    # Try simple model
    print(f"\n🤖 TESTING SIMPLE MODEL:")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    
    # Fill any remaining NaN in features
    X = X.fillna(0)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Small sample for testing
    X_small, _, y_small, _ = train_test_split(X, y, train_size=1000, random_state=42, stratify=y)
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_small, y_small)
    
    print(f"✅ Simple model trained successfully!")
    print(f"Feature importance: {dict(zip(X.columns, model.feature_importances_))}")
    
else:
    print("❌ No 'target' column found!")
    