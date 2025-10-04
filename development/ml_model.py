import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ScholarshipMLModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_data(self, data_path='ml_ready_data.csv'):
        """Load the processed ML data"""
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Loaded data: {df.shape}")
            
            # Separate features and target
            if 'target' in df.columns:
                X = df.drop('target', axis=1)
                y = df['target']
                print(f"Features: {list(X.columns)}")
                print(f"Target distribution: {y.value_counts().to_dict()}")
                return X, y
            else:
                print("❌ No 'target' column found")
                return None, None
                
        except FileNotFoundError:
            print(f"❌ File {data_path} not found. Please run data analysis first.")
            return None, None
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split and scale the data"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features (important for logistic regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training target distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def train_multiple_models(self, X_train, y_train, X_train_scaled):
        """Train multiple ML models and compare performance"""
        
        print("\n🤖 TRAINING MULTIPLE MODELS")
        print("=" * 50)
        
        # Define models
        models_to_try = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models_to_try.items():
            print(f"\n📊 Training {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            X_to_use = X_train_scaled if name == 'Logistic Regression' else X_train
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_to_use, y_train, cv=5, scoring='roc_auc')
            
            # Train on full training set
            model.fit(X_to_use, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   ROC-AUC Score: {results[best_model_name]['cv_mean']:.4f}")
        
        self.models = results
        return results
    
    def evaluate_model(self, X_test, y_test, X_test_scaled):
        """Evaluate the best model on test data"""
        
        if self.best_model is None:
            print("❌ No trained model found")
            return
        
        print(f"\n📈 EVALUATING BEST MODEL: {self.best_model_name}")
        print("=" * 50)
        
        # Use appropriate data (scaled for logistic regression)
        X_to_use = X_test_scaled if self.best_model_name == 'Logistic Regression' else X_test
        
        # Predictions
        y_pred = self.best_model.predict(X_to_use)
        y_pred_proba = self.best_model.predict_proba(X_to_use)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return roc_auc, y_pred, y_pred_proba
    
    def analyze_feature_importance(self, X_train):
        """Analyze which features are most important for prediction"""
        
        if self.best_model is None:
            print("❌ No trained model found")
            return
        
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        # Get feature importance (works for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
            self.feature_importance = importance_df
            return importance_df
        
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, model_path='scholarship_model.pkl'):
        """Save the trained model"""
        
        if self.best_model is None:
            print("❌ No trained model to save")
            return
        
        # Save model and scaler
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self, model_path='scholarship_model.pkl'):
        """Load a saved model"""
        
        try:
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.best_model_name = model_data['model_name']
            self.feature_importance = model_data.get('feature_importance')
            
            print(f"✅ Model loaded: {self.best_model_name}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file {model_path} not found")
            return False
    
    def predict_scholarship_chance(self, student_profile):
        """Predict scholarship chance for a new student"""
        
        if self.best_model is None:
            print("❌ No trained model available")
            return None
        
        # Convert to DataFrame for consistent processing
        if isinstance(student_profile, dict):
            student_df = pd.DataFrame([student_profile])
        else:
            student_df = student_profile
        
        # Scale if needed
        if self.best_model_name == 'Logistic Regression':
            student_scaled = self.scaler.transform(student_df)
            prediction_proba = self.best_model.predict_proba(student_scaled)[0, 1]
        else:
            prediction_proba = self.best_model.predict_proba(student_df)[0, 1]
        
        prediction = self.best_model.predict(student_df if self.best_model_name != 'Logistic Regression' else self.scaler.transform(student_df))[0]
        
        return {
            'probability': prediction_proba,
            'prediction': prediction,
            'recommendation': 'High chance' if prediction_proba > 0.7 else 'Medium chance' if prediction_proba > 0.4 else 'Low chance'
        }

def main():
    """Main function to train the ML model"""
    
    print("🤖 SCHOLARSHIP ML MODEL TRAINER")
    print("=" * 60)
    
    # Initialize model trainer
    ml_model = ScholarshipMLModel()
    
    # Load data
    print("\n📁 Loading processed data...")
    X, y = ml_model.load_data()
    
    if X is None or y is None:
        print("❌ Could not load data. Please run data analysis first.")
        return
    
    # Prepare data
    print("\n🔄 Preparing data...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = ml_model.prepare_data(X, y)
    
    # Train models
    print("\n🚀 Training models...")
    results = ml_model.train_multiple_models(X_train, y_train, X_train_scaled)
    
    # Evaluate best model
    print("\n📊 Evaluating model...")
    roc_auc, y_pred, y_pred_proba = ml_model.evaluate_model(X_test, y_test, X_test_scaled)
    
    # Feature importance
    print("\n🔍 Analyzing features...")
    importance_df = ml_model.analyze_feature_importance(X_train)
    
    # Save model
    print("\n💾 Saving model...")
    ml_model.save_model()
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"Best Model: {ml_model.best_model_name}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    example_student = {
        'gender_encoded': 0,  # Adjust based on your encoding
        'income_numerical': 150000,
        'has_disability_binary': 0,
        'sports_participation_binary': 1,
        # Add other features as needed
    }
    
    # Note: You'll need to adjust this based on your actual feature columns
    print("   (Run with actual student data after confirming feature columns)")
    
    return ml_model

if __name__ == "__main__":
    ml_model = main()