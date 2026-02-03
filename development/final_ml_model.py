import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

import final_ml_model
warnings.filterwarnings('ignore')

class FinalScholarshipModel:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.feature_names = None
        self.feature_importance = None
        
    def load_clean_data(self):
        """Load the cleaned data"""
        try:
            df = pd.read_csv('clean_ml_data.csv')
            print(f"✅ Loaded clean data: {df.shape}")
            
            X = df.drop('target', axis=1)
            y = df['target'].astype(int)  # Convert to integer
            
            self.feature_names = list(X.columns)
            
            print(f"Features: {self.feature_names}")
            print(f"Target distribution: {Counter(y)}")
            
            return X, y
            
        except FileNotFoundError:
            print("❌ clean_ml_data.csv not found. Run check_data.py first.")
            return None, None
    
    def train_best_model(self, X, y):
        """Train the best performing model"""
        
        print("\n🚀 TRAINING FINAL MODEL")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Best model based on previous analysis
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',  # Handle imbalance
            n_jobs=-1
        )
        
        print("📊 Training Random Forest model...")
        
        # Cross-validation
        cv_roc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_pr_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='average_precision')
        
        print(f"CV ROC-AUC: {cv_roc_scores.mean():.4f} (+/- {cv_roc_scores.std() * 2:.4f})")
        print(f"CV PR-AUC:  {cv_pr_scores.mean():.4f} (+/- {cv_pr_scores.std() * 2:.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Test evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        test_pr_auc = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n📈 TEST PERFORMANCE:")
        print(f"ROC-AUC: {test_roc_auc:.4f}")
        print(f"PR-AUC:  {test_pr_auc:.4f}")
        
        self.model = model
        self.model_name = "Random Forest (Balanced)"
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def detailed_evaluation(self, X_test, y_test, y_pred, y_pred_proba):
        """Detailed model evaluation"""
        
        print(f"\n📊 DETAILED EVALUATION")
        print("=" * 50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Rejected', 'Accepted'],
                                  digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Confusion Matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Rejected', 'Accepted'],
                   yticklabels=['Rejected', 'Accepted'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Plot 2: Precision-Recall Curve
        plt.subplot(1, 3, 2)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Probability Distribution
        plt.subplot(1, 3, 3)
        accepted_proba = y_pred_proba[y_test == 1]
        rejected_proba = y_pred_proba[y_test == 0]
        
        plt.hist(rejected_proba, bins=50, alpha=0.7, label='Rejected', color='red')
        plt.hist(accepted_proba, bins=50, alpha=0.7, label='Accepted', color='green')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Probability Distribution')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Probability analysis
        print(f"\nProbability Analysis:")
        print(f"Accepted students - Mean probability: {accepted_proba.mean():.4f}")
        print(f"Rejected students - Mean probability: {rejected_proba.mean():.4f}")
        
        # Threshold analysis
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        print(f"\nThreshold Analysis:")
        print("Threshold | Precision | Recall | F1-Score")
        print("-" * 45)
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            if y_pred_thresh.sum() > 0:  # Avoid division by zero
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(y_test, y_pred_thresh)
                recall = recall_score(y_test, y_pred_thresh)
                f1 = f1_score(y_test, y_pred_thresh)
                
                print(f"  {threshold:.1f}     |   {precision:.3f}   |  {recall:.3f}  |  {f1:.3f}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        
        if self.model is None:
            return
        
        print(f"\n🔍 FEATURE IMPORTANCE")
        print("=" * 30)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance:")
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance for Scholarship Prediction')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.feature_importance = importance_df
        return importance_df
    
    def create_predictor(self):
        """Create prediction function"""
        
        def predict_scholarship_eligibility(student_data):
            """
            Predict scholarship eligibility for a student
            
            Args:
                student_data: dict with keys:
                    - gender_encoded: 0=Female, 1=Male
                    - community_encoded: 0=General, 1=Minority, 2=OBC, 3=SC/ST
                    - religion_encoded: 0=Christian, 1=Hindu, 2=Muslim, 3=Others  
                    - location_encoded: 0=In, 1=Out
                    - has_disability_binary: 0=No, 1=Yes
                    - sports_participation_binary: 0=No, 1=Yes
                    - income_numerical: Annual income in rupees
            
            Returns:
                Dictionary with prediction results
            """
            
            if self.model is None:
                return {"error": "Model not trained"}
            
            # Create DataFrame
            df = pd.DataFrame([student_data])
            
            # Ensure all features are present
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                return {"error": f"Missing features: {missing}"}
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Predict
            probability = self.model.predict_proba(df)[0, 1]
            prediction = self.model.predict(df)[0]
            
            # Create recommendation
            if probability >= 0.8:
                recommendation = "🎉 Excellent chance! Apply immediately!"
                confidence = "Very High"
            elif probability >= 0.6:
                recommendation = "✅ Good chance! Strongly recommended to apply."
                confidence = "High"
            elif probability >= 0.4:
                recommendation = "👍 Moderate chance. Apply with strong documents."
                confidence = "Medium"
            elif probability >= 0.2:
                recommendation = "📝 Lower chance, but worth trying."
                confidence = "Low"
            else:
                recommendation = "📋 Very low chance. Focus on profile improvement."
                confidence = "Very Low"
            
            return {
                "probability": round(probability, 4),
                "prediction": int(prediction),
                "recommendation": recommendation,
                "confidence": confidence,
                "percentage": f"{probability * 100:.2f}%"
            }
        
        return predict_scholarship_eligibility
    
    def save_model(self, filename='production/final_scholarship_model.pkl'):
        """Save the trained model"""

        if self.model is None:
            print("❌ No model to save")
            return

        import os
        os.makedirs("production", exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, filename)
        print(f"💾 Model saved to: {filename}")

    
    def load_model(self, filename='final_scholarship_model.pkl'):
        """Load saved model"""
        
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data.get('feature_importance')
            
            print(f"✅ Model loaded: {self.model_name}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file not found: {filename}")
            return False

def main():
    """Main training function"""
    
    print("🚀 FINAL SCHOLARSHIP MODEL TRAINER")
    print("=" * 60)
    
    # Initialize
    model_trainer = FinalScholarshipModel()
    
    # Load clean data
    X, y = model_trainer.load_clean_data()
    if X is None:
        return None
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = model_trainer.train_best_model(X, y)
    
    # Detailed evaluation
    model_trainer.detailed_evaluation(X_test, y_test, y_pred, y_pred_proba)
    
    # Feature importance
    model_trainer.analyze_feature_importance()
    
    # Save model
    model_trainer.save_model()
    
    # Create predictor
    predictor = model_trainer.create_predictor()
    
    print(f"\n✅ MODEL TRAINING COMPLETE!")
    
    # Test with examples
    print(f"\n🧪 TESTING WITH EXAMPLES:")
    
    examples = [
        {
            "name": "High-chance student",
            "profile": {
                'gender_encoded': 0,  # Female
                'community_encoded': 3,  # SC/ST
                'religion_encoded': 1,  # Hindu
                'location_encoded': 0,  # In
                'has_disability_binary': 1,  # Has disability
                'sports_participation_binary': 1,  # Sports
                'income_numerical': 80000  # Low income
            }
        },
        {
            "name": "Medium-chance student", 
            "profile": {
                'gender_encoded': 1,  # Male
                'community_encoded': 2,  # OBC
                'religion_encoded': 1,  # Hindu
                'location_encoded': 0,  # In
                'has_disability_binary': 0,  # No disability
                'sports_participation_binary': 1,  # Sports
                'income_numerical': 180000  # Medium income
            }
        },
        {
            "name": "Low-chance student",
            "profile": {
                'gender_encoded': 1,  # Male
                'community_encoded': 0,  # General
                'religion_encoded': 1,  # Hindu
                'location_encoded': 1,  # Out
                'has_disability_binary': 0,  # No disability
                'sports_participation_binary': 0,  # No sports
                'income_numerical': 400000  # High income
            }
        }
    ]
    
    for example in examples:
        result = predictor(example["profile"])
        print(f"\n{example['name']}:")
        print(f"  Probability: {result['percentage']}")
        print(f"  {result['recommendation']}")
        print(f"  Confidence: {result['confidence']}")
    
    return model_trainer, predictor

if __name__ == "__main__":
    model_trainer, predictor = main()