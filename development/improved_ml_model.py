import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ImprovedScholarshipMLModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.class_weights = None
        
    def load_data(self, data_path='ml_ready_data.csv', sample_size=None):
        """Load data with optional sampling for faster processing"""
        try:
            print(f"📁 Loading data from {data_path}...")
            df = pd.read_csv(data_path)
            
            print(f"Original data shape: {df.shape}")
            
            # Optional sampling for faster development/testing
            if sample_size and sample_size < len(df):
                print(f"🔄 Sampling {sample_size} records for faster processing...")
                # Use stratified sampling to maintain class distribution
                from sklearn.model_selection import train_test_split
                df_sample, _ = train_test_split(
                    df, train_size=sample_size, random_state=42, 
                    stratify=df['target']
                )
                df = df_sample
                print(f"Sampled data shape: {df.shape}")
            
            # Separate features and target
            if 'target' in df.columns:
                X = df.drop('target', axis=1)
                y = df['target']
                
                self.feature_names = list(X.columns)
                
                print(f"Features: {self.feature_names}")
                print(f"Target distribution:")
                target_counts = y.value_counts()
                for value, count in target_counts.items():
                    percentage = (count / len(y)) * 100
                    status = "Got scholarship" if value == 1.0 else "Didn't get"
                    print(f"  {int(value)} ({status}): {count:,} ({percentage:.2f}%)")
                
                return X, y
            else:
                print("❌ No 'target' column found")
                return None, None
                
        except FileNotFoundError:
            print(f"❌ File {data_path} not found.")
            return None, None
    
    def handle_imbalanced_data(self, X, y, strategy='class_weight'):
        """Handle imbalanced dataset using different strategies"""
        
        print(f"\n⚖️ HANDLING IMBALANCED DATA - Strategy: {strategy}")
        print("=" * 50)
        
        original_distribution = Counter(y)
        print(f"Original distribution: {original_distribution}")
        
        if strategy == 'class_weight':
            # Calculate class weights
            classes = np.unique(y)
            self.class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, self.class_weights))
            print(f"Calculated class weights: {class_weight_dict}")
            return X, y, class_weight_dict
        
        elif strategy == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)  # 10:1 ratio
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            new_distribution = Counter(y_resampled)
            print(f"After undersampling: {new_distribution}")
            return X_resampled, y_resampled, None
        
        elif strategy == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=42, sampling_strategy=0.1)  # 10:1 ratio
            X_resampled, y_resampled = smote.fit_resample(X, y)
            new_distribution = Counter(y_resampled)
            print(f"After SMOTE: {new_distribution}")
            return X_resampled, y_resampled, None
        
        else:
            return X, y, None
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split the data"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # Maintain class distribution
        )
        
        print(f"\n📊 DATA SPLIT:")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        print(f"Training distribution: {train_dist}")
        print(f"Test distribution: {test_dist}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models_for_imbalanced_data(self, X_train, y_train, class_weight_dict=None):
        """Train models optimized for imbalanced data"""
        
        print("\n🤖 TRAINING MODELS FOR IMBALANCED DATA")
        print("=" * 50)
        
        # Models optimized for imbalanced data
        models_to_try = {
            'Random Forest (Balanced)': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',  # Handle imbalance
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                learning_rate=0.1
            ),
            'Logistic Regression (Balanced)': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                solver='liblinear'
            )
        }
        
        # If we have custom class weights, use them
        if class_weight_dict:
            models_to_try['Random Forest (Custom Weights)'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=-1
            )
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models_to_try.items():
            print(f"\n📊 Training {name}...")
            
            # Use stratified k-fold for imbalanced data
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            roc_scores = []
            pr_scores = []  # Precision-Recall AUC (better for imbalanced data)
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Predictions
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                
                # Scores
                roc_score = roc_auc_score(y_fold_val, y_pred_proba)
                pr_score = average_precision_score(y_fold_val, y_pred_proba)
                
                roc_scores.append(roc_score)
                pr_scores.append(pr_score)
            
            # Train final model on full training data
            model.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'roc_auc_mean': np.mean(roc_scores),
                'roc_auc_std': np.std(roc_scores),
                'pr_auc_mean': np.mean(pr_scores),
                'pr_auc_std': np.std(pr_scores),
                'roc_scores': roc_scores,
                'pr_scores': pr_scores
            }
            
            print(f"   ROC-AUC: {np.mean(roc_scores):.4f} (+/- {np.std(roc_scores) * 2:.4f})")
            print(f"   PR-AUC:  {np.mean(pr_scores):.4f} (+/- {np.std(pr_scores) * 2:.4f})")
        
        # Find best model based on PR-AUC (better for imbalanced data)
        best_model_name = max(results, key=lambda x: results[x]['pr_auc_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   ROC-AUC: {results[best_model_name]['roc_auc_mean']:.4f}")
        print(f"   PR-AUC:  {results[best_model_name]['pr_auc_mean']:.4f}")
        
        self.models = results
        return results
    
    def evaluate_model_detailed(self, X_test, y_test):
        """Detailed evaluation for imbalanced data"""
        
        if self.best_model is None:
            print("❌ No trained model found")
            return
        
        print(f"\n📈 DETAILED EVALUATION: {self.best_model_name}")
        print("=" * 60)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Standard metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score:  {pr_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Rejected', 'Accepted']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 5))
        
        # Confusion Matrix Plot
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Accepted'],
                   yticklabels=['Rejected', 'Accepted'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AUC={pr_auc:.3f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Probability distribution analysis
        print(f"\nPrediction Probability Distribution:")
        accepted_proba = y_pred_proba[y_test == 1]
        rejected_proba = y_pred_proba[y_test == 0]
        
        print(f"Accepted students - Mean probability: {accepted_proba.mean():.4f}")
        print(f"Rejected students - Mean probability: {rejected_proba.mean():.4f}")
        
        return roc_auc, pr_auc, y_pred, y_pred_proba
    
    def analyze_feature_importance(self, X_train):
        """Analyze feature importance"""
        
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            print("❌ Feature importance not available")
            return None
        
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance Ranking:")
        for idx, row in importance_df.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance for Scholarship Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def create_recommendation_function(self):
        """Create a function to recommend scholarships to new students"""
        
        def recommend_scholarships(student_profile, top_k=10):
            """
            Recommend scholarships for a student
            
            student_profile: dict with keys matching your features
            Example: {
                'gender_encoded': 0,  # 0=Female, 1=Male
                'community_encoded': 1,  # 0=General, 1=Minority, 2=OBC, 3=SC/ST
                'religion_encoded': 1,  # 0=Christian, 1=Hindu, 2=Muslim, 3=Others
                'location_encoded': 0,  # 0=In, 1=Out
                'has_disability_binary': 0,  # 0=No, 1=Yes
                'sports_participation_binary': 1,  # 0=No, 1=Yes
                'income_numerical': 150000  # Annual income in rupees
            }
            """
            
            if self.best_model is None:
                return "❌ No trained model available"
            
            # Convert to DataFrame
            student_df = pd.DataFrame([student_profile])
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(student_df.columns)
            if missing_features:
                return f"❌ Missing features: {missing_features}"
            
            # Get prediction
            probability = self.best_model.predict_proba(student_df[self.feature_names])[0, 1]
            prediction = self.best_model.predict(student_df[self.feature_names])[0]
            
            # Create recommendation
            if probability > 0.8:
                recommendation = "🎉 Excellent chance! Strongly recommended to apply."
            elif probability > 0.6:
                recommendation = "👍 Good chance! Recommended to apply."
            elif probability > 0.4:
                recommendation = "🤔 Moderate chance. Consider applying with strong application."
            elif probability > 0.2:
                recommendation = "📝 Lower chance, but still worth trying."
            else:
                recommendation = "📋 Low chance, focus on improving profile or other scholarships."
            
            return {
                'probability': round(probability, 4),
                'prediction': int(prediction),
                'recommendation': recommendation,
                'confidence': 'High' if max(probability, 1-probability) > 0.8 else 'Medium'
            }
        
        return recommend_scholarships
    
    def save_model(self, model_path='scholarship_model_v2.pkl'):
        """Save the trained model"""
        
        if self.best_model is None:
            print("❌ No trained model to save")
            return
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")

def main_improved():
    """Main function with improved handling for imbalanced data"""
    
    print("🚀 IMPROVED SCHOLARSHIP ML MODEL TRAINER")
    print("=" * 60)
    
    # Initialize
    ml_model = ImprovedScholarshipMLModel()
    
    # Load data (use sampling for faster development)
    print("📁 Loading data...")
    X, y = ml_model.load_data(sample_size=50000)  # Use 50K samples for faster training
    
    if X is None or y is None:
        print("❌ Could not load data.")
        return
    
    # Handle imbalanced data
    print("⚖️ Handling imbalanced data...")
    X_balanced, y_balanced, class_weights = ml_model.handle_imbalanced_data(
        X, y, strategy='class_weight'  # Try: 'class_weight', 'undersample', 'smote'
    )
    
    # Prepare data
    print("🔄 Preparing data...")
    X_train, X_test, y_train, y_test = ml_model.prepare_data(X_balanced, y_balanced)
    
    # Train models
    print("🚀 Training models...")
    results = ml_model.train_models_for_imbalanced_data(X_train, y_train, class_weights)
    
    # Evaluate
    print("📊 Evaluating model...")
    roc_auc, pr_auc, y_pred, y_pred_proba = ml_model.evaluate_model_detailed(X_test, y_test)
    
    # Feature importance
    print("🔍 Analyzing features...")
    importance_df = ml_model.analyze_feature_importance(X_train)
    
    # Save model
    print("💾 Saving model...")
    ml_model.save_model()
    
    # Create recommendation function
    recommend_func = ml_model.create_recommendation_function()
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"Best Model: {ml_model.best_model_name}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    example_student = {
        'gender_encoded': 0,  # Female
        'community_encoded': 3,  # SC/ST
        'religion_encoded': 1,  # Hindu
        'location_encoded': 0,  # In
        'has_disability_binary': 0,  # No disability
        'sports_participation_binary': 1,  # Sports participation
        'income_numerical': 150000  # 1.5L income
    }
    
    result = recommend_func(example_student)
    print(f"Result: {result}")
    
    return ml_model, recommend_func

if __name__ == "__main__":
    ml_model, recommend_func = main_improved()