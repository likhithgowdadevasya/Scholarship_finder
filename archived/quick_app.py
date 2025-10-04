# quick_train_and_run.py - Combines model training and app in one file

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Model training function
def train_quick_model():
    """Quick model training function"""
    print("🚀 Training model...")
    
    try:
        # Load clean data
        df = pd.read_csv('clean_ml_data.csv')
        print(f"Loaded data: {df.shape}")
        
        # Prepare data
        X = df.drop('target', axis=1)
        y = df['target'].astype(int)
        
        # Quick train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,  # Fewer trees for speed
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Quick evaluation
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        
        # Save model
        model_data = {
            'model': model,
            'feature_names': list(X.columns),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        joblib.dump(model_data, 'scholarship_model_quick.pkl')
        print("✅ Model saved as 'scholarship_model_quick.pkl'")
        
        return model_data
        
    except FileNotFoundError:
        print("❌ clean_ml_data.csv not found. Please run check_data.py first.")
        return None
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return None

# Streamlit app
def run_app():
    st.set_page_config(
        page_title="Scholarship Recommender",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Scholarship Finder & Recommender")
    st.subheader("AI-Powered Scholarship Matching for Students")
    
    # Try to load model
    model_data = None
    try:
        model_data = joblib.load('scholarship_model_quick.pkl')
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Training new model...")
        
        # Train model button
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training model... This may take a few minutes..."):
                model_data = train_quick_model()
            
            if model_data:
                st.success("✅ Model trained and saved successfully!")
                st.rerun()  # Refresh the app
            else:
                st.error("❌ Failed to train model. Please check your data files.")
                st.stop()
    
    if model_data is None:
        st.info("👆 Please train the model first by clicking the button above.")
        st.stop()
    
    # Get model components
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Display model performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROC-AUC Score", f"{model_data.get('roc_auc', 0):.3f}")
    with col2:
        st.metric("PR-AUC Score", f"{model_data.get('pr_auc', 0):.3f}")
    with col3:
        st.metric("Model Status", "Ready ✅")
    
    # Sidebar for input
    st.sidebar.header("📝 Student Profile")
    
    # Input fields
    with st.sidebar:
        # Gender
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_encoded = 0 if gender == "Female" else 1
        
        # Community
        community = st.selectbox("Community", ["General", "Minority", "OBC", "SC/ST"])
        community_mapping = {"General": 0, "Minority": 1, "OBC": 2, "SC/ST": 3}
        community_encoded = community_mapping[community]
        
        # Religion
        religion = st.selectbox("Religion", ["Christian", "Hindu", "Muslim", "Others"])
        religion_mapping = {"Christian": 0, "Hindu": 1, "Muslim": 2, "Others": 3}
        religion_encoded = religion_mapping[religion]
        
        # Location
        location = st.selectbox("Location", ["In-State", "Out-of-State"])
        location_encoded = 0 if location == "In-State" else 1
        
        # Disability
        has_disability = st.selectbox("Disability Status", ["No", "Yes"])
        disability_encoded = 1 if has_disability == "Yes" else 0
        
        # Sports
        sports = st.selectbox("Sports Participation", ["No", "Yes"])
        sports_encoded = 1 if sports == "Yes" else 0
        
        # Income
        income = st.number_input(
            "Annual Family Income (₹)",
            min_value=0,
            max_value=2000000,
            value=200000,
            step=10000
        )
        
        # Predict button
        if st.button("🔮 Get Recommendation", type="primary"):
            # Prepare input data
            student_data = {
                'gender_encoded': gender_encoded,
                'community_encoded': community_encoded,
                'religion_encoded': religion_encoded,
                'location_encoded': location_encoded,
                'has_disability_binary': disability_encoded,
                'sports_participation_binary': sports_encoded,
                'income_numerical': income
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([student_data])
            input_df = input_df[feature_names]  # Ensure correct order
            
            # Predict
            probability = model.predict_proba(input_df)[0, 1]
            prediction = model.predict(input_df)[0]
            percentage = probability * 100
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Main results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Recommendation
                if percentage >= 70:
                    st.success(f"🎉 **Excellent chance! ({percentage:.1f}%)**")
                    st.write("✅ Strongly recommended to apply immediately!")
                elif percentage >= 50:
                    st.info(f"👍 **Good chance! ({percentage:.1f}%)**")
                    st.write("✅ Recommended to apply with strong documents.")
                elif percentage >= 30:
                    st.warning(f"🤔 **Moderate chance ({percentage:.1f}%)**")
                    st.write("📝 Consider applying, focus on strong application.")
                else:
                    st.error(f"📋 **Lower chance ({percentage:.1f}%)**")
                    st.write("🔄 Consider profile improvement or other scholarships.")
                
                # Profile summary
                st.subheader("📋 Your Profile")
                profile_data = {
                    'Category': ['Gender', 'Community', 'Religion', 'Location', 'Disability', 'Sports', 'Income'],
                    'Value': [gender, community, religion, location, has_disability, sports, f"₹{income:,}"]
                }
                st.dataframe(pd.DataFrame(profile_data), hide_index=True)
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    title={'text': "Success Probability"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Main content area
    if 'student_data' not in locals():
        st.markdown("""
        ## 🌟 How to Use This Tool
        
        1. **Fill your profile** in the sidebar with accurate information
        2. **Click 'Get Recommendation'** to analyze your eligibility
        3. **Review your results** and follow the suggestions
        4. **Apply to matching scholarships** with confidence!
        
        ### Key Features:
        - ✅ **AI-powered predictions** based on real scholarship data
        - ✅ **Personalized recommendations** for your profile
        - ✅ **Success probability** scoring
        - ✅ **Profile analysis** and improvement suggestions
        
        **Ready to find your perfect scholarship match?**
        """)

if __name__ == "__main__":
    run_app()