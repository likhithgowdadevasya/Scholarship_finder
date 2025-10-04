import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Scholarship Finder & Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ScholarshipApp:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_data = joblib.load('final_scholarship_model.pkl')
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            return True
        except FileNotFoundError:
            st.error("❌ Model file not found. Please train the model first by running final_ml_model.py")
            return False
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def predict_scholarship(self, student_data):
        """Predict scholarship eligibility"""
        if self.model is None:
            return None
        
        # Create DataFrame
        df = pd.DataFrame([student_data])
        df = df[self.feature_names]  # Ensure correct order
        
        # Predict
        probability = self.model.predict_proba(df)[0, 1]
        prediction = self.model.predict(df)[0]
        
        # Create detailed response
        if probability >= 0.8:
            recommendation = "🎉 Excellent chance! Apply immediately!"
            confidence = "Very High"
            color = "green"
        elif probability >= 0.6:
            recommendation = "✅ Good chance! Strongly recommended to apply."
            confidence = "High"
            color = "lightgreen"
        elif probability >= 0.4:
            recommendation = "👍 Moderate chance. Apply with strong documents."
            confidence = "Medium"
            color = "orange"
        elif probability >= 0.2:
            recommendation = "📝 Lower chance, but worth trying."
            confidence = "Low"
            color = "salmon"
        else:
            recommendation = "📋 Very low chance. Focus on profile improvement."
            confidence = "Very Low"
            color = "red"
        
        return {
            "probability": probability,
            "prediction": int(prediction),
            "recommendation": recommendation,
            "confidence": confidence,
            "percentage": probability * 100,
            "color": color
        }

def main():
    # Initialize app
    app = ScholarshipApp()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎓 Scholarship Finder & Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Scholarship Matching for Students</p>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if app.model is None:
        st.error("Please ensure the model is trained and saved first.")
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📝 Student Profile")
    st.sidebar.markdown("Please fill in your details below:")
    
    # Input fields
    with st.sidebar:
        st.subheader("Personal Information")
        
        # Gender
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            help="Select your gender"
        )
        gender_encoded = 0 if gender == "Female" else 1
        
        # Community
        community = st.selectbox(
            "Community Category",
            options=["General", "Minority", "OBC", "SC/ST"],
            help="Select your community category"
        )
        community_mapping = {"General": 0, "Minority": 1, "OBC": 2, "SC/ST": 3}
        community_encoded = community_mapping[community]
        
        # Religion
        religion = st.selectbox(
            "Religion",
            options=["Christian", "Hindu", "Muslim", "Others"],
            help="Select your religion"
        )
        religion_mapping = {"Christian": 0, "Hindu": 1, "Muslim": 2, "Others": 3}
        religion_encoded = religion_mapping[religion]
        
        # Location
        location = st.selectbox(
            "Location",
            options=["In-State", "Out-of-State"],
            help="Are you from the same state as the scholarship?"
        )
        location_encoded = 0 if location == "In-State" else 1
        
        st.subheader("Additional Information")
        
        # Disability
        has_disability = st.selectbox(
            "Do you have any disability?",
            options=["No", "Yes"],
            help="Select if you have any certified disability"
        )
        disability_encoded = 1 if has_disability == "Yes" else 0
        
        # Sports
        sports_participation = st.selectbox(
            "Sports Participation",
            options=["No", "Yes"],
            help="Do you participate in sports activities?"
        )
        sports_encoded = 1 if sports_participation == "Yes" else 0
        
        # Income
        st.subheader("Family Income")
        income = st.number_input(
            "Annual Family Income (in ₹)",
            min_value=0,
            max_value=2000000,
            value=200000,
            step=10000,
            help="Enter your annual family income in rupees"
        )
        
        # Predict button
        predict_button = st.button("🔮 Get Scholarship Recommendation", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            # Prepare student data
            student_data = {
                'gender_encoded': gender_encoded,
                'community_encoded': community_encoded,
                'religion_encoded': religion_encoded,
                'location_encoded': location_encoded,
                'has_disability_binary': disability_encoded,
                'sports_participation_binary': sports_encoded,
                'income_numerical': income
            }
            
            # Get prediction
            with st.spinner("Analyzing your profile..."):
                result = app.predict_scholarship(student_data)
            
            if result:
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Main result card
                st.markdown(f"""
                <div class="result-card" style="border-left-color: {result['color']};">
                <h3>📊 Your Scholarship Eligibility</h3>
                <h2 style="color: {result['color']};">{result['percentage']:.2f}%</h2>
                <p style="font-size: 1.2rem;"><strong>{result['recommendation']}</strong></p>
                <p><strong>Confidence Level:</strong> {result['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['percentage'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Scholarship Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': result['color']},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgray"},
                            {'range': [20, 40], 'color': "gray"},
                            {'range': [40, 60], 'color': "orange"},
                            {'range': [60, 80], 'color': "lightgreen"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                st.subheader("📋 Profile Summary")
                
                profile_cols = st.columns(4)
                
                with profile_cols[0]:
                    st.metric("Gender", gender)
                    st.metric("Community", community)
                
                with profile_cols[1]:
                    st.metric("Religion", religion)
                    st.metric("Location", location)
                
                with profile_cols[2]:
                    st.metric("Disability Status", has_disability)
                    st.metric("Sports Participation", sports_participation)
                
                with profile_cols[3]:
                    st.metric("Annual Income", f"₹{income:,}")
                    st.metric("Confidence", result['confidence'])
                
                # Recommendations based on score
                st.subheader("💡 Personalized Recommendations")
                
                if result['percentage'] < 40:
                    st.warning("""
                    **Ways to improve your chances:**
                    - Look for scholarships specifically for your community/category
                    - Participate in sports or extracurricular activities
                    - Apply for need-based scholarships if eligible
                    - Improve your academic performance
                    - Look for scholarships with less competition
                    """)
                elif result['percentage'] < 70:
                    st.info("""
                    **Next steps:**
                    - Prepare a strong application with all required documents
                    - Write compelling essays highlighting your achievements
                    - Get good recommendation letters
                    - Apply early before deadlines
                    - Consider multiple scholarships to increase chances
                    """)
                else:
                    st.success("""
                    **You're in great shape!**
                    - Apply immediately with confidence
                    - Focus on scholarships that match your profile exactly
                    - Prepare thorough documentation
                    - Consider applying to multiple high-probability scholarships
                    - Network with scholarship alumni if possible
                    """)
        
        else:
            # Welcome content
            st.markdown("""
            ## 🌟 Welcome to the AI-Powered Scholarship Recommender!
            
            This system uses machine learning to analyze your profile and predict your likelihood of receiving scholarships. 
            
            ### How it works:
            1. **Fill in your profile** in the sidebar with accurate information
            2. **Click 'Get Recommendation'** to analyze your eligibility  
            3. **Review your results** and personalized recommendations
            4. **Apply to scholarships** with confidence!
            
            ### What we analyze:
            - **Demographics:** Gender, community, religion, location
            - **Special circumstances:** Disability status, sports participation  
            - **Financial need:** Family income level
            
            ### Built with:
            - **Advanced ML algorithms** trained on real scholarship data
            - **221,000+ student records** for accurate predictions
            - **Balanced models** that account for different student backgrounds
            
            **Ready to find your perfect scholarship match? Fill in your details and let's get started!** 🚀
            """)
    
    with col2:
        # Statistics and info panel
        st.subheader("📊 System Statistics")
        
        st.markdown("""
        <div class="metric-card">
        <h4>Model Accuracy</h4>
        <h2>89.2%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h4>Students Analyzed</h4>
        <h2>221,183</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h4>Success Rate</h4>
        <h2>5.4%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎯 Most Important Factors")
        
        # Feature importance (based on your model)
        importance_data = {
            'Factor': ['Income Level', 'Religion', 'Community', 'Disability Status', 'Sports', 'Location', 'Gender'],
            'Importance': [21.0, 23.5, 15.6, 10.7, 10.8, 10.2, 8.0]
        }
        
        fig = px.bar(
            x=importance_data['Importance'],
            y=importance_data['Factor'],
            orientation='h',
            title="Feature Importance (%)",
            color=importance_data['Importance'],
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("ℹ️ About")
        st.markdown("""
        **Developed by:** VTU Mini Project Team  
        **Model Type:** Random Forest Classifier  
        **Dataset:** National Overseas Scholarship Scheme  
        **Last Updated:** December 2024
        
        This tool provides predictions based on historical data and should be used as guidance only. 
        Always verify eligibility criteria with official scholarship providers.
        """)

if __name__ == "__main__":
    main()