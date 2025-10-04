import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced Scholarship Database
SCHOLARSHIP_DATABASE = [
    {
        "name": "National Overseas Scholarship for SC/ST/OBC/Minorities",
        "provider": "Ministry of Social Justice & Empowerment",
        "amount": 500000,
        "max_income": 800000,
        "eligible_communities": ["SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "In-State",
        "disability_friendly": True,
        "sports_bonus": False,
        "application_link": "https://scholarships.gov.in/",
        "deadline": "2024-03-31",
        "description": "For pursuing higher studies abroad in various fields",
        "eligibility_score_multiplier": 1.5
    },
    {
        "name": "Post Matric Scholarship for SC Students",
        "provider": "Government of India",
        "amount": 250000,
        "max_income": 300000,
        "eligible_communities": ["SC/ST"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": True,
        "sports_bonus": False,
        "application_link": "https://scholarships.gov.in/",
        "deadline": "2024-02-28",
        "description": "Financial assistance for post-matriculation studies",
        "eligibility_score_multiplier": 1.3
    },
    {
        "name": "Merit-cum-Means Scholarship for Minorities",
        "provider": "Ministry of Minority Affairs",
        "amount": 200000,
        "max_income": 500000,
        "eligible_communities": ["Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": False,
        "sports_bonus": False,
        "application_link": "https://minorityaffairs.gov.in/",
        "deadline": "2024-01-31",
        "description": "For professional and technical courses",
        "eligibility_score_multiplier": 1.2
    },
    {
        "name": "Prime Minister's Special Scholarship for J&K Students",
        "provider": "AICTE",
        "amount": 300000,
        "max_income": 800000,
        "eligible_communities": ["General", "SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Out-of-State",
        "disability_friendly": True,
        "sports_bonus": True,
        "application_link": "https://www.aicte-india.org/",
        "deadline": "2024-04-15",
        "description": "For students from Jammu & Kashmir",
        "eligibility_score_multiplier": 1.4
    },
    {
        "name": "Inspire Scholarship for Higher Education",
        "provider": "Department of Science & Technology",
        "amount": 400000,
        "max_income": 600000,
        "eligible_communities": ["General", "SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": False,
        "sports_bonus": True,
        "application_link": "https://online-inspire.gov.in/",
        "deadline": "2024-05-30",
        "description": "For pursuing science courses",
        "eligibility_score_multiplier": 1.1
    },
    {
        "name": "Sitaram Jindal Foundation Scholarship",
        "provider": "Sitaram Jindal Foundation",
        "amount": 150000,
        "max_income": 400000,
        "eligible_communities": ["General", "SC/ST", "OBC"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": True,
        "sports_bonus": True,
        "application_link": "https://www.sitaramjindalfoundation.org/",
        "deadline": "2024-03-15",
        "description": "Need-based scholarship for meritorious students",
        "eligibility_score_multiplier": 1.0
    },
    {
        "name": "Kishore Vaigyanik Protsahan Yojana (KVPY)",
        "provider": "Indian Institute of Science",
        "amount": 700000,
        "max_income": 450000,
        "eligible_communities": ["General", "SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": False,
        "sports_bonus": False,
        "application_link": "http://www.kvpy.iisc.ernet.in/",
        "deadline": "2024-02-15",
        "description": "For students pursuing research in science",
        "eligibility_score_multiplier": 0.9
    },
    {
        "name": "Girl Child Education Scholarship",
        "provider": "Various State Governments",
        "amount": 100000,
        "max_income": 200000,
        "eligible_communities": ["General", "SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female"],
        "location_preference": "In-State",
        "disability_friendly": True,
        "sports_bonus": False,
        "application_link": "https://scholarships.gov.in/",
        "deadline": "2024-06-30",
        "description": "Promoting girl child education",
        "eligibility_score_multiplier": 1.6
    },
    {
        "name": "Sports Excellence Scholarship",
        "provider": "Ministry of Youth Affairs and Sports",
        "amount": 350000,
        "max_income": 1000000,
        "eligible_communities": ["General", "SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": True,
        "sports_bonus": True,
        "application_link": "https://yas.nic.in/",
        "deadline": "2024-04-30",
        "description": "For students excelling in sports",
        "eligibility_score_multiplier": 2.0
    },
    {
        "name": "Disability Support Scholarship",
        "provider": "Ministry of Social Justice & Empowerment",
        "amount": 200000,
        "max_income": 600000,
        "eligible_communities": ["General", "SC/ST", "OBC", "Minority"],
        "eligible_genders": ["Female", "Male"],
        "location_preference": "Any",
        "disability_friendly": True,
        "sports_bonus": False,
        "application_link": "https://scholarships.gov.in/",
        "deadline": "2024-03-31",
        "description": "Supporting students with disabilities",
        "eligibility_score_multiplier": 2.5
    }
]

class EnhancedScholarshipRecommender:
    def __init__(self):
        self.model_data = None
        self.scholarship_df = pd.DataFrame(SCHOLARSHIP_DATABASE)
        self.load_model()
    
    def load_model(self):
        """Load the ML model"""
        try:
            self.model_data = joblib.load('scholarship_model_quick.pkl')
            return True
        except FileNotFoundError:
            return False
    
    def calculate_scholarship_match_score(self, student_profile, scholarship):
        """Calculate how well a scholarship matches a student profile"""
        score = 0
        max_score = 100
        
        # Income eligibility (30 points)
        if student_profile['income'] <= scholarship['max_income']:
            score += 30
        else:
            # Partial points if close to limit
            income_ratio = scholarship['max_income'] / student_profile['income']
            if income_ratio > 0.8:
                score += 20
            elif income_ratio > 0.6:
                score += 10
        
        # Community match (25 points)
        if student_profile['community'] in scholarship['eligible_communities']:
            score += 25
        
        # Gender match (10 points)
        if student_profile['gender'] in scholarship['eligible_genders']:
            score += 10
        
        # Location preference (15 points)
        if scholarship['location_preference'] == 'Any' or student_profile['location'] == scholarship['location_preference']:
            score += 15
        
        # Disability bonus (10 points)
        if student_profile['has_disability'] and scholarship['disability_friendly']:
            score += 10
        
        # Sports bonus (10 points)
        if student_profile['sports_participation'] and scholarship['sports_bonus']:
            score += 10
        
        # Apply ML model probability as multiplier
        if self.model_data:
            ml_input = {
                'gender_encoded': 0 if student_profile['gender'] == 'Female' else 1,
                'community_encoded': {'General': 0, 'Minority': 1, 'OBC': 2, 'SC/ST': 3}[student_profile['community']],
                'religion_encoded': {'Christian': 0, 'Hindu': 1, 'Muslim': 2, 'Others': 3}[student_profile['religion']],
                'location_encoded': 0 if student_profile['location'] == 'In-State' else 1,
                'has_disability_binary': 1 if student_profile['has_disability'] else 0,
                'sports_participation_binary': 1 if student_profile['sports_participation'] else 0,
                'income_numerical': student_profile['income']
            }
            
            input_df = pd.DataFrame([ml_input])
            input_df = input_df[self.model_data['feature_names']]
            
            ml_probability = self.model_data['model'].predict_proba(input_df)[0, 1]
            
            # Apply ML probability and scholarship-specific multiplier
            final_score = (score / max_score) * ml_probability * scholarship['eligibility_score_multiplier'] * 100
        else:
            final_score = (score / max_score) * scholarship['eligibility_score_multiplier'] * 50
        
        return min(final_score, 100)  # Cap at 100%
    
    def recommend_scholarships(self, student_profile, top_k=10):
        """Recommend top scholarships for a student"""
        recommendations = []
        
        for _, scholarship in self.scholarship_df.iterrows():
            match_score = self.calculate_scholarship_match_score(student_profile, scholarship.to_dict())
            
            if match_score > 10:  # Only show scholarships with >10% match
                recommendations.append({
                    'scholarship': scholarship.to_dict(),
                    'match_score': match_score,
                    'recommendation_level': self.get_recommendation_level(match_score)
                })
        
        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        return recommendations[:top_k]
    
    def get_recommendation_level(self, score):
        """Get recommendation level based on score"""
        if score >= 80:
            return "🌟 Excellent Match"
        elif score >= 65:
            return "✅ Very Good Match"
        elif score >= 50:
            return "👍 Good Match"
        elif score >= 35:
            return "🤔 Moderate Match"
        else:
            return "📝 Consider with Caution"

def main():
    st.set_page_config(
        page_title="Scholarship Recommender",
        page_icon="🎓",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .scholarship-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .match-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FFD700;
    }
    .scholarship-amount {
        font-size: 1.2rem;
        color: #90EE90;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Enhanced Scholarship Finder & Recommender")
    st.subheader("AI-Powered Scholarship Matching with Real Scholarship Database")
    
    # Initialize recommender
    recommender = EnhancedScholarshipRecommender()
    
    if not recommender.model_data:
        st.error("❌ ML Model not found. Please train the model first.")
        if st.button("🚀 Train Model Now"):
            st.info("Please run the training script first and then refresh this page.")
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📝 Your Profile")
    
    # Input fields
    with st.sidebar:
        # Gender
        gender = st.selectbox("Gender", ["Female", "Male"])
        
        # Community
        community = st.selectbox("Community", ["General", "Minority", "OBC", "SC/ST"])
        
        # Religion
        religion = st.selectbox("Religion", ["Christian", "Hindu", "Muslim", "Others"])
        
        # Location
        location = st.selectbox("Location Status", ["In-State", "Out-of-State"])
        
        # Disability
        has_disability = st.selectbox("Disability Status", [False, True], format_func=lambda x: "Yes" if x else "No")
        
        # Sports
        sports_participation = st.selectbox("Sports Participation", [False, True], format_func=lambda x: "Yes" if x else "No")
        
        # Income
        income = st.number_input(
            "Annual Family Income (₹)",
            min_value=0,
            max_value=2000000,
            value=200000,
            step=10000
        )
        
        # Find scholarships button
        if st.button("🔍 Find My Scholarships", type="primary"):
            # Prepare student profile
            student_profile = {
                'gender': gender,
                'community': community,
                'religion': religion,
                'location': location,
                'has_disability': has_disability,
                'sports_participation': sports_participation,
                'income': income
            }
            
            # Get recommendations
            with st.spinner("Finding perfect scholarships for you..."):
                recommendations = recommender.recommend_scholarships(student_profile, top_k=8)
            
            # Store in session state
            st.session_state['recommendations'] = recommendations
            st.session_state['student_profile'] = student_profile
            st.session_state['show_results'] = True
    
    # Main content
    if st.session_state.get('show_results', False):
        recommendations = st.session_state['recommendations']
        student_profile = st.session_state['student_profile']
        
        if recommendations:
            st.success(f"🎉 Found {len(recommendations)} matching scholarships for you!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                excellent_matches = len([r for r in recommendations if r['match_score'] >= 80])
                st.metric("🌟 Excellent Matches", excellent_matches)
            
            with col2:
                total_amount = sum([r['scholarship']['amount'] for r in recommendations[:3]])
                st.metric("💰 Top 3 Total Value", f"₹{total_amount:,}")
            
            with col3:
                avg_score = np.mean([r['match_score'] for r in recommendations])
                st.metric("📊 Average Match", f"{avg_score:.1f}%")
            
            with col4:
                urgent_deadlines = len([r for r in recommendations 
                                     if datetime.strptime(r['scholarship']['deadline'], '%Y-%m-%d') < datetime.now() + timedelta(days=30)])
                st.metric("⏰ Urgent (30 days)", urgent_deadlines)
            
            # Display recommendations
            st.subheader("🎯 Your Personalized Scholarship Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                scholarship = rec['scholarship']
                score = rec['match_score']
                level = rec['recommendation_level']
                
                # Create expandable card for each scholarship
                with st.expander(f"#{i} {scholarship['name']} - {level} ({score:.1f}%)", expanded=(i <= 3)):
                    
                    # Scholarship details in columns
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **Provider:** {scholarship['provider']}
                        
                        **Description:** {scholarship['description']}
                        
                        **Eligibility:**
                        - Max Income: ₹{scholarship['max_income']:,}
                        - Communities: {', '.join(scholarship['eligible_communities'])}
                        - Location: {scholarship['location_preference']}
                        - Disability Friendly: {'✅' if scholarship['disability_friendly'] else '❌'}
                        - Sports Bonus: {'✅' if scholarship['sports_bonus'] else '❌'}
                        
                        **Application Deadline:** {scholarship['deadline']}
                        """)
                        
                        # Application button
                        st.link_button(
                            "🔗 Apply Now",
                            scholarship['application_link'],
                            help=f"Opens {scholarship['provider']} application portal"
                        )
                    
                    with col2:
                        # Match score gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': f"Match Score"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkgreen" if score >= 80 else "green" if score >= 65 else "orange" if score >= 50 else "red"},
                                'steps': [
                                    {'range': [0, 35], 'color': "lightgray"},
                                    {'range': [35, 50], 'color': "yellow"},
                                    {'range': [50, 65], 'color': "orange"},
                                    {'range': [65, 80], 'color': "lightgreen"},
                                    {'range': [80, 100], 'color': "green"}
                                ]
                            }
                        ))
                        fig.update_layout(height=200, font=dict(size=10))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Amount and why it matches
                        st.markdown(f"""
                        <div class="scholarship-amount">
                        💰 Amount: ₹{scholarship['amount']:,}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Application tips
            st.subheader("💡 Application Tips Based on Your Profile")
            
            if student_profile['income'] < 300000:
                st.info("💡 **Low Income Advantage:** You qualify for most need-based scholarships. Focus on preparing strong financial documents.")
            
            if student_profile['community'] in ['SC/ST', 'OBC', 'Minority']:
                st.info("💡 **Category Advantage:** Many scholarships are specifically for your community. Apply to category-specific scholarships first.")
            
            if student_profile['has_disability']:
                st.success("💡 **Disability Support:** You have access to special scholarships and additional support. Don't miss disability-specific opportunities!")
            
            if student_profile['sports_participation']:
                st.success("💡 **Sports Excellence:** Your sports participation opens additional scholarship opportunities. Highlight your achievements!")
            
            if student_profile['gender'] == 'Female':
                st.success("💡 **Girl Child Education:** Many scholarships prioritize female students. Look for women empowerment scholarships!")
        
        else:
            st.warning("😔 No matching scholarships found for your profile. Try adjusting your criteria or improving your profile.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🌟 Welcome to Enhanced Scholarship Recommender!
        
        ### What's New:
        ✅ **Real Scholarship Database** - 10+ actual scholarships with live links  
        ✅ **Smart Matching Algorithm** - ML + Rule-based recommendations  
        ✅ **Personalized Scoring** - Each scholarship gets a match score  
        ✅ **Application Guidance** - Tips based on your specific profile  
        ✅ **Deadline Tracking** - Never miss important dates  
        
        ### How It Works:
        1. **Fill your profile** in the sidebar (takes 2 minutes)
        2. **Get personalized recommendations** ranked by compatibility
        3. **Apply directly** through provided links
        4. **Follow application tips** for better success rates
        
        ### Scholarship Categories Available:
        - 🎓 **Government Scholarships** (NSP, State schemes)
        - 👥 **Community-Specific** (SC/ST, OBC, Minorities)
        - 🏃‍♀️ **Sports Excellence** scholarships
        - ♿ **Disability Support** scholarships
        - 👩‍🎓 **Girl Child Education** scholarships
        - 🔬 **Merit-based** academic scholarships
        
        **Ready to find your perfect scholarship match? Start by filling your profile! 👈**
        """)
        
        # Sample scholarship preview
        st.subheader("📋 Sample Scholarships in Our Database")
        
        sample_scholarships = SCHOLARSHIP_DATABASE[:3]
        for scholarship in sample_scholarships:
            st.markdown(f"""
            **{scholarship['name']}**  
            💰 Amount: ₹{scholarship['amount']:,} | 🏢 {scholarship['provider']}  
            📝 {scholarship['description']}
            """)

if __name__ == "__main__":
    main()