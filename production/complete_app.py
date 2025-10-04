import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Database Manager
class ScholarshipDB:
    def __init__(self, db_path='scholarships.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scholarships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            provider TEXT NOT NULL,
            amount INTEGER,
            max_income INTEGER,
            min_percentage REAL DEFAULT 50.0,
            eligible_communities TEXT,
            eligible_genders TEXT,
            eligible_religions TEXT,
            location_preference TEXT,
            disability_friendly BOOLEAN DEFAULT 0,
            sports_required BOOLEAN DEFAULT 0,
            application_link TEXT,
            deadline DATE,
            description TEXT,
            field_of_study TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_data TEXT,
            scholarship_id INTEGER,
            match_score REAL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scholarship_id) REFERENCES scholarships(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
        # Insert sample data if empty
        self.insert_sample_scholarships()
    
    def insert_sample_scholarships(self):
        """Insert sample scholarships if database is empty"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM scholarships")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        sample_data = [
            {
                "name": "National Overseas Scholarship for SC/ST/OBC/Minorities",
                "provider": "Ministry of Social Justice & Empowerment",
                "amount": 500000,
                "max_income": 800000,
                "min_percentage": 60.0,
                "eligible_communities": "SC/ST,OBC,Minority",
                "eligible_genders": "Female,Male",
                "eligible_religions": "Hindu,Muslim,Christian,Others",
                "location_preference": "Any",
                "disability_friendly": 1,
                "sports_required": 0,
                "application_link": "https://scholarships.gov.in/",
                "deadline": "2025-03-31",
                "description": "For pursuing higher studies abroad",
                "field_of_study": "All"
            },
            {
                "name": "Post Matric Scholarship for SC Students",
                "provider": "Government of India",
                "amount": 250000,
                "max_income": 300000,
                "min_percentage": 50.0,
                "eligible_communities": "SC/ST",
                "eligible_genders": "Female,Male",
                "eligible_religions": "Any",
                "location_preference": "In-State",
                "disability_friendly": 1,
                "sports_required": 0,
                "application_link": "https://scholarships.gov.in/",
                "deadline": "2025-02-28",
                "description": "Post-matriculation financial support",
                "field_of_study": "All"
            },
            {
                "name": "Girl Child Education Scholarship",
                "provider": "State Government",
                "amount": 100000,
                "max_income": 200000,
                "min_percentage": 45.0,
                "eligible_communities": "General,SC/ST,OBC,Minority",
                "eligible_genders": "Female",
                "eligible_religions": "Any",
                "location_preference": "In-State",
                "disability_friendly": 1,
                "sports_required": 0,
                "application_link": "https://scholarships.gov.in/",
                "deadline": "2025-06-30",
                "description": "Promoting girl child education",
                "field_of_study": "All"
            },
            {
                "name": "Sports Excellence Scholarship",
                "provider": "Ministry of Youth Affairs",
                "amount": 350000,
                "max_income": 1000000,
                "min_percentage": 40.0,
                "eligible_communities": "General,SC/ST,OBC,Minority",
                "eligible_genders": "Female,Male",
                "eligible_religions": "Any",
                "location_preference": "Any",
                "disability_friendly": 1,
                "sports_required": 1,
                "application_link": "https://yas.nic.in/",
                "deadline": "2025-04-30",
                "description": "For students with sports achievements",
                "field_of_study": "All"
            },
            {
                "name": "Disability Support Scholarship",
                "provider": "Ministry of Social Justice",
                "amount": 200000,
                "max_income": 600000,
                "min_percentage": 40.0,
                "eligible_communities": "General,SC/ST,OBC,Minority",
                "eligible_genders": "Female,Male",
                "eligible_religions": "Any",
                "location_preference": "Any",
                "disability_friendly": 1,
                "sports_required": 0,
                "application_link": "https://scholarships.gov.in/",
                "deadline": "2025-03-31",
                "description": "Support for students with disabilities",
                "field_of_study": "All"
            },
            {
                "name": "Merit-cum-Means Scholarship for Minorities",
                "provider": "Ministry of Minority Affairs",
                "amount": 200000,
                "max_income": 500000,
                "min_percentage": 55.0,
                "eligible_communities": "Minority",
                "eligible_genders": "Female,Male",
                "eligible_religions": "Muslim,Christian,Sikh,Buddhist,Jain",
                "location_preference": "Any",
                "disability_friendly": 0,
                "sports_required": 0,
                "application_link": "https://minorityaffairs.gov.in/",
                "deadline": "2025-01-31",
                "description": "For professional and technical courses",
                "field_of_study": "Engineering,Medical,Management"
            }
        ]
        
        for scholarship in sample_data:
            cursor.execute('''
            INSERT INTO scholarships (
                name, provider, amount, max_income, min_percentage,
                eligible_communities, eligible_genders, eligible_religions,
                location_preference, disability_friendly, sports_required,
                application_link, deadline, description, field_of_study
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scholarship['name'], scholarship['provider'],
                scholarship['amount'], scholarship['max_income'],
                scholarship['min_percentage'], scholarship['eligible_communities'],
                scholarship['eligible_genders'], scholarship['eligible_religions'],
                scholarship['location_preference'], scholarship['disability_friendly'],
                scholarship['sports_required'], scholarship['application_link'],
                scholarship['deadline'], scholarship['description'],
                scholarship['field_of_study']
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Inserted {len(sample_data)} sample scholarships")
    
    def get_matching_scholarships(self, student_profile):
        """Get scholarships matching student profile"""
        conn = self.get_connection()
        
        query = """
        SELECT * FROM scholarships 
        WHERE is_active = 1
        AND max_income >= ?
        AND (eligible_communities LIKE ? OR eligible_communities LIKE '%Any%')
        AND (eligible_genders LIKE ? OR eligible_genders LIKE '%Any%')
        AND (eligible_religions LIKE ? OR eligible_religions LIKE '%Any%')
        """
        
        params = [
            student_profile['income'],
            f"%{student_profile['community']}%",
            f"%{student_profile['gender']}%",
            f"%{student_profile['religion']}%"
        ]
        
        # Add disability filter
        if student_profile.get('has_disability'):
            query += " AND disability_friendly = 1"
        
        # Add sports filter
        if student_profile.get('sports_participation'):
            query += " AND (sports_required = 0 OR sports_required = 1)"
        else:
            query += " AND sports_required = 0"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_all_scholarships(self):
        """Get all active scholarships"""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM scholarships WHERE is_active = 1", conn)
        conn.close()
        return df
    
    def add_scholarship(self, data):
        """Add new scholarship"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO scholarships (
            name, provider, amount, max_income, min_percentage,
            eligible_communities, eligible_genders, eligible_religions,
            location_preference, disability_friendly, sports_required,
            application_link, deadline, description, field_of_study
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(data.values()))
        
        conn.commit()
        scholarship_id = cursor.lastrowid
        conn.close()
        return scholarship_id
    
    def track_application(self, student_data, scholarship_id, match_score):
        """Track student application"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO applications (student_data, scholarship_id, match_score)
        VALUES (?, ?, ?)
        ''', (json.dumps(student_data), scholarship_id, match_score))
        
        conn.commit()
        conn.close()

# ML Model Handler
class MLModelHandler:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load ML model"""
        model_paths = [
            'scholarship_model_quick.pkl',
            'production/scholarship_model_quick.pkl',
            '../scholarship_model_quick.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    data = joblib.load(path)
                    self.model = data['model']
                    self.feature_names = data['feature_names']
                    return True
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
        
        return False
    
    def predict_probability(self, student_profile):
        """Predict scholarship success probability"""
        if not self.model:
            return 0.5  # Default probability if model not loaded
        
        try:
            ml_input = {
                'gender_encoded': 0 if student_profile['gender'] == 'Female' else 1,
                'community_encoded': {'General': 0, 'Minority': 1, 'OBC': 2, 'SC/ST': 3}.get(student_profile['community'], 0),
                'religion_encoded': {'Christian': 0, 'Hindu': 1, 'Muslim': 2, 'Others': 3}.get(student_profile['religion'], 3),
                'location_encoded': 0 if student_profile['location'] == 'In-State' else 1,
                'has_disability_binary': 1 if student_profile.get('has_disability') else 0,
                'sports_participation_binary': 1 if student_profile.get('sports_participation') else 0,
                'income_numerical': student_profile['income']
            }
            
            input_df = pd.DataFrame([ml_input])[self.feature_names]
            probability = self.model.predict_proba(input_df)[0, 1]
            return probability
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5

# Recommendation Engine
class ScholarshipRecommender:
    def __init__(self):
        self.db = ScholarshipDB()
        self.ml_model = MLModelHandler()
    
    def calculate_match_score(self, student_profile, scholarship):
        """Calculate match score between student and scholarship"""
        score = 0
        max_score = 100
        
        # Income match (30 points)
        if student_profile['income'] <= scholarship['max_income']:
            score += 30
        else:
            ratio = scholarship['max_income'] / student_profile['income']
            if ratio > 0.7:
                score += 15
        
        # Community match (20 points)
        communities = scholarship['eligible_communities'].split(',')
        if student_profile['community'] in communities or 'Any' in communities:
            score += 20
        
        # Gender match (15 points)
        genders = scholarship['eligible_genders'].split(',')
        if student_profile['gender'] in genders or 'Any' in genders:
            score += 15
        
        # Religion match (10 points)
        religions = scholarship['eligible_religions'].split(',')
        if student_profile['religion'] in religions or 'Any' in religions:
            score += 10
        
        # Location match (10 points)
        if scholarship['location_preference'] == 'Any' or student_profile['location'] == scholarship['location_preference']:
            score += 10
        
        # Disability match (10 points)
        if student_profile.get('has_disability') and scholarship['disability_friendly']:
            score += 10
        
        # Sports match (5 points)
        if student_profile.get('sports_participation'):
            if scholarship['sports_required']:
                score += 5
            elif not scholarship['sports_required']:
                score += 2
        
        # ML probability boost
        ml_probability = self.ml_model.predict_probability(student_profile)
        final_score = (score / max_score) * 100 * ml_probability
        
        return min(final_score, 100)
    
    def get_recommendations(self, student_profile):
        """Get personalized scholarship recommendations"""
        # Get matching scholarships from database
        matching_scholarships = self.db.get_matching_scholarships(student_profile)
        
        if matching_scholarships.empty:
            return []
        
        # Calculate match scores
        recommendations = []
        for _, scholarship in matching_scholarships.iterrows():
            match_score = self.calculate_match_score(student_profile, scholarship)
            
            if match_score >= 20:  # Minimum threshold
                recommendations.append({
                    'scholarship': scholarship.to_dict(),
                    'match_score': match_score,
                    'ml_probability': self.ml_model.predict_probability(student_profile)
                })
        
        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        return recommendations

# Streamlit App
def main():
    st.set_page_config(
        page_title="Scholarship Recommender",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Scholarship Finder & Recommender System")
    st.markdown("**AI-Powered Scholarship Matching with Real Database**")
    
    # Initialize recommender
    recommender = ScholarshipRecommender()
    
    # Check model status
    model_status = "✅ Loaded" if recommender.ml_model.model else "⚠️ Not Loaded (Using Fallback)"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_scholarships = len(recommender.db.get_all_scholarships())
        st.metric("📚 Scholarships Available", total_scholarships)
    with col2:
        st.metric("🤖 ML Model", model_status)
    with col3:
        st.metric("💾 Database", "Connected")
    
    # Admin Panel
    with st.sidebar.expander("⚙️ Admin: Manage Scholarships"):
        if st.button("➕ Add New Scholarship"):
            st.session_state['show_add_form'] = True
        
        if st.session_state.get('show_add_form'):
            with st.form("add_scholarship"):
                name = st.text_input("Scholarship Name*")
                provider = st.text_input("Provider*")
                amount = st.number_input("Amount (₹)*", min_value=0, value=100000)
                max_income = st.number_input("Max Income (₹)*", min_value=0, value=500000)
                deadline = st.date_input("Deadline*")
                
                submitted = st.form_submit_button("Add Scholarship")
                if submitted and name and provider:
                    scholarship_data = {
                        'name': name,
                        'provider': provider,
                        'amount': amount,
                        'max_income': max_income,
                        'min_percentage': 50.0,
                        'eligible_communities': 'General,SC/ST,OBC,Minority',
                        'eligible_genders': 'Female,Male',
                        'eligible_religions': 'Any',
                        'location_preference': 'Any',
                        'disability_friendly': 0,
                        'sports_required': 0,
                        'application_link': 'https://scholarships.gov.in/',
                        'deadline': deadline.strftime('%Y-%m-%d'),
                        'description': 'New scholarship',
                        'field_of_study': 'All'
                    }
                    
                    scholarship_id = recommender.db.add_scholarship(scholarship_data)
                    st.success(f"✅ Added scholarship (ID: {scholarship_id})")
                    st.session_state['show_add_form'] = False
                    st.rerun()
        
        # View all scholarships
        all_scholarships = recommender.db.get_all_scholarships()
        st.dataframe(all_scholarships[['name', 'amount', 'deadline']], height=200)
    
    # Student Profile Form
    st.sidebar.header("📝 Your Profile")
    
    gender = st.sidebar.selectbox("Gender*", ["Female", "Male"])
    community = st.sidebar.selectbox("Community*", ["General", "Minority", "OBC", "SC/ST"])
    religion = st.sidebar.selectbox("Religion", ["Hindu", "Muslim", "Christian", "Others"])
    location = st.sidebar.selectbox("Location", ["In-State", "Out-of-State"])
    
    has_disability = st.sidebar.checkbox("I have a disability")
    sports_participation = st.sidebar.checkbox("I participate in sports")
    
    income = st.sidebar.number_input(
        "Annual Family Income (₹)*",
        min_value=0,
        max_value=5000000,
        value=200000,
        step=10000
    )
    
    if st.sidebar.button("🔍 Find My Scholarships", type="primary"):
        student_profile = {
            'gender': gender,
            'community': community,
            'religion': religion,
            'location': location,
            'has_disability': has_disability,
            'sports_participation': sports_participation,
            'income': income
        }
        
        with st.spinner("Searching for matching scholarships..."):
            recommendations = recommender.get_recommendations(student_profile)
        
        st.session_state['recommendations'] = recommendations
        st.session_state['student_profile'] = student_profile
    
    # Display Results
    if 'recommendations' in st.session_state:
        recommendations = st.session_state['recommendations']
        
        if recommendations:
            st.success(f"🎉 Found {len(recommendations)} matching scholarships!")
            
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                excellent = len([r for r in recommendations if r['match_score'] >= 70])
                st.metric("🌟 Excellent Matches", excellent)
            with col2:
                total_value = sum([r['scholarship']['amount'] for r in recommendations[:3]])
                st.metric("💰 Top 3 Value", f"₹{total_value:,}")
            with col3:
                avg_score = np.mean([r['match_score'] for r in recommendations])
                st.metric("📊 Avg Match", f"{avg_score:.1f}%")
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                s = rec['scholarship']
                score = rec['match_score']
                
                with st.expander(f"#{i} {s['name']} - {score:.1f}% Match", expanded=(i <= 3)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**{s['provider']}**")
                        st.write(s['description'])
                        st.write(f"💰 **Amount:** ₹{s['amount']:,}")
                        st.write(f"📅 **Deadline:** {s['deadline']}")
                        st.write(f"📊 **Max Income:** ₹{s['max_income']:,}")
                        st.link_button("🔗 Apply Now", s['application_link'])
                    
                    with col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': "Match Score"},
                            gauge={'axis': {'range': [None, 100]}}
                        ))
                        fig.update_layout(height=200)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("😔 No matching scholarships found. Try adjusting your profile or check back later for new scholarships.")
    
    else:
        st.info("👈 Fill in your profile and click 'Find My Scholarships' to get personalized recommendations!")

if __name__ == "__main__":
    main()