"""
Causeway Prototype - Streamlit Testing App

A comprehensive multi-page application to test all Causeway features.
Run with: streamlit run prototype/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Causeway Prototype",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #10b98115 0%, #059b7915 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🔮 Causeway</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Agentic Decision Support System with Causal Intelligence</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview
    st.markdown("""
    <div class="info-box">
    <h3>📋 Welcome to Causeway Prototype</h3>
    <p>This application allows you to test all Causeway features in a guided, chronological workflow.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview with columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        Follow these pages **in order** to experience the full Causeway workflow:
        
        1. **📊 System Status** - Check if all services are running
        2. **📁 Document Upload** - Upload evidence documents
        3. **🔍 Document Indexing** - Index documents for retrieval
        4. **🌐 World Model Builder** - Run Mode 1 to build causal models
        5. **🤔 Decision Support** - Run Mode 2 for recommendations
        6. **🗺️ Model Explorer** - View and explore world models
        7. **📈 Training Dashboard** - View training metrics
        """)
    
    with col2:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        ```
        ┌──────────────────────────────────────┐
        │           Causeway System            │
        ├──────────────────────────────────────┤
        │  Mode 1: World Model Construction    │
        │  Mode 2: Decision Support            │
        ├──────────────────────────────────────┤
        │  Causal Intelligence Layer           │
        │  (DAG Engine, Path Finder)           │
        ├──────────────────────────────────────┤
        │  Retrieval Infrastructure            │
        │  (PageIndex, Haystack, Router)       │
        ├──────────────────────────────────────┤
        │  Storage Layer                       │
        │  (PostgreSQL, Redis, MinIO, Qdrant)  │
        └──────────────────────────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    # Quick status check
    st.markdown("### ⚡ Quick Status Check")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔌 Check API"):
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API Running")
                else:
                    st.error("❌ API Error")
            except:
                st.error("❌ API Offline")
    
    with col2:
        if st.button("🐘 Check PostgreSQL"):
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    user="causeway",
                    password="causeway_dev",
                    database="causeway"
                )
                conn.close()
                st.success("✅ PostgreSQL OK")
            except:
                st.error("❌ PostgreSQL Offline")
    
    with col3:
        if st.button("🔴 Check Redis"):
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379)
                r.ping()
                st.success("✅ Redis OK")
            except:
                st.error("❌ Redis Offline")
    
    with col4:
        if st.button("📦 Check MinIO"):
            try:
                import requests
                response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
                if response.status_code == 200:
                    st.success("✅ MinIO OK")
                else:
                    st.error("❌ MinIO Error")
            except:
                st.error("❌ MinIO Offline")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #888; margin-top: 2rem;">
        <p>Causeway v0.1.0 | Built with ❤️ for Decision Intelligence</p>
        <p>Use the sidebar to navigate between pages →</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
