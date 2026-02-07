"""
Page 1: System Status

Check the status of all Causeway services and dependencies.
"""

import streamlit as st
import sys
sys.path.insert(0, '/home/abishai/Desktop/causeway')

st.set_page_config(page_title="System Status", page_icon="📊", layout="wide")

st.markdown("# 📊 System Status")
st.markdown("Check the health and connectivity of all Causeway services.")

st.markdown("---")

st.markdown("""
> **Instructions:** Click each button to verify the service is running.
> All services should show ✅ before proceeding to other pages.
""")

st.markdown("---")

# Services Grid
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔌 Core Services")
    
    # API Health
    st.markdown("#### FastAPI Server")
    if st.button("Check API Health", key="api"):
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            data = response.json()
            st.success(f"✅ API is healthy")
            st.json(data)
        except Exception as e:
            st.error(f"❌ API is offline: {e}")
            st.info("Start the API with: `uvicorn src.api.main:app --reload`")
    
    # API Metrics
    st.markdown("#### API Metrics")
    if st.button("Get Metrics", key="metrics"):
        try:
            import requests
            response = requests.get("http://localhost:8000/metrics", timeout=5)
            data = response.json()
            st.success("✅ Metrics retrieved")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Uptime", f"{data.get('uptime_seconds', 0):.1f}s")
            col_b.metric("Requests", data.get('request_count', 0))
            col_c.metric("Errors", data.get('error_count', 0))
        except Exception as e:
            st.error(f"❌ Cannot get metrics: {e}")

with col2:
    st.markdown("### 🗄️ Database Services")
    
    # PostgreSQL
    st.markdown("#### PostgreSQL")
    if st.button("Check PostgreSQL", key="pg"):
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                user="causeway",
                password="causeway_dev",
                database="causeway"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            conn.close()
            st.success("✅ PostgreSQL connected")
            st.code(version, language="text")
        except Exception as e:
            st.error(f"❌ PostgreSQL error: {e}")
            st.info("Start PostgreSQL with: `docker compose up -d postgres`")
    
    # Redis
    st.markdown("#### Redis")
    if st.button("Check Redis", key="redis"):
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            info = r.info()
            st.success("✅ Redis connected")
            col_a, col_b = st.columns(2)
            col_a.metric("Version", info.get('redis_version', 'N/A'))
            col_b.metric("Connected Clients", info.get('connected_clients', 0))
        except Exception as e:
            st.error(f"❌ Redis error: {e}")
            st.info("Start Redis with: `docker compose up -d redis`")

st.markdown("---")

# Storage Services
st.markdown("### 📦 Storage Services")

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### MinIO (S3-Compatible)")
    if st.button("Check MinIO", key="minio"):
        try:
            import requests
            response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
            if response.status_code == 200:
                st.success("✅ MinIO is healthy")
                st.info("📎 MinIO Console: http://localhost:9001")
                st.info("📎 Credentials: minioadmin / minioadmin")
            else:
                st.warning(f"⚠️ MinIO returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ MinIO error: {e}")
            st.info("Start MinIO with: `docker compose up -d minio`")

with col4:
    st.markdown("#### Qdrant (Vector Store)")
    if st.button("Check Qdrant", key="qdrant"):
        try:
            import requests
            # Use /readyz for newer Qdrant versions
            response = requests.get("http://localhost:6333/readyz", timeout=5)
            if response.status_code == 200:
                st.success("✅ Qdrant is healthy")
                
                # Get collections
                collections = requests.get("http://localhost:6333/collections", timeout=5)
                if collections.status_code == 200:
                    data = collections.json()
                    coll_list = data.get('result', {}).get('collections', [])
                    st.info(f"📎 Collections: {len(coll_list)}")
            else:
                st.warning(f"⚠️ Qdrant returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Qdrant error: {e}")
            st.info("Start Qdrant with: `docker compose up -d qdrant`")

st.markdown("---")

# Docker Status
st.markdown("### 🐳 Docker Containers")
if st.button("Show Docker Status", key="docker"):
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "compose", "ps"],
            cwd="/home/abishai/Desktop/causeway",
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            st.code(result.stdout, language="text")
        else:
            st.error(f"Error: {result.stderr}")
    except Exception as e:
        st.error(f"Cannot run docker compose: {e}")

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✅ Once all services are green, proceed to <b>Document Upload</b> →</p>
</div>
""", unsafe_allow_html=True)
