"""
Page 6: Model Explorer - View and explore causal world models.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Model Explorer", page_icon="🗺️", layout="wide")

st.markdown("# 🗺️ Model Explorer")
st.markdown("View and explore causal world models.")
st.markdown("---")

# List Models
st.markdown("### 📋 Available World Models")

if st.button("🔄 Refresh Models"):
    try:
        response = requests.get("http://localhost:8000/api/v1/world-models", timeout=10)
        if response.status_code == 200:
            st.session_state['models'] = response.json()
            st.success(f"✅ Found {len(st.session_state['models'])} models")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Cannot fetch: {e}")

if 'models' in st.session_state and st.session_state['models']:
    for model in st.session_state['models']:
        with st.expander(f"🌐 {model['domain']} - {model['node_count']} nodes"):
            st.json(model)
else:
    st.info("No models found. Build one first!")

st.markdown("---")

# Direct Access
st.markdown("### 🔬 Direct Causal Service")

if st.button("📊 Load Service"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        from src.causal.service import CausalService
        
        causal = CausalService()
        for domain in causal.list_domains():
            summary = causal.get_model_summary(domain)
            st.markdown(f"**{domain}:** {summary['node_count']} vars, {summary['edge_count']} edges")
            st.markdown("Variables: " + ", ".join(f"`{v}`" for v in summary['variables'][:10]))
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# Create Test Model
st.markdown("### 🧪 Create Sample Model")

if st.button("🔨 Create Sample"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        from src.causal.service import CausalService
        
        causal = CausalService()
        causal.create_world_model("sample_pricing")
        for var in [("price", "Price", "Unit price"), ("demand", "Demand", "Quantity"), ("revenue", "Revenue", "Total")]:
            causal.add_variable(*var)
        causal.add_causal_link("price", "demand", "Price affects demand")
        causal.add_causal_link("demand", "revenue", "Demand drives revenue")
        
        st.success("✅ Sample model created!")
    except Exception as e:
        st.error(f"Error: {e}")
