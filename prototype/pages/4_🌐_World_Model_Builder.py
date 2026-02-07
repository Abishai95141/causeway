"""
Page 4: World Model Builder (Mode 1)

Build causal world models from evidence.
"""

import streamlit as st
import requests
import json

st.set_page_config(page_title="World Model Builder", page_icon="🌐", layout="wide")

st.markdown("# 🌐 World Model Builder (Mode 1)")
st.markdown("Construct causal world models from your indexed evidence documents.")

st.markdown("---")

st.markdown("""
> **Instructions:**
> 1. Enter a domain name (e.g., "pricing", "marketing", "operations")
> 2. Provide an initial query describing what you want to model
> 3. Click "Build World Model" to start Mode 1
> 4. Review and approve the resulting model
""")

st.markdown("---")

# Mode 1 Configuration
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚙️ Configuration")
    
    domain = st.text_input(
        "Domain Name",
        placeholder="pricing",
        help="Name for this causal domain (e.g., pricing, marketing)"
    )
    
    initial_query = st.text_area(
        "Initial Query",
        placeholder="What are the key factors that influence product pricing and customer demand?",
        help="Describe what causal relationships you want to discover",
        height=100
    )
    
    with st.expander("Advanced Options"):
        max_variables = st.slider("Max Variables", 5, 50, 20)
        max_edges = st.slider("Max Edges", 10, 100, 50)

with col2:
    st.markdown("### 📖 Mode 1 Workflow")
    st.markdown("""
    Mode 1 follows these stages:
    
    1. **Variable Discovery** 🔍
       - LLM identifies potential causal variables from evidence
    
    2. **Evidence Gathering** 📚
       - Deep retrieval for each variable
    
    3. **DAG Drafting** 🗺️
       - Build causal graph with edges
    
    4. **Evidence Triangulation** ⚖️
       - Assign edge strengths based on evidence
    
    5. **Human Review** 👤
       - Present model for approval
    """)

st.markdown("---")

# Run Mode 1
st.markdown("### 🚀 Build World Model")

if st.button("🔨 Build World Model", type="primary", use_container_width=True):
    if domain and initial_query:
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.info("📡 Sending request to Mode 1...")
            progress.progress(10)
            
            response = requests.post(
                "http://localhost:8000/api/v1/mode1/run",
                json={
                    "domain": domain,
                    "initial_query": initial_query,
                    "max_variables": max_variables,
                    "max_edges": max_edges
                },
                timeout=120
            )
            
            progress.progress(80)
            
            if response.status_code == 200:
                result = response.json()
                progress.progress(100)
                
                status.success("✅ World model construction complete!")
                
                # Save to session
                if 'world_models' not in st.session_state:
                    st.session_state.world_models = []
                st.session_state.world_models.append(result)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Trace ID", result['trace_id'][:12] + "...")
                col2.metric("Variables", result['variables_discovered'])
                col3.metric("Edges", result['edges_created'])
                col4.metric("Evidence", result['evidence_linked'])
                
                st.markdown("---")
                
                # Model details
                with st.expander("📋 Full Response"):
                    st.json(result)
                
                if result.get('requires_review'):
                    st.warning("⚠️ This model requires human review before activation.")
                    
                    if st.button("✅ Approve Model"):
                        try:
                            approve_response = requests.post(
                                "http://localhost:8000/api/v1/mode1/approve",
                                json={
                                    "domain": domain,
                                    "approved_by": "prototype_user"
                                },
                                timeout=30
                            )
                            
                            if approve_response.status_code == 200:
                                st.success("✅ Model approved and activated!")
                                st.json(approve_response.json())
                            else:
                                st.error(f"❌ Approval failed: {approve_response.text}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                if result.get('error'):
                    st.error(f"⚠️ Error: {result['error']}")
                    
            else:
                progress.progress(100)
                status.error(f"❌ Error: {response.text}")
                
        except requests.exceptions.Timeout:
            progress.progress(100)
            status.error("❌ Request timed out. Mode 1 may still be running in the background.")
        except Exception as e:
            progress.progress(100)
            status.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter both domain and initial query")

st.markdown("---")

# Direct Mode 1 Test (Without API)
st.markdown("### 🧪 Direct Mode 1 Test")
st.markdown("Test Mode 1 directly without the API (for debugging)")

if st.button("🔬 Run Direct Test"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        
        import asyncio
        from src.modes.mode1 import Mode1WorldModelConstruction
        
        async def run_mode1():
            mode1 = Mode1WorldModelConstruction()
            await mode1.initialize()
            
            result = await mode1.run(
                domain=domain or "test_domain",
                initial_query=initial_query or "What are the key factors?",
                max_variables=10,
                max_edges=20
            )
            return result
        
        with st.spinner("Running Mode 1 directly..."):
            result = asyncio.run(run_mode1())
        
        st.success("✅ Mode 1 completed!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Variables", result.variables_discovered)
        col2.metric("Edges", result.edges_created)
        col3.metric("Evidence", result.evidence_linked)
        
        if result.error:
            st.warning(f"⚠️ Note: {result.error}")
        
        st.markdown("**Audit Trail:**")
        for entry in result.audit_entries:
            st.markdown(f"- {entry.action}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✅ After building a world model, proceed to <b>Decision Support</b> →</p>
</div>
""", unsafe_allow_html=True)
