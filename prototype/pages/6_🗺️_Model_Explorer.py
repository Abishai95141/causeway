"""
Page 6: Model Explorer - View and explore causal world models + Cross-Model Bridges.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Model Explorer", page_icon="🗺️", layout="wide")

API = "http://localhost:8000/api/v1"

st.markdown("# 🗺️ Model Explorer")
st.markdown("View and explore causal world models.")
st.markdown("---")

# List Models
st.markdown("### 📋 Available World Models")

if st.button("🔄 Refresh Models"):
    try:
        response = requests.get(f"{API}/world-models", timeout=10)
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


# =====================================================================
# Cross-Model Bridge Builder
# =====================================================================
st.markdown("---")
st.header("🌉 Cross-Model Bridge Builder")
st.markdown(
    "Link two domain models via shared concepts and cross-domain causal edges.  "
    "Uses domain-prefixed variables (`finance::demand` vs `labour::demand`) "
    "to guarantee namespace isolation."
)

col_src, col_tgt = st.columns(2)
with col_src:
    bridge_source = st.text_input("Source domain", placeholder="e.g. pricing", key="bridge_src")
with col_tgt:
    bridge_target = st.text_input("Target domain", placeholder="e.g. supply_chain", key="bridge_tgt")

use_llm = st.checkbox("Use LLM for concept mapping (recommended)", value=True, key="bridge_llm")

if st.button("🔗 Build Bridge", type="primary", key="build_bridge"):
    if not bridge_source or not bridge_target:
        st.warning("Please enter both source and target domains.")
    elif bridge_source == bridge_target:
        st.warning("Source and target must be different domains.")
    else:
        with st.spinner(f"Building bridge between **{bridge_source}** ↔ **{bridge_target}**…"):
            try:
                resp = requests.post(
                    f"{API}/world-models/bridge",
                    json={
                        "source_domain": bridge_source,
                        "target_domain": bridge_target,
                        "use_llm": use_llm,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ Bridge **{data['bridge_id'][:8]}…** created!  "
                        f"{len(data.get('bridge_edges', []))} edges, "
                        f"{len(data.get('shared_concepts', []))} shared concepts"
                    )

                    # Show shared concepts
                    concepts = data.get("shared_concepts", [])
                    if concepts:
                        st.markdown("#### Shared Concepts")
                        for c in concepts:
                            st.write(
                                f"- **{c['source_var']}** ↔ **{c['target_var']}** "
                                f"(similarity: {c['similarity_score']:.2f}) — {c.get('mapping_rationale','')}"
                            )

                    # Show bridge edges
                    edges = data.get("bridge_edges", [])
                    if edges:
                        st.markdown("#### Bridge Edges")
                        for e in edges:
                            st.write(
                                f"- **{e['source_domain']}::{e['source_var']}** → "
                                f"**{e['target_domain']}::{e['target_var']}**  "
                                f"({e['strength']}, conf={e['confidence']:.2f})"
                            )
                            if e.get("mechanism"):
                                st.caption(f"   Mechanism: {e['mechanism']}")

                    st.json(data)
                else:
                    st.error(f"❌ Bridge failed ({resp.status_code}): {resp.text[:400]}")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

# -- List existing bridges --
st.markdown("---")
st.markdown("### 📜 Existing Bridges")

if st.button("🔄 Refresh Bridges", key="refresh_bridges"):
    try:
        resp = requests.get(f"{API}/world-models/bridges", timeout=10)
        if resp.status_code == 200:
            st.session_state["bridges"] = resp.json()
            st.success(f"Found {len(st.session_state['bridges'])} bridges")
        else:
            st.error(f"Error: {resp.text}")
    except Exception as exc:
        st.error(f"Cannot fetch: {exc}")

for b in st.session_state.get("bridges", []):
    with st.expander(
        f"🌉 {b.get('source_version_id', '?')} ↔ {b.get('target_version_id', '?')} "
        f"— {b.get('edge_count', 0)} edges, {b.get('concept_count', 0)} concepts  "
        f"[{b.get('status', '?')}]"
    ):
        # Fetch full detail
        try:
            detail = requests.get(f"{API}/world-models/bridges/{b['bridge_id']}", timeout=10)
            if detail.status_code == 200:
                st.json(detail.json())
            else:
                st.json(b)
        except Exception:
            st.json(b)
