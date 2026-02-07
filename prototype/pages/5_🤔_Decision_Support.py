"""
Page 5: Decision Support (Mode 2)

Get causal reasoning-backed recommendations.
"""

import streamlit as st
import requests
import json

st.set_page_config(page_title="Decision Support", page_icon="🤔", layout="wide")

st.markdown("# 🤔 Decision Support (Mode 2)")
st.markdown("Get causal reasoning-backed recommendations for your decision questions.")

st.markdown("---")

st.markdown("""
> **Instructions:**
> 1. Enter your decision question in natural language
> 2. Optionally specify a domain hint (if you know which world model to use)
> 3. Click "Get Recommendation" to run Mode 2
> 4. Review the causal reasoning and recommended actions
""")

st.markdown("---")

# Mode 2 Configuration
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ❓ Your Decision Question")
    
    query = st.text_area(
        "Decision Question",
        placeholder="Should we increase our product prices by 10% next quarter? We want to maximize revenue while maintaining customer retention.",
        help="Describe your decision question with context",
        height=120
    )
    
    domain_hint = st.text_input(
        "Domain Hint (optional)",
        placeholder="pricing",
        help="If you know which world model domain applies, enter it here"
    )

with col2:
    st.markdown("### 📖 Mode 2 Workflow")
    st.markdown("""
    Mode 2 follows these stages:
    
    1. **Query Parsing** 📝
       - Extract objective, levers, constraints
    
    2. **Model Retrieval** 🗺️
       - Find relevant world model
    
    3. **Evidence Refresh** 📚
       - Get latest evidence
    
    4. **Causal Reasoning** 🧠
       - Trace paths, find confounders
    
    5. **Recommendation** 💡
       - Synthesize actionable advice
    """)

st.markdown("---")

# Run Mode 2
st.markdown("### 🚀 Get Recommendation")

if st.button("💡 Get Recommendation", type="primary", use_container_width=True):
    if query:
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.info("📡 Sending request to Mode 2...")
            progress.progress(10)
            
            response = requests.post(
                "http://localhost:8000/api/v1/mode2/run",
                json={
                    "query": query,
                    "domain_hint": domain_hint if domain_hint else None
                },
                timeout=120
            )
            
            progress.progress(80)
            
            if response.status_code == 200:
                result = response.json()
                progress.progress(100)
                
                # Check for escalation
                if result.get('escalate_to_mode1'):
                    status.warning("⚠️ Escalation to Mode 1 recommended")
                    
                    st.warning(f"""
                    **Mode 2 recommends running Mode 1 first:**
                    
                    {result.get('escalation_reason', 'No world model found for this domain.')}
                    
                    → Go to **World Model Builder** to create a model first.
                    """)
                else:
                    status.success("✅ Recommendation generated!")
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Trace ID", result['trace_id'][:12] + "...")
                col2.metric("Evidence Used", result.get('evidence_count', 0))
                col3.metric("Model", result.get('model_used', 'N/A')[:12] if result.get('model_used') else 'N/A')
                confidence = result.get('confidence', 'N/A')
                col4.metric("Confidence", confidence.upper() if confidence else 'N/A')
                
                st.markdown("---")
                
                # Main Recommendation
                if result.get('recommendation'):
                    st.markdown("### 💡 Recommendation")
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                                padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    <h4 style="margin: 0 0 0.5rem 0;">{result.get('recommendation', 'No recommendation')}</h4>
                    <p style="color: #666; margin: 0;">Confidence: <b>{confidence.upper() if confidence else 'N/A'}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Full response
                with st.expander("📋 Full Response"):
                    st.json(result)
                
                if result.get('error'):
                    st.error(f"⚠️ Note: {result['error']}")
                    
            else:
                progress.progress(100)
                status.error(f"❌ Error: {response.text}")
                
        except requests.exceptions.Timeout:
            progress.progress(100)
            status.error("❌ Request timed out. Mode 2 may still be running.")
        except Exception as e:
            progress.progress(100)
            status.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a decision question")

st.markdown("---")

# Direct Mode 2 Test
st.markdown("### 🧪 Direct Mode 2 Test")

if st.button("🔬 Run Direct Test"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        
        import asyncio
        from src.modes.mode2 import Mode2DecisionSupport
        from src.causal.service import CausalService
        
        async def run_mode2():
            # First create a test world model
            causal = CausalService()
            if domain_hint or "test":
                test_domain = domain_hint or "test"
                try:
                    causal.create_world_model(test_domain)
                    causal.add_variable("price", "Price", "Product price")
                    causal.add_variable("demand", "Demand", "Customer demand")
                    causal.add_variable("revenue", "Revenue", "Total revenue")
                    causal.add_causal_link("price", "demand", "Price affects demand")
                    causal.add_causal_link("demand", "revenue", "Demand drives revenue")
                except:
                    pass  # Model may already exist
            
            mode2 = Mode2DecisionSupport(causal_service=causal)
            await mode2.initialize()
            
            result = await mode2.run(
                query=query or "Should we increase prices?",
                domain_hint=domain_hint or "test"
            )
            return result
        
        with st.spinner("Running Mode 2 directly..."):
            result = asyncio.run(run_mode2())
        
        st.success("✅ Mode 2 completed!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Evidence", result.evidence_count)
        col2.metric("Insights", len(result.causal_insights))
        col3.metric("Escalate?", "Yes" if result.escalate_to_mode1 else "No")
        
        if result.recommendation:
            st.markdown("**Recommendation:**")
            st.info(result.recommendation.recommendation)
        
        if result.error:
            st.warning(f"⚠️ Note: {result.error}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")

# Example Questions
st.markdown("### 📝 Example Questions")

examples = [
    "Should we increase our product prices by 10% next quarter?",
    "What would happen if we reduced marketing spend by 20%?",
    "How can we improve customer retention while reducing costs?",
    "Is it safe to launch in a new market given current competition?",
]

st.markdown("Try one of these example questions:")
for example in examples:
    if st.button(f"📌 {example[:50]}...", key=f"ex_{hash(example)}"):
        st.session_state['example_query'] = example
        st.rerun()

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✅ After getting recommendations, explore models in <b>Model Explorer</b> →</p>
</div>
""", unsafe_allow_html=True)
