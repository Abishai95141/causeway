"""
Page 7: Training Dashboard - View training metrics and trajectories.
"""

import streamlit as st

st.set_page_config(page_title="Training Dashboard", page_icon="📈", layout="wide")

st.markdown("# 📈 Training Dashboard")
st.markdown("View Agent Lightning training metrics and trajectories.")
st.markdown("---")

# Span Collector Stats
st.markdown("### 🔍 Span Collection")

if st.button("📊 View Spans"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        from src.training.spans import SpanCollector
        
        collector = SpanCollector()
        trace_id = collector.start_trace("test_trace")
        span_id = collector.start_span("test_span")
        collector.end_span(span_id)
        
        exported = collector.export_trace(trace_id)
        st.success(f"✅ Collected {len(exported)} spans")
        st.json(exported)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# Reward Function Demo
st.markdown("### 🏆 Reward Function")

if st.button("🧪 Test Reward Computation"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        from src.training.rewards import DefaultRewardFunction
        from src.training.spans import SpanCollector
        
        collector = SpanCollector()
        collector.start_trace()
        span_id = collector.start_span("test")
        collector.end_span(span_id)
        
        reward_fn = DefaultRewardFunction()
        result = reward_fn.compute(
            "test_traj",
            [collector.get_span(span_id)],
            {"success": True, "evidence_count": 5, "causal_paths": 2}
        )
        
        col1, col2 = st.columns(2)
        col1.metric("Reward", f"{result.reward:.3f}")
        col2.metric("Completion", f"{result.components['completion']:.1%}")
        
        st.markdown("**Explanation:**")
        st.info(result.explanation)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# Trajectory Store
st.markdown("### 📦 Trajectory Storage")

if st.button("📋 View Trajectories"):
    try:
        import sys
        sys.path.insert(0, '/home/abishai/Desktop/causeway')
        from src.training.trajectories import TrajectoryStore, Trajectory
        
        store = TrajectoryStore()
        
        # Add sample trajectory
        traj = Trajectory(
            trajectory_id="traj_demo",
            trace_id="trace_demo",
            mode="mode2",
            input_data={"query": "Should we raise prices?"},
            spans=[],
            outcome={"success": True},
            reward=0.85
        )
        store.save(traj)
        
        st.success(f"✅ Total trajectories: {store.count()}")
        
        positive = store.list_positive_examples(reward_threshold=0.7)
        st.markdown(f"**Positive examples (reward >= 0.7):** {len(positive)}")
        
        for t in positive[:5]:
            st.markdown(f"- `{t.trajectory_id}`: reward={t.reward:.2f}, mode={t.mode}")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown("**✅ Prototype walkthrough complete!**")
