"""
Page 4: World Model Builder (Mode 1)

Build causal world models from evidence.
Uses a background thread + live stage polling so the request never
appears to "time out" in the browser.
"""

import streamlit as st
import requests
import time
import threading
from dataclasses import dataclass
from typing import Optional

API = "http://localhost:8000/api/v1"

st.set_page_config(page_title="World Model Builder", page_icon="🌐", layout="wide")

st.markdown("# 🌐 World Model Builder (Mode 1)")
st.markdown("Construct causal world models from your indexed evidence documents.")

# ── Stage metadata ──────────────────────────────────────────────
STAGE_INFO = {
    "variable_discovery":      {"icon": "🔍", "label": "Variable Discovery",       "pct": 15},
    "evidence_gathering":      {"icon": "📚", "label": "Evidence Gathering",       "pct": 35},
    "dag_drafting":            {"icon": "🗺️", "label": "DAG Drafting",             "pct": 55},
    "evidence_triangulation":  {"icon": "⚖️", "label": "Evidence Triangulation",   "pct": 75},
    "human_review":            {"icon": "👤", "label": "Human Review",              "pct": 95},
    "complete":                {"icon": "✅", "label": "Complete",                  "pct": 100},
}


# ── Helper: run POST in a thread ────────────────────────────────
@dataclass
class _ThreadResult:
    response: Optional[requests.Response] = None
    error: Optional[str] = None
    done: bool = False


def _post_in_thread(url: str, payload: dict, result: _ThreadResult, timeout: int = 600):
    """Fire a blocking POST and stash the result."""
    try:
        result.response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        result.error = "Request timed out after 10 minutes."
    except requests.exceptions.ConnectionError:
        result.error = "Cannot connect to API server. Is it running on port 8000?"
    except Exception as exc:
        result.error = str(exc)
    finally:
        result.done = True


def _poll_stage() -> str:
    """Ask the API for the current Mode 1 stage (best-effort)."""
    try:
        r = requests.get(f"{API}/mode1/status", timeout=3)
        if r.ok:
            return r.json().get("stage", "unknown")
    except Exception:
        pass
    return "unknown"


# ── Configuration form ──────────────────────────────────────────
st.markdown("---")

col_cfg, col_info = st.columns(2)

with col_cfg:
    st.markdown("### ⚙️ Configuration")

    domain = st.text_input(
        "Domain Name",
        placeholder="pricing",
        help="Name for this causal domain (e.g., pricing, marketing)",
    )

    initial_query = st.text_area(
        "Initial Query",
        placeholder="What are the key factors that influence product pricing and customer demand?",
        help="Describe what causal relationships you want to discover",
        height=100,
    )

    with st.expander("Advanced Options"):
        max_variables = st.slider("Max Variables", 5, 50, 20)
        max_edges = st.slider("Max Edges", 10, 100, 50)

with col_info:
    st.markdown("### 📖 Mode 1 Workflow")
    for stage_key in ["variable_discovery", "evidence_gathering", "dag_drafting",
                       "evidence_triangulation", "human_review"]:
        si = STAGE_INFO[stage_key]
        st.markdown(f"- {si['icon']} **{si['label']}**")

# ── Build button ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Build World Model")

if st.button("🔨 Build World Model", type="primary", use_container_width=True):
    if not domain or not initial_query:
        st.warning("Please enter both a domain name and an initial query.")
    else:
        # Quick connectivity check
        try:
            requests.get(f"{API}/mode1/status", timeout=3)
        except Exception:
            st.error(
                "❌ Cannot reach the API server at `localhost:8000`. "
                "Make sure it is running (`uvicorn src.api.main:app --reload --port 8000`)."
            )
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        stage_display = st.empty()

        # Launch the request in a background thread
        result_box = _ThreadResult()
        thread = threading.Thread(
            target=_post_in_thread,
            args=(
                f"{API}/mode1/run",
                {
                    "domain": domain,
                    "initial_query": initial_query,
                    "max_variables": max_variables,
                    "max_edges": max_edges,
                },
                result_box,
            ),
            daemon=True,
        )
        thread.start()

        # Poll until the thread finishes
        last_stage = ""
        elapsed = 0
        while not result_box.done:
            stage = _poll_stage()
            si = STAGE_INFO.get(stage)

            if si and stage != last_stage:
                progress_bar.progress(si["pct"])
                stage_display.info(f"{si['icon']}  **{si['label']}** — working…")
                last_stage = stage

            status_text.caption(f"⏳ Waiting for response… ({elapsed}s)")
            time.sleep(2)
            elapsed += 2

        progress_bar.progress(100)
        stage_display.empty()

        # ── Handle result ───────────────────────────────────────
        if result_box.error:
            status_text.error(f"❌ {result_box.error}")
        elif result_box.response is not None:
            resp = result_box.response
            if resp.status_code == 200:
                result = resp.json()
                status_text.success(
                    f"✅ World model built in ~{elapsed}s!  "
                    f"**{result['variables_discovered']} variables**, "
                    f"**{result['edges_created']} edges**, "
                    f"**{result['evidence_linked']} evidence bundles**"
                )

                # Persist in session
                st.session_state.setdefault("world_models", []).append(result)
                st.session_state["last_domain"] = domain

                # Metrics row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trace ID", result["trace_id"][:12] + "…")
                c2.metric("Variables", result["variables_discovered"])
                c3.metric("Edges", result["edges_created"])
                c4.metric("Evidence", result["evidence_linked"])

                with st.expander("📋 Full Response"):
                    st.json(result)

                if result.get("error"):
                    st.warning(f"⚠️ Note: {result['error']}")

                # ── Approval section ────────────────────────────
                if result.get("requires_review"):
                    st.markdown("---")
                    st.markdown("### 👤 Human Review")
                    st.info(
                        "This model is in **review** status. "
                        "Approve it to make it available for Mode 2 Decision Support."
                    )

                    if st.button("✅ Approve & Activate Model", key="approve"):
                        with st.spinner("Approving…"):
                            try:
                                ar = requests.post(
                                    f"{API}/mode1/approve",
                                    json={"domain": domain, "approved_by": "prototype_user"},
                                    timeout=30,
                                )
                                if ar.status_code == 200:
                                    ad = ar.json()
                                    st.success(
                                        f"✅ Model **{ad.get('version_id', '')}** approved!  "
                                        f"{ad.get('node_count', '?')} nodes, "
                                        f"{ad.get('edge_count', '?')} edges — "
                                        f"status **{ad.get('status', '?')}**"
                                    )
                                    st.json(ad)
                                else:
                                    st.error(f"❌ Approval failed ({ar.status_code}): {ar.text}")
                            except Exception as exc:
                                st.error(f"❌ Error approving: {exc}")
            else:
                status_text.error(f"❌ API returned {resp.status_code}: {resp.text[:300]}")
        else:
            status_text.error("❌ Unexpected error — no response received.")

# ── Previously built models ─────────────────────────────────────
if st.session_state.get("world_models"):
    st.markdown("---")
    st.markdown("### 📚 Previously Built Models (this session)")
    for i, wm in enumerate(reversed(st.session_state["world_models"])):
        with st.expander(
            f"{wm.get('domain', '?')} — {wm.get('variables_discovered', '?')} vars, "
            f"{wm.get('edges_created', '?')} edges  "
            f"({wm.get('stage', '?')})"
        ):
            st.json(wm)

st.markdown("---")
st.caption("✅ After building a world model, proceed to **Decision Support** →")
