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
TASK_KEY = "mode1_build_task"

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

    # Document selector — restricts evidence to chosen documents
    _doc_options: list[dict] = []
    try:
        _doc_resp = requests.get(f"{API}/documents", timeout=5)
        if _doc_resp.ok:
            _doc_options = [
                d for d in _doc_resp.json()
                if d.get("status") == "indexed"
            ]
    except Exception:
        pass

    if _doc_options:
        _labels = {d["doc_id"]: f"{d['filename']}  ({d['doc_id']})" for d in _doc_options}
        selected_doc_ids = st.multiselect(
            "📄 Restrict to Documents (optional)",
            options=[d["doc_id"] for d in _doc_options],
            format_func=lambda did: _labels.get(did, did),
            help="If empty, all indexed documents are searched.",
        )
    else:
        selected_doc_ids = []

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

        # Start async build task in a background thread and persist in session state.
        result_box = _ThreadResult()
        payload = {
            "domain": domain,
            "initial_query": initial_query,
            "max_variables": max_variables,
            "max_edges": max_edges,
        }
        if selected_doc_ids:
            payload["doc_ids"] = selected_doc_ids
        thread = threading.Thread(
            target=_post_in_thread,
            args=(f"{API}/mode1/run", payload, result_box),
            daemon=True,
        )
        thread.start()

        st.session_state[TASK_KEY] = {
            "thread": thread,
            "result_box": result_box,
            "domain": domain,
            "started_at": time.time(),
        }
        st.rerun()


task = st.session_state.get(TASK_KEY)
if task:
    result_box: _ThreadResult = task["result_box"]
    elapsed = int(time.time() - task["started_at"])

    progress_bar = st.progress(0)
    status_text = st.empty()
    stage_display = st.empty()

    if not result_box.done:
        # Live stage polling while request is running.
        stage = _poll_stage()
        si = STAGE_INFO.get(stage)

        if si:
            progress_bar.progress(si["pct"])
            stage_display.info(f"{si['icon']}  **{si['label']}** — working…")
        else:
            # Fallback progress when stage endpoint is unavailable.
            synthetic = min(90, max(5, elapsed // 2))
            progress_bar.progress(synthetic)
            stage_display.info("⏳ Processing world model build request…")

        status_text.caption(f"⏳ Waiting for response… ({elapsed}s)")

        # Re-run every 2 seconds so UI stays responsive and updates continuously.
        time.sleep(2)
        st.rerun()

    # Task finished
    progress_bar.progress(100)
    stage_display.empty()

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
            st.session_state["last_domain"] = task["domain"]

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

                # ── Fetch and display model details ──────────
                try:
                    detail_resp = requests.get(
                        f"{API}/world-models/{task['domain']}/detail",
                        timeout=10,
                    )
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()

                        # Variables table
                        st.markdown("#### 🔍 Discovered Variables")
                        var_rows = [
                            {
                                "ID": v["variable_id"],
                                "Name": v["name"],
                                "Definition": v["definition"],
                                "Type": v.get("var_type", "—"),
                                "Role": v.get("role", "—"),
                            }
                            for v in detail.get("variables", [])
                        ]
                        if var_rows:
                            st.dataframe(var_rows, use_container_width=True)
                        else:
                            st.caption("No variables returned.")

                        # Edges table
                        st.markdown("#### 🔗 Causal Edges")
                        edge_rows = [
                            {
                                "From": e["from_var"],
                                "To": e["to_var"],
                                "Mechanism": e["mechanism"],
                                "Strength": e.get("strength", "—"),
                                "Confidence": f"{e['confidence']:.0%}" if e.get("confidence") is not None else "—",
                            }
                            for e in detail.get("edges", [])
                        ]
                        if edge_rows:
                            st.dataframe(edge_rows, use_container_width=True)
                        else:
                            st.caption("No edges returned.")
                    else:
                        st.warning("⚠️ Could not fetch model details for preview.")
                except Exception as detail_err:
                    st.warning(f"⚠️ Preview unavailable: {detail_err}")

                if st.button("✅ Approve & Activate Model", key="approve"):
                    with st.spinner("Approving…"):
                        try:
                            ar = requests.post(
                                f"{API}/mode1/approve",
                                json={"domain": task["domain"], "approved_by": "prototype_user"},
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

    # Clear finished task so the page is ready for the next run.
    st.session_state.pop(TASK_KEY, None)

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

# =====================================================================
# Update Existing Model — Incremental Patch
# =====================================================================
st.markdown("---")
st.header("🔧 Update Existing Model")
st.markdown(
    "Apply incremental changes to an existing world model: "
    "add or remove variables and edges, or tweak edge metadata."
)

update_domain = st.text_input(
    "Domain to update",
    placeholder="e.g. pricing",
    key="update_domain",
)

if update_domain:
    with st.expander("➕ Add Variables", expanded=False):
        add_var_id = st.text_input("Variable ID (snake_case)", key="av_id")
        add_var_name = st.text_input("Name", key="av_name")
        add_var_def = st.text_input("Definition", key="av_def")
        add_var_type = st.selectbox(
            "Type", ["CONTINUOUS", "BINARY", "CATEGORICAL", "ORDINAL", "COUNT"],
            key="av_type",
        )
        if st.button("Stage Variable", key="av_btn"):
            if add_var_id:
                staged = st.session_state.setdefault("_patch_add_vars", [])
                staged.append({
                    "variable_id": add_var_id,
                    "name": add_var_name or add_var_id,
                    "definition": add_var_def or "",
                    "type": add_var_type,
                })
                st.success(f"Staged variable **{add_var_id}**")

    with st.expander("➖ Remove Variables", expanded=False):
        rm_var = st.text_input("Variable ID to remove", key="rm_var")
        if st.button("Stage Removal", key="rm_btn"):
            if rm_var:
                staged = st.session_state.setdefault("_patch_rm_vars", [])
                staged.append(rm_var)
                st.success(f"Staged removal of **{rm_var}**")

    with st.expander("🔗 Add Edges", expanded=False):
        ae_from = st.text_input("From variable", key="ae_from")
        ae_to = st.text_input("To variable", key="ae_to")
        ae_mech = st.text_input("Mechanism", key="ae_mech")
        ae_str = st.selectbox(
            "Strength",
            ["HYPOTHESIS", "MODERATE", "STRONG", "CONTESTED"],
            key="ae_str",
        )
        if st.button("Stage Edge", key="ae_btn"):
            if ae_from and ae_to:
                staged = st.session_state.setdefault("_patch_add_edges", [])
                staged.append({
                    "from_var": ae_from,
                    "to_var": ae_to,
                    "mechanism": ae_mech or "",
                    "strength": ae_str,
                })
                st.success(f"Staged edge **{ae_from} → {ae_to}**")

    with st.expander("✂️ Remove Edges", expanded=False):
        re_from = st.text_input("From variable", key="re_from")
        re_to = st.text_input("To variable", key="re_to")
        if st.button("Stage Edge Removal", key="re_btn"):
            if re_from and re_to:
                staged = st.session_state.setdefault("_patch_rm_edges", [])
                staged.append({"from_var": re_from, "to_var": re_to})
                st.success(f"Staged removal of **{re_from} → {re_to}**")

    # Show staged changes
    staged_vars = st.session_state.get("_patch_add_vars", [])
    staged_rm = st.session_state.get("_patch_rm_vars", [])
    staged_edges = st.session_state.get("_patch_add_edges", [])
    staged_rm_edges = st.session_state.get("_patch_rm_edges", [])

    if any([staged_vars, staged_rm, staged_edges, staged_rm_edges]):
        st.markdown("#### Staged Changes")
        if staged_vars:
            st.write(f"**Add variables:** {[v['variable_id'] for v in staged_vars]}")
        if staged_rm:
            st.write(f"**Remove variables:** {staged_rm}")
        if staged_edges:
            edge_labels = [e['from_var'] + '→' + e['to_var'] for e in staged_edges]
            st.write(f"**Add edges:** {edge_labels}")
        if staged_rm_edges:
            rm_edge_labels = [e['from_var'] + '→' + e['to_var'] for e in staged_rm_edges]
            st.write(f"**Remove edges:** {rm_edge_labels}")

        if st.button("🚀 Apply Patch", key="apply_patch", type="primary"):
            payload = {
                "add_variables": staged_vars,
                "remove_variables": staged_rm,
                "add_edges": staged_edges,
                "remove_edges": staged_rm_edges,
                "update_edges": [],
            }
            with st.spinner("Applying patch…"):
                try:
                    resp = requests.patch(
                        f"{API}/world-models/{update_domain}",
                        json=payload,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ Patch applied!  "
                            f"+{data.get('variables_added',0)} vars, "
                            f"-{data.get('variables_removed',0)} vars, "
                            f"+{data.get('edges_added',0)} edges, "
                            f"-{data.get('edges_removed',0)} edges"
                        )
                        if data.get("conflicts"):
                            st.warning("Conflicts:\n" + "\n".join(data["conflicts"]))
                        st.json(data)
                        # Clear staged
                        for k in ["_patch_add_vars", "_patch_rm_vars", "_patch_add_edges", "_patch_rm_edges"]:
                            st.session_state.pop(k, None)
                    else:
                        st.error(f"❌ Patch failed ({resp.status_code}): {resp.text[:300]}")
                except Exception as exc:
                    st.error(f"❌ Error applying patch: {exc}")
