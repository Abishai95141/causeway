"""
Page 3: Document Indexing

Index uploaded documents for retrieval and test semantic search quality.
"""

import sys
sys.path.insert(0, '/home/abishai/Desktop/causeway')

import streamlit as st
import requests

st.set_page_config(page_title="Document Indexing", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Document Indexing")
st.markdown("Index uploaded documents to make them searchable for evidence retrieval.")

st.markdown("---")

st.markdown("""
> **Instructions:**
> 1. Enter the `doc_id` from the previous upload step
> 2. Click "Index Document" to trigger indexing
> 3. Wait for indexing to complete
> 4. Test retrieval with a sample query
""")

st.markdown("---")

# ── Pipeline Health Banner ──────────────────────────────────────
st.markdown("### 🩺 Pipeline Health")
health_col1, health_col2, health_col3 = st.columns(3)

with health_col1:
    try:
        from src.haystack_svc.pipeline import HaystackPipeline
        import asyncio

        _p = HaystackPipeline()
        _loop = asyncio.new_event_loop()
        try:
            _loop.run_until_complete(_p.initialize())
        finally:
            _loop.close()
        if _p.is_mock_mode:
            st.error("⚠️ **MOCK MODE** — No real embeddings")
            st.caption("Install nltk, sentence-transformers, and ensure Qdrant is up")
        else:
            st.success("✅ **PRODUCTION MODE** — Real embeddings active")
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")

with health_col2:
    try:
        resp = requests.get("http://localhost:6333/collections/causeway_chunks", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            points = data.get("result", {}).get("points_count", 0)
            st.metric("📊 Vectors in Qdrant", points)
        else:
            st.metric("📊 Vectors in Qdrant", "N/A")
    except Exception:
        st.metric("📊 Vectors in Qdrant", "Offline")

with health_col3:
    try:
        resp = requests.get("http://localhost:8000/health", timeout=3)
        if resp.ok:
            st.success("✅ API Running")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline")

st.markdown("---")

# ── Indexing Section ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Index a Document")

    doc_ids = []
    if 'uploaded_docs' in st.session_state:
        doc_ids = [doc['doc_id'] for doc in st.session_state.uploaded_docs]

    if doc_ids:
        doc_id = st.selectbox("Select from uploaded documents", doc_ids)
    else:
        doc_id = st.text_input("Enter doc_id", placeholder="doc_abc123...")

    if st.button("🔧 Index Document", key="index"):
        if doc_id:
            with st.spinner("Indexing document (embedding with all-MiniLM-L6-v2)..."):
                try:
                    response = requests.post(
                        f"http://localhost:8000/api/v1/index/{doc_id}",
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.json(result)

                        if 'indexed_docs' not in st.session_state:
                            st.session_state.indexed_docs = []
                        st.session_state.indexed_docs.append(doc_id)
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a doc_id")

with col2:
    st.markdown("### 📋 Indexed Documents")

    if 'indexed_docs' in st.session_state and st.session_state.indexed_docs:
        for idx, did in enumerate(st.session_state.indexed_docs):
            st.markdown(f"✅ `{did}`")
    else:
        st.info("No documents indexed yet in this session.")

st.markdown("---")

# ── Retrieval Testing ──────────────────────────────────────────
st.markdown("### 🔎 Test Retrieval")
st.markdown("Test if documents are being retrieved correctly with **real semantic search**.")

query = st.text_input(
    "Enter a test query",
    placeholder="What factors affect pricing decisions?",
    help="Enter a question to search across indexed documents"
)

col_opts1, col_opts2 = st.columns([1, 4])
with col_opts1:
    top_k = st.number_input("Results", min_value=1, max_value=20, value=5)

# Build list of known doc_ids for scoping
all_doc_ids = []
if 'indexed_docs' in st.session_state:
    all_doc_ids = list(st.session_state.indexed_docs)
if 'uploaded_docs' in st.session_state:
    for d in st.session_state.uploaded_docs:
        if d['doc_id'] not in all_doc_ids:
            all_doc_ids.append(d['doc_id'])

filter_options = ["All documents"] + all_doc_ids
selected_filter = st.selectbox(
    "🔎 Scope search to document",
    filter_options,
    help="Select a specific document to restrict results, or search all",
)
filter_doc_id = None if selected_filter == "All documents" else selected_filter

if st.button("🔍 Search", key="search"):
    if query:
        with st.spinner("Searching with semantic embeddings..."):
            try:
                payload = {
                    "query": query,
                    "max_results": top_k,
                }
                if filter_doc_id:
                    payload["doc_id"] = filter_doc_id

                resp = requests.post(
                    "http://localhost:8000/api/v1/search",
                    json=payload,
                    timeout=15,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])

                    if results:
                        st.success(f"✅ Found {len(results)} results")

                        # Score distribution analysis
                        scores = [r['score'] for r in results]
                        max_score = max(scores)
                        min_score = min(scores)
                        score_range = max_score - min_score

                        # Score quality indicator
                        if score_range < 0.01 and len(results) > 1:
                            st.warning(
                                "⚠️ **All scores nearly identical** — this may "
                                "indicate mock mode or poor embeddings"
                            )
                        elif score_range > 0.1:
                            st.info(
                                f"📊 Score range: {min_score:.3f} – {max_score:.3f} "
                                f"(spread: {score_range:.3f}) — "
                                f"**Good semantic differentiation**"
                            )

                        for i, r in enumerate(results):
                            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"

                            with st.expander(
                                f"{rank_emoji}  Score: {r['score']:.4f}  |  "
                                f"{r['doc_title']}"
                            ):
                                # Score bar
                                st.progress(min(r['score'], 1.0))

                                st.markdown("**Content:**")
                                content_text = r['content']
                                st.markdown(
                                    content_text[:500] + "..."
                                    if len(content_text) > 500
                                    else content_text
                                )
                                st.markdown(
                                    f"**Source:** `{r['doc_id']}` — *{r['doc_title']}*"
                                )
                                if r.get('page'):
                                    st.markdown(f"**Page:** {r['page']}")
                                if r.get('section'):
                                    st.markdown(f"**Section:** {r['section']}")
                    else:
                        st.warning(
                            "No results found. Try indexing more documents or a "
                            "different query."
                        )
                else:
                    st.error(
                        f"❌ Search failed ({resp.status_code}): {resp.text[:300]}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to API server. Is it running on port 8000?"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a search query")

st.markdown("---")

# ── Multi-Query Robustness Test ─────────────────────────────────
st.markdown("### 🧪 Robustness Test")
st.markdown(
    "Run multiple diverse queries to test semantic differentiation. "
    "Good retrieval shows **varied** scores across queries."
)

robustness_queries = st.text_area(
    "Queries (one per line)",
    value=(
        "What are the main revenue drivers?\n"
        "How does soil health affect crop yield?\n"
        "What equipment is needed for a small farm?\n"
        "How to market farm products directly to consumers?\n"
        "What are common pests and diseases?"
    ),
    height=140,
)

if st.button("🚀 Run Robustness Test", key="robustness"):
    queries = [q.strip() for q in robustness_queries.strip().split("\n") if q.strip()]
    if not queries:
        st.warning("Enter at least one query")
    else:
        results_table = []
        progress = st.progress(0)

        for idx, q in enumerate(queries):
            try:
                payload = {"query": q, "max_results": 3}
                if filter_doc_id:
                    payload["doc_id"] = filter_doc_id
                resp = requests.post(
                    "http://localhost:8000/api/v1/search",
                    json=payload,
                    timeout=15,
                )
                if resp.status_code == 200:
                    hits = resp.json().get("results", [])
                    top_score = hits[0]["score"] if hits else 0.0
                    min_score = hits[-1]["score"] if hits else 0.0
                    snippet = hits[0]["content"][:80] + "..." if hits else "—"
                    results_table.append({
                        "Query": q[:50],
                        "Hits": len(hits),
                        "Top Score": f"{top_score:.4f}",
                        "Min Score": f"{min_score:.4f}",
                        "Top Snippet": snippet,
                    })
            except Exception as e:
                results_table.append({
                    "Query": q[:50], "Hits": 0,
                    "Top Score": "ERR", "Min Score": "ERR",
                    "Top Snippet": str(e)[:60],
                })
            progress.progress((idx + 1) / len(queries))

        st.dataframe(results_table, use_container_width=True)

        # Analyse score distribution
        top_scores = [
            float(r["Top Score"])
            for r in results_table
            if r["Top Score"] not in ("ERR", "0.0000")
        ]
        if len(top_scores) >= 2:
            import statistics
            std = statistics.stdev(top_scores)
            mean = statistics.mean(top_scores)
            if std > 0.05:
                st.success(
                    f"✅ **Good differentiation** — mean={mean:.3f}, "
                    f"stdev={std:.3f}"
                )
            else:
                st.warning(
                    f"⚠️ **Low differentiation** — mean={mean:.3f}, "
                    f"stdev={std:.3f}. May indicate mock mode."
                )

st.markdown("---")

# ── Infrastructure Status ──────────────────────────────────────
st.markdown("### 📊 Retrieval Infrastructure Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Haystack Pipeline")
    if st.button("Check Haystack", key="haystack"):
        try:
            import asyncio
            from src.haystack_svc.pipeline import HaystackPipeline

            async def _check():
                p = HaystackPipeline()
                await p.initialize()
                return p

            _loop = asyncio.new_event_loop()
            try:
                p = _loop.run_until_complete(_check())
            finally:
                _loop.close()
            if p.is_mock_mode:
                st.error("⚠️ **MOCK MODE** — simple word-overlap scoring")
                st.caption("Real embeddings NOT active. Check nltk, Qdrant.")
            else:
                st.success("✅ **Production Mode**")
                st.info(
                    f"Model: `all-MiniLM-L6-v2` (384-dim)\n\n"
                    f"Splitter: sentence-aware (3 sent/chunk, 1 overlap)\n\n"
                    f"Docs in store: {p.chunk_count}"
                )
        except Exception as e:
            st.error(f"❌ Error: {e}")

with col2:
    st.markdown("#### Qdrant Vector Store")
    if st.button("Check Qdrant Collections", key="qdrant"):
        try:
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                data = response.json()
                collections = data.get('result', {}).get('collections', [])
                st.success(f"✅ Qdrant connected — {len(collections)} collections")
                for coll in collections:
                    name = coll.get('name', 'unknown')
                    detail = requests.get(
                        f"http://localhost:6333/collections/{name}", timeout=3
                    )
                    if detail.status_code == 200:
                        info = detail.json().get("result", {})
                        pts = info.get("points_count", "?")
                        dim = info.get("config", {}).get("params", {}).get(
                            "vectors", {}
                        ).get("size", "?")
                        st.markdown(
                            f"- `{name}`: **{pts}** vectors, {dim}-dim"
                        )
                    else:
                        st.markdown(f"- `{name}`")
            else:
                st.warning(f"Qdrant returned {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to Qdrant: {e}")

st.markdown("---")

# ── Danger Zone: Purge All Document Data ───────────────────────
st.markdown("### 🗑️ Danger Zone")
st.markdown(
    "Permanently **delete all documents** from the vector store, "
    "object storage, and database. This action cannot be undone."
)

purge_confirmed = st.checkbox(
    "I understand this will permanently delete all documents",
    key="purge_confirm",
)

purge_clicked = st.button(
    "🗑️ Purge All Document Data",
    key="purge_btn",
    disabled=not purge_confirmed,
    type="primary",
)

if purge_clicked and purge_confirmed:
    with st.spinner("Purging all document data…"):
        try:
            resp = requests.post(
                "http://localhost:8000/api/v1/admin/purge-documents",
                json={"confirm": True},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    st.success(
                        f"✅ Purge complete — "
                        f"{data['documents_deleted']} document(s), "
                        f"{data['vectors_deleted']} vector(s), "
                        f"{data['files_deleted']} file(s) removed."
                    )
                else:
                    st.warning(
                        f"⚠️ Purge finished with errors: "
                        f"{'; '.join(data.get('errors', []))}"
                    )
                # Show warnings
                for w in data.get("warnings", []):
                    st.info(f"ℹ️ {w}")

                # Clear session-state document lists
                for key in ("uploaded_docs", "indexed_docs"):
                    if key in st.session_state:
                        del st.session_state[key]
            else:
                st.error(f"❌ API error ({resp.status_code}): {resp.text[:300]}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server. Is it running on port 8000?")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✅ After indexing documents, proceed to <b>World Model Builder</b> →</p>
</div>
""", unsafe_allow_html=True)
