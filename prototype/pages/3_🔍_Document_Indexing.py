"""
Page 3: Document Indexing

Index uploaded documents for retrieval.
"""

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

# Indexing Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Index a Document")
    
    # Get doc_ids from session if available
    doc_ids = []
    if 'uploaded_docs' in st.session_state:
        doc_ids = [doc['doc_id'] for doc in st.session_state.uploaded_docs]
    
    if doc_ids:
        doc_id = st.selectbox("Select from uploaded documents", doc_ids)
    else:
        doc_id = st.text_input("Enter doc_id", placeholder="doc_abc123...")
    
    if st.button("🔧 Index Document", key="index"):
        if doc_id:
            with st.spinner("Indexing document..."):
                try:
                    response = requests.post(
                        f"http://localhost:8000/api/v1/index/{doc_id}",
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.json(result)
                        
                        # Save to session
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
        for idx, doc_id in enumerate(st.session_state.indexed_docs):
            st.markdown(f"✅ `{doc_id}`")
    else:
        st.info("No documents indexed yet in this session.")

st.markdown("---")

# Retrieval Testing
st.markdown("### 🔎 Test Retrieval")
st.markdown("Test if documents are being retrieved correctly.")

query = st.text_input(
    "Enter a test query",
    placeholder="What factors affect pricing decisions?",
    help="Enter a question to search across indexed documents"
)

col_opts1, col_opts2 = st.columns([1, 4])
with col_opts1:
    top_k = st.number_input("Results", min_value=1, max_value=10, value=5)

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
        with st.spinner("Searching..."):
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

                        for i, r in enumerate(results):
                            with st.expander(f"Result {i+1}: Score {r['score']:.3f}"):
                                st.markdown("**Content:**")
                                content_text = r['content']
                                st.markdown(content_text[:500] + "..." if len(content_text) > 500 else content_text)
                                st.markdown(f"**Source:** `{r['doc_id']}` — *{r['doc_title']}*")
                                if r.get('page'):
                                    st.markdown(f"**Page:** {r['page']}")
                                if r.get('section'):
                                    st.markdown(f"**Section:** {r['section']}")
                    else:
                        st.warning("No results found. Try indexing more documents or a different query.")
                else:
                    st.error(f"❌ Search failed ({resp.status_code}): {resp.text[:300]}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server. Is it running on port 8000?")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a search query")

st.markdown("---")

# Haystack Status
st.markdown("### 📊 Retrieval Infrastructure Status")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Haystack Pipeline")
    if st.button("Check Haystack", key="haystack"):
        try:
            import sys
            sys.path.insert(0, '/home/abishai/Desktop/causeway')
            
            from src.haystack_svc.service import HaystackService
            
            service = HaystackService()
            st.success("✅ Haystack service initialized")
            st.info(f"Mode: {'Mock' if service.is_mock_mode else 'Production'}")
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
                st.success(f"✅ Qdrant connected - {len(collections)} collections")
                for coll in collections:
                    st.markdown(f"- `{coll.get('name', 'unknown')}`")
            else:
                st.warning(f"Qdrant returned {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to Qdrant: {e}")

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✅ After indexing documents, proceed to <b>World Model Builder</b> →</p>
</div>
""", unsafe_allow_html=True)
