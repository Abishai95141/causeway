"""
Page 2: Document Upload

Upload documents for evidence gathering.
"""

import streamlit as st
import requests
import hashlib
from datetime import datetime

st.set_page_config(page_title="Document Upload", page_icon="📁", layout="wide")

st.markdown("# 📁 Document Upload")
st.markdown("Upload evidence documents for the Causeway system to process.")

st.markdown("---")

st.markdown("""
> **Instructions:**
> 1. Upload one or more documents (PDF, TXT, MD, XLSX, CSV)
> 2. Each uploaded document gets a unique `doc_id`
> 3. Save the `doc_id` - you'll need it for indexing
""")

st.markdown("---")

# Upload Section
st.markdown("### 📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=["pdf", "txt", "md", "xlsx", "csv"],
    accept_multiple_files=True,
    help="Supported formats: PDF, TXT, Markdown, Excel, CSV"
)

# Store uploaded docs in session state
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

if uploaded_files:
    st.markdown("---")
    st.markdown("### 📋 Files Ready for Upload")
    
    for file in uploaded_files:
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**📄 {file.name}**")
            st.caption(f"Size: {file.size / 1024:.1f} KB | Type: {file.type}")
        
        with col2:
            # Show file preview for text files
            if file.type in ["text/plain", "text/markdown"]:
                content = file.read().decode('utf-8')[:500]
                file.seek(0)  # Reset file pointer
                with st.expander("Preview"):
                    st.text(content + "..." if len(content) == 500 else content)
        
        with col3:
            if st.button(f"Upload", key=f"upload_{file.name}"):
                # Upload to API
                try:
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    data = {"description": f"Uploaded via prototype at {datetime.now()}"}
                    
                    response = requests.post(
                        "http://localhost:8000/api/v1/uploads",
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.uploaded_docs.append(result)
                        st.success(f"✅ Uploaded! doc_id: `{result['doc_id']}`")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Make sure the API is running at http://localhost:8000")

st.markdown("---")

# Uploaded Documents History
st.markdown("### 📚 Uploaded Documents")

if st.session_state.uploaded_docs:
    for doc in st.session_state.uploaded_docs:
        with st.expander(f"📄 {doc['filename']} - `{doc['doc_id']}`"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Document ID:** `{doc['doc_id']}`")
                st.markdown(f"**Filename:** {doc['filename']}")
                st.markdown(f"**Status:** {doc['status']}")
            with col2:
                st.markdown(f"**Content Hash:** `{doc['content_hash'][:16]}...`")
                st.markdown(f"**Storage URI:** `{doc['storage_uri']}`")
                st.markdown(f"**Created:** {doc['created_at']}")
            
            # Copy doc_id button
            st.code(doc['doc_id'], language="text")
            st.caption("☝️ Copy this doc_id for the next step (Indexing)")
else:
    st.info("No documents uploaded yet. Upload some files above!")

# Manual doc_id entry
st.markdown("---")
st.markdown("### 🔍 Look Up Document")

doc_id_lookup = st.text_input("Enter doc_id to look up", placeholder="doc_abc123...")

if st.button("Look Up", key="lookup"):
    if doc_id_lookup:
        try:
            response = requests.get(
                f"http://localhost:8000/api/v1/documents/{doc_id_lookup}",
                timeout=10
            )
            if response.status_code == 200:
                st.success("✅ Document found!")
                st.json(response.json())
            else:
                st.error(f"❌ Not found: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a doc_id")

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✅ After uploading documents, proceed to <b>Document Indexing</b> →</p>
</div>
""", unsafe_allow_html=True)
