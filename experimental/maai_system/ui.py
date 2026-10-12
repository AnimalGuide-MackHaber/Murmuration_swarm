import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import asyncio
import os

from database import client, CORE_KNOWLEDGE
from governor import calculate_telemetry, control_logic
from agents import process_producer_consumer, genai_client
from config import config

# Page configuration
st.set_page_config(
    page_title="MAAI Control Tower",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MAAI System: 3D Topological Visualization")

def fetch_data():
    try:
        results = client.query(
            collection_name=CORE_KNOWLEDGE,
            filter="id != ''",
            output_fields=["id", "content", "embedding", "confidence_score"],
            limit=1000
        )
        return results
    except Exception as e:
        st.error(f"Error fetching data from Milvus: {e}")
        return []

@st.cache_data
def get_embedding_models():
    try:
        models = []
        for m in genai_client.models.list():
            if "embedContent" in m.supported_actions:
                models.append(m.name)
        return sorted(models)
    except Exception as e:
        return ["models/gemini-embedding-2", "models/gemini-embedding-001"]

st.sidebar.header("Agent Controls")
seed_context = st.sidebar.text_input("Seed Context", value="The impact of multi-agent systems on open-ended learning")

st.sidebar.subheader("Model Configuration")
embedding_models = get_embedding_models()
default_idx = embedding_models.index(config.embedding_model_name) if config.embedding_model_name in embedding_models else 0
selected_embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    options=embedding_models,
    index=default_idx,
    help="Select the Gemini model used for vector embeddings."
)

if selected_embedding_model != config.embedding_model_name:
    config.update_setting("embedding_model_name", selected_embedding_model)
    st.toast(f"Embedding model updated to {selected_embedding_model}")

if st.sidebar.button("Run Producer Loop (1 iteration)"):
    if not config.google_api_key:
        st.sidebar.error("GEMINI_API_KEY not set in environment or config.")
    else:
        with st.spinner("Running Producer & Consumer..."):
            asyncio.run(process_producer_consumer(seed_context))
        st.sidebar.success("Loop complete!")

data = fetch_data()

if len(data) < 3:
    st.warning(f"Not enough data to visualize. Current nodes: {len(data)}. Please run the Producer loop to generate more nodes (requires at least 3 for UMAP).")
else:
    # Prepare data for visualization
    embeddings = np.array([res['embedding'] for res in data])
    contents = [res['content'] for res in data]
    ids = [res['id'] for res in data]
    scores = [res.get('confidence_score', 0.5) for res in data]

    # PCA Projection (fallback for UMAP)
    reducer = PCA(n_components=3, random_state=42)
    with st.spinner("Calculating PCA projection..."):
        projections = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': projections[:, 0],
        'y': projections[:, 1],
        'z': projections[:, 2],
        'id': ids,
        'content': [c[:100] + "..." if len(c) > 100 else c for c in contents],
        'confidence': scores
    })

    # Telemetry
    st.subheader("Criticality Governor Telemetry")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with st.spinner("Calculating Topology metrics..."):
        spectral_radius, betti_1 = calculate_telemetry(embeddings)
        governor_status = control_logic()
    
    col1.metric("Spectral Radius", f"{spectral_radius:.3f}")
    col2.metric("Betti-1 (H1 Holes)", f"{betti_1}")
    col3.metric("VSE Threshold", config.visitation_threshold)
    
    producer_status = "Paused" if config.producers_paused else "Running"
    col4.metric("Producers", producer_status)
    
    col5.info(f"Governor State: {governor_status}")

    # Manual Override Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Overrides")
    if st.sidebar.button("Manual Iconoclast (0.95)"):
        from agents import run_iconoclast
        run_iconoclast(0.95)
        st.sidebar.success("Iconoclast complete!")
        st.rerun()

    if st.sidebar.button("Manual XOH Injection"):
        from governor import inject_xoh_nodes
        inject_xoh_nodes(5)
        st.sidebar.success("XOH nodes injected!")
        st.rerun()

    # Plotly 3D Scatter
    st.subheader("Epistemic Network Topology")
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='confidence',
        hover_data=['id', 'content'],
        title="PCA 3D Projection of Core Knowledge",
        opacity=0.7,
        color_continuous_scale='Viridis'
    )
    
    # Update layout for modern aesthetics
    fig.update_layout(
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3',
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, width='stretch')
