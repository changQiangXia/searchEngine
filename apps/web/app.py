"""NexusMind Web Application - Streamlit-based UI.

This module provides a web interface for NexusMind with:
- Semantic search with visual results
- 3D semantic galaxy visualization
- Concept interpolation explorer
- Attention heatmap visualization
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import numpy as np
from PIL import Image

# Page config must be first
st.set_page_config(
    page_title="NexusMind 🔮",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = "./data/workspaces/default"


def get_engine():
    """Get or initialize NexusEngine."""
    if st.session_state.engine is None:
        with st.spinner("🚀 Initializing NexusMind Engine..."):
            from nexus_mind.core.engine import NexusEngine
            st.session_state.engine = NexusEngine(
                workspace_dir=st.session_state.current_workspace
            )
    return st.session_state.engine


def render_header():
    """Render page header."""
    st.markdown('<h1 class="main-header">NexusMind 🔮</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Next-Generation Multimodal Semantic Search</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Workspace selection
        st.subheader("📁 Workspace")
        workspace = st.text_input(
            "Workspace Path",
            value=st.session_state.current_workspace,
        )
        if workspace != st.session_state.current_workspace:
            st.session_state.current_workspace = workspace
            st.session_state.engine = None  # Force reinit
            st.rerun()
        
        # Engine status
        engine = get_engine()
        if engine.index:
            st.success(f"✅ Index: {len(engine.index)} vectors")
        else:
            st.warning("⚠️ No index loaded")
        
        # Navigation
        st.subheader("🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["🔍 Search", "🌌 Semantic Galaxy", "🎭 Concept Explorer", "🔥 Attention Map", "📊 Stats"],
        )
        
        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.top_k = st.slider("Top K Results", 1, 50, 10)
        
        return page


def render_search_page():
    """Render main search page."""
    st.header("🔍 Semantic Search")
    
    engine = get_engine()
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_type = st.radio(
            "Query Type",
            ["Text", "Image"],
            horizontal=True,
        )
        
        if query_type == "Text":
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., a sunset over mountains",
            )
        else:
            query_file = st.file_uploader(
                "Upload query image",
                type=["jpg", "jpeg", "png"],
            )
            query = query_file if query_file else None
    
    with col2:
        st.write("")  # Spacer
        st.write("")
        
        # Search options
        use_diverse = st.checkbox("Diverse Results (MMR)")
        use_negative = st.checkbox("Negative Search")
        
        if use_negative:
            negative_query = st.text_input(
                "Exclude:",
                placeholder="e.g., people, cars",
            )
        else:
            negative_query = None
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.warning("Please enter a query")
            return
        
        if engine.index is None:
            st.error("No index loaded. Please build an index first.")
            return
        
        with st.spinner("Searching..."):
            try:
                if use_negative and negative_query and query_type == "Text":
                    results = engine.negative_search(
                        positive=query,
                        negative=negative_query,
                        top_k=st.session_state.top_k,
                    )
                elif use_diverse and query_type == "Text":
                    results = engine.search_diverse(
                        query=query,
                        top_k=st.session_state.top_k,
                    )
                else:
                    if query_type == "Image" and query_file:
                        image = Image.open(query_file)
                        results = engine.search(image, top_k=st.session_state.top_k)
                    else:
                        results = engine.search(query, top_k=st.session_state.top_k)
                
                # Store in history
                st.session_state.search_history.append({
                    "query": query if query_type == "Text" else "[Image]",
                    "results_count": len(results),
                })
                
                # Display results
                st.subheader(f"📊 Results ({len(results)} found)")
                
                cols = st.columns(4)
                for i, result in enumerate(results):
                    with cols[i % 4]:
                        with st.container():
                            # Try to load and display image
                            img_path = result["metadata"].get("path", "")
                            if Path(img_path).exists():
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, use_column_width=True)
                                except:
                                    st.error("Could not load image")
                            
                            # Show metadata
                            st.markdown(f"**Score:** {result['score']:.3f}")
                            st.caption(Path(img_path).name)
                            
                            # Expand for details
                            with st.expander("Details"):
                                st.json(result["metadata"])
                
            except Exception as e:
                st.error(f"Search failed: {e}")


def render_galaxy_page():
    """Render 3D semantic galaxy visualization."""
    st.header("🌌 Semantic Galaxy")
    st.info("3D visualization of your image collection in semantic space")
    
    engine = get_engine()
    
    if engine.index is None or len(engine.index) < 5:
        st.warning("Need at least 5 images in index to visualize galaxy")
        return
    
    # Galaxy parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Sample Size", 10, min(500, len(engine.index)), 100)
    with col2:
        dim_reduction = st.selectbox(
            "Dimensionality Reduction",
            ["PCA", "t-SNE", "UMAP"],
        )
    with col3:
        color_by = st.selectbox(
            "Color By",
            ["Similarity to Query", "Cluster", "Random"],
        )
    
    # Optional query for similarity coloring
    if color_by == "Similarity to Query":
        query = st.text_input(
            "Reference query for coloring",
            placeholder="e.g., sunset beach",
        )
    else:
        query = None
    
    if st.button("🌌 Generate Galaxy", type="primary"):
        with st.spinner("Computing galaxy..."):
            try:
                # Import here to avoid loading on startup
                from apps.web.visualizations.galaxy import generate_galaxy_plot
                
                fig = generate_galaxy_plot(
                    engine=engine,
                    n_samples=n_samples,
                    method=dim_reduction.lower(),
                    color_by=color_by,
                    query=query,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to generate galaxy: {e}")
                st.info("Note: Galaxy visualization requires embeddings to be cached")


def render_concept_explorer():
    """Render concept interpolation explorer."""
    st.header("🎭 Concept Explorer")
    st.info("Discover intermediate concepts between two ideas")
    
    engine = get_engine()
    
    if engine.index is None:
        st.error("No index loaded. Please build an index first.")
        return
    
    # Concept inputs
    col1, col2 = st.columns(2)
    
    with col1:
        concept_a = st.text_input(
            "Starting Concept",
            value="vintage",
            placeholder="e.g., vintage, cat, warm",
        )
    
    with col2:
        concept_b = st.text_input(
            "Ending Concept",
            value="futuristic",
            placeholder="e.g., futuristic, tiger, cold",
        )
    
    # Parameters
    col3, col4 = st.columns(2)
    with col3:
        steps = st.slider("Interpolation Steps", 3, 10, 5)
    with col4:
        top_k = st.slider("Results per Step", 1, 5, 2)
    
    if st.button("🎭 Explore Path", type="primary"):
        with st.spinner("Computing interpolation path..."):
            try:
                path = engine.interpolate_concepts(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    steps=steps,
                    top_k=top_k,
                )
                
                # Display as timeline
                st.subheader("🛤️ Concept Path")
                
                for i, point in enumerate(path):
                    col_img, col_info = st.columns([1, 3])
                    
                    with col_img:
                        if point["neighbors"]:
                            img_path = point["neighbors"][0]["metadata"].get("path", "")
                            if Path(img_path).exists():
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, use_column_width=True)
                                except:
                                    st.write("📷")
                    
                    with col_info:
                        st.markdown(f"### {point['description']}")
                        
                        if point["neighbors"]:
                            for neighbor in point["neighbors"]:
                                name = neighbor["metadata"].get("name", "Unknown")
                                score = neighbor["score"]
                                st.markdown(f"- **{name}** (score: {score:.3f})")
                        else:
                            st.caption("No matches found")
                    
                    if i < len(path) - 1:
                        st.markdown("⬇️")
                
            except Exception as e:
                st.error(f"Interpolation failed: {e}")


def render_attention_page():
    """Render attention heatmap visualization."""
    st.header("🔥 Attention Heatmap")
    st.info("Visualize where CLIP focuses its attention")
    
    engine = get_engine()
    
    # Image selection
    image_file = st.file_uploader(
        "Upload image to analyze",
        type=["jpg", "jpeg", "png"],
    )
    
    query = st.text_input(
        "Query text (optional)",
        placeholder="e.g., a red car",
    )
    
    if image_file and st.button("🔥 Generate Heatmap", type="primary"):
        with st.spinner("Computing attention..."):
            try:
                from apps.web.visualizations.attention import generate_attention_heatmap
                
                image = Image.open(image_file)
                
                heatmap, overlay = generate_attention_heatmap(
                    engine=engine,
                    image=image,
                    query=query if query else None,
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Heatmap")
                    st.image(heatmap, use_column_width=True)
                
                with col3:
                    st.subheader("Overlay")
                    st.image(overlay, use_column_width=True)
                
            except Exception as e:
                st.error(f"Failed to generate heatmap: {e}")
                st.info("Note: This feature requires Grad-CAM implementation")


def render_stats_page():
    """Render statistics page."""
    st.header("📊 System Statistics")
    
    engine = get_engine()
    
    # System info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📁 Index")
        if engine.index:
            st.metric("Total Vectors", len(engine.index))
            st.metric("Index Type", engine.index.index_type)
            st.metric("On GPU", "Yes" if engine.index.use_gpu else "No")
        else:
            st.warning("No index")
    
    with col2:
        st.subheader("🧠 Memory")
        stats = engine.get_stats()
        mem = stats.get("memory", {})
        st.metric("GPU Used", f"{mem.get('gpu_used_gb', 0):.2f} GB")
        st.metric("GPU Total", f"{mem.get('gpu_total_gb', 0):.2f} GB")
        st.metric("Usage", f"{mem.get('gpu_usage_pct', 0):.1f}%")
    
    with col3:
        st.subheader("🔍 Search History")
        history = st.session_state.search_history[-10:]  # Last 10
        if history:
            for h in reversed(history):
                st.caption(f"{h['query']}: {h['results_count']} results")
        else:
            st.caption("No searches yet")
    
    # Search history chart
    if st.session_state.search_history:
        st.subheader("📈 Search Activity")
        import pandas as pd
        df = pd.DataFrame(st.session_state.search_history)
        st.bar_chart(df.groupby("query")["results_count"].sum())


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    page = render_sidebar()
    
    # Route to appropriate page
    if "Search" in page:
        render_search_page()
    elif "Galaxy" in page:
        render_galaxy_page()
    elif "Concept" in page:
        render_concept_explorer()
    elif "Attention" in page:
        render_attention_page()
    elif "Stats" in page:
        render_stats_page()


if __name__ == "__main__":
    main()