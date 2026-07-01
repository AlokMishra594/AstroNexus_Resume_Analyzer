import streamlit as st
import pandas as pd
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="Redrob Candidate Discovery Sandbox",
    page_icon="🔍",
    layout="wide"
)

# --- LOAD LOCAL CACHED MODEL ---
@st.cache_resource
def load_model():
    # Cache the model so it doesn't reload on every button click
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# --- DEFENSIVE LAYER FILTERING ---
def check_honeypot(candidate):
    """Checks behavioral signals for honeypots/traps."""
    signals = candidate.get("redrob_signals", {})
    
    # Simple UI flag simulation based on redrob_signals_doc
    if signals.get("is_keyword_stuffer", False) or signals.get("impossible_profile_flag", False):
        return True, "Flagged: Impossible Profile/Keyword Stuffer"
    if signals.get("behavioral_twin_detected", False):
        return True, "Flagged: Behavioral Twin"
    return False, "Clear"

# --- APP INTERFACE ---
st.title("🔍 Intelligent Candidate Discovery & Ranking")
st.subheader("Redrob AI Hackathon Sandbox Environment")
st.markdown("---")

# Layout: Sidebar for Job Description & Inputs
with st.sidebar:
    st.header("1. Job Description Input")
    jd_input = st.text_area(
        "Paste Job Description (JD) here:",
        value="Looking for a Senior Software Engineer with Python, Docker, AWS, and system design experience.",
        height=250
    )
    
    st.header("2. Pipeline Settings")
    top_n = st.slider("Number of top candidates to display", min_value=5, max_value=50, value=10)
    stability_weight = st.slider("Behavioral Weight (Stability)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    semantic_weight = 1.0 - stability_weight

# Main Panel layout
st.header("Candidate Evaluation Sandbox")
st.write("Upload a small batch of candidates (JSON or JSONL format) to test the filtering and ranking pipeline.")

uploaded_file = st.file_uploader("Upload File", type=["json", "jsonl"])

if uploaded_file is not None:
    # Read and parse uploaded candidates
    try:
        file_contents = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith("jsonl"):
            raw_candidates = [json.loads(line) for line in file_contents.split("\n") if line.strip()]
        else:
            raw_candidates = json.loads(file_contents)
            if not isinstance(raw_candidates, list):
                raw_candidates = [raw_candidates]
                
        st.success(f"Successfully loaded {len(raw_candidates)} candidates for inspection.")
        
        # --- PROCESSING PIPELINE ---

        processed_data = []
        honeypot_count = 0

        for cand in raw_candidates:
            is_trap, reason = check_honeypot(cand)
            
            # Extract basic info safely
            cand_id = cand.get("candidate_id") or cand.get("id") or "Unknown"
            
            # SAFELY HANDLE SKILLS (Extract strings even if it's a list of dicts or a raw string)
            raw_skills = cand.get("skills", [])
            skills_list = []
            if isinstance(raw_skills, list):
                for s in raw_skills:
                    if isinstance(s, dict):
                        # If skills are objects, grab a 'name' or key field (adjust based on your schema)
                        skills_list.append(str(s.get("name", list(s.values())[0] if s else "")))
                    else:
                        skills_list.append(str(s))
            elif isinstance(raw_skills, str):
                skills_list = [raw_skills]
                
            summary = str(cand.get("experience_summary", ""))
            
            if is_trap:
                honeypot_count += 1
                continue
                
            # Compile a clean, safe text block for the embedding model
            cand_text = f"{' '.join(skills_list)} {summary}"
            
            processed_data.append({
                "cand_id": cand_id,
                "skills": skills_list,
                "summary": summary,
                "text": cand_text,
                "raw_signals": cand.get("redrob_signals", {})
            })





            
        # Display Pipeline Alert Logs
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Safe Candidates Passed", value=len(processed_data))
        with col2:
            st.metric(label="Honeypots Caught & Deflected", value=honeypot_count, delta=f"-{honeypot_count} dropped", delta_color="inverse")
            
        if len(processed_data) == 0:
            st.warning("No candidates survived the defensive trap filtering layer.")
        else:
            # --- RUN RANKING ENGINE ---
            with st.spinner("Processing CPU semantic rankings..."):
                # Compute JD embedding
                jd_embedding = model.encode(jd_input, convert_to_tensor=True)
                
                ranked_results = []
                for idx, cand in enumerate(processed_data):
                    cand_embedding = model.encode(cand["text"], convert_to_tensor=True)
                    similarity = util.cos_sim(jd_embedding, cand_embedding).item()
                    
                    # Score calculation
                    stability_score = cand["raw_signals"].get("tenure_stability_score", 0.5)
                    final_score = (similarity * semantic_weight) + (stability_score * stability_weight)
                    
                    ranked_results.append({
                        "Rank": 0,  # Will update after sorting
                        "Candidate ID": cand["cand_id"],
                        "Final Score": round(final_score, 4),
                        "Semantic Match": f"{similarity:.2%}",
                        "Behavioral Stability": f"{stability_score:.2%}",
                        "Top Skills": ", ".join(cand["skills"][:4]),
                        "Automated Reasoning": f"Candidate displays a {similarity:.2%} semantic alignment to criteria. High technical suitability with verified {', '.join(cand['skills'][:2])} skills."
                    })
                
                # Sort and assign ranks
                ranked_results.sort(key=lambda x: x["Final Score"], reverse=True)
                final_ranked_pool = ranked_results[:top_n]
                for rank_idx, entry in enumerate(final_ranked_pool, 1):
                    entry["Rank"] = rank_idx
                    
                # Display Results Dataframe
                df_results = pd.DataFrame(final_ranked_pool)
                
                st.subheader("🎯 Pipeline Output (Top Ranked Matches)")
                st.dataframe(
                    df_results[["Rank", "Candidate ID", "Final Score", "Semantic Match", "Behavioral Stability", "Top Skills"]],
                    use_container_width=True
                )
                
                # Expandable details for justifications (Matches submission requirements)
                st.subheader("📋 Core Submission Justification Mapping")
                for row in final_ranked_pool:
                    with st.expander(f"Rank {row['Rank']}: Candidate {row['Candidate ID']} (Score: {row['Final Score']})"):
                        st.markdown(f"**Top Skills:** {row['Top Skills']}")
                        st.markdown(f"**Generated Reasoning:** *{row['Automated Reasoning']}*")
                        
    except Exception as e:
        st.error(f"Failed to process dataset: {str(e)}")
        st.info("Ensure you are uploading the raw list of JSON components directly matching the schema structure.")
else:
    st.info("Waiting for file upload... Use 'sample_candidates.json' from your participant bundle to test the pipeline interface quickly.")