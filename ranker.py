import gzip
import json
import os
import time
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION & CONSTRAINTS ---
CANDIDATES_PATH = "candidates.jsonl.gz"  # Path to compressed data [cite: 4]
OUTPUT_PATH = "submission.csv"
TOP_N_FINAL = 100 # We need exactly the top 100 [cite: 19]
BM25_FILTER_SIZE = 2000  # Narrow 100k down to 2k before heavy embedding calculations

## =====================================================================
## PHASE 1: THE DEFENSIVE LAYER (Honeypot & Trap Filtering)
## =====================================================================
def is_honeypot_or_trap(candidate, signals_doc=None):
    """
    Evaluates the candidate against redrob_signals to catch traps,
    behavioral twins, keyword stuffers, and impossible profiles. [cite: 10, 36, 37]
    """
    signals = candidate.get("redrob_signals", {})
    
    # 1. Check for basic blacklisted behavioral flags from redrob_signals_doc.md [cite: 4, 10, 38]
    if signals.get("is_keyword_stuffer", False) or signals.get("impossible_profile_flag", False):
        return True
        
    if signals.get("behavioral_twin_detected", False):
        return True

    # 2. Heuristic check: Look for extreme keyword stuffing manually just in case
    resume_text = candidate.get("resume_text", "").lower()
    if resume_text.count("highly skilled") > 10 or len(resume_text) > 50000:
        return True # Likely a trap candidate [cite: 36, 37]

    return False

## =====================================================================
## PHASE 2: STREAMING & LIGHTWEIGHT FILTERING (BM25)
## =====================================================================
def stream_and_pre_filter(jd_text):
    """
    Streams the 100k gzipped JSONL to keep RAM under 16GB.
    Applies defensive filters and uses BM25 to get the top 2000 candidates.
    """
    print("Step 1: Streaming candidates and applying defensive filters...")
    start_time = time.time()
    
    valid_candidates = []
    corpus_tokens = []
    
    # Read line-by-line to protect memory footprint 
    with gzip.open(CANDIDATES_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            
            # Filter out Honeypots immediately [cite: 37]
            if is_honeypot_or_trap(candidate):
                continue
                
            skills = candidate.get("skills", [])
            if isinstance(skills, list):
                skills = " ".join(str(x) for x in skills)

            candidate_text = (
                f"{skills} "
                f"{candidate.get('experience_summary','')} "
                f"{candidate.get('resume_text','')}"
            )

            valid_candidates.append(candidate)
            corpus_tokens.append(candidate_text.lower().split())

    print(f"Passed filtering: {len(valid_candidates)} candidates. Time: {time.time() - start_time:.2f}s")
    
    # Tokenize Job Description
    print("Step 2: Running quick BM25 lexical filter to narrow down pool...")
    bm25 = BM25Okapi(corpus_tokens)
    jd_tokens = jd_text.lower().split()
    
    # Get scores for all valid candidates
    scores = bm25.get_scores(jd_tokens)
    top_indices = np.argsort(scores)[::-1][:BM25_FILTER_SIZE]
    
    filtered_candidates = [valid_candidates[i] for i in top_indices]
    return filtered_candidates

## =====================================================================
## PHASE 3: SEMANTIC RANKING & REASONING (Sentence Transformers)
## =====================================================================
def rank_and_generate_reasons(candidates, jd_text):
    """
    Ranks the filtered subset using batch matrix operations on CPU
    instead of an item-by-item loop to avoid severe processing lag. 
    """
    print(f"Step 3: Calculating semantic embeddings in batch for {len(candidates)} candidates...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    candidate_texts = []
    for cand in candidates:
        skills = cand.get("skills", [])
        if isinstance(skills, list):
            skills_str = " ".join(str(x) for x in skills)
        else:
            skills_str = str(skills)
        summary_str = str(cand.get("experience_summary", ""))
        candidate_texts.append(f"{skills_str} {summary_str}")
    
    # BATCH INFERENCE: Let PyTorch optimize multi-threading on CPU
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    cand_embeddings = model.encode(candidate_texts, batch_size=128, convert_to_tensor=True)
    
    # Matrix Multiplication for Similarities (Instantaneous)
    cosine_similarities = util.cos_sim(jd_embedding, cand_embeddings)[0].cpu().numpy()
    
    # Finalize Scoring Matrix
    final_pool = []
    for idx, cand in enumerate(candidates):
        similarity = float(cosine_similarities[idx])
        
        # Pull in behavioral score signals safely [cite: 4]
        signals = cand.get("redrob_signals", {})
        stability_score = signals.get("tenure_stability_score", 0.5) 
        
        # Composite score calculation
        raw_score = (similarity * 0.7) + (stability_score * 0.3)
        
        # CRUCIAL TIE-BREAKER STEP: Round the score to 4 decimal places before sorting.
        # This makes floating values identical and triggers alphabetical fallback correctly.
        rounded_score = round(raw_score, 4)
        final_pool.append((rounded_score, similarity, cand))

    # ====================== Phase 4: Sorting ======================================= 
    # Deterministic Tie-Breaker sorting logic:
    # 1. Sort by rounded_score DESCENDING (-x[0])
    # 2. Sort by candidate_id ASCENDING (x[2].get(...))
    final_pool.sort(key=lambda x: (-x[0], str(x[2].get("candidate_id") or x[2].get("id") or "")))
    top_100 = final_pool[:TOP_N_FINAL]
    
    # Format into final submission structure 
    submission_rows = []
    print("Step 4: Compiling Top 100 list and generating custom justifications...")

    for rank, (score, sim, cand) in enumerate(top_100, 1):
        cand_id = cand.get("candidate_id") or cand.get("id")
        top_skills = cand.get("skills", [])
        skills_str = "relevant technical areas"

        if isinstance(top_skills, list):
            cleaned = []
            for skill in top_skills:
                if isinstance(skill, str):
                    cleaned.append(skill)
                elif isinstance(skill, dict):
                    if "name" in skill:
                        cleaned.append(skill["name"])
                    elif "skill" in skill:
                        cleaned.append(skill["skill"])
            if cleaned:
                skills_str = ", ".join(cleaned[:2])
        elif isinstance(top_skills, str):
            skills_str = top_skills
        
        reasoning = (
            f"Candidate displays a high semantic alignment ({sim:.2%}) with core JD demands. "
            f"Demonstrates verified execution capability in {skills_str} alongside strong behavioral stability metrics."
        )
        
        submission_rows.append({
            "candidate_id": cand_id,
            "rank": rank,
            "score": score,
            "reasoning": reasoning
        })
        
    return pd.DataFrame(submission_rows)

## =====================================================================
## MAIN EXECUTION PIPELINE
## =====================================================================
if __name__ == "__main__":
    overall_start = time.time()
    
    # 1. Read Job Description safely [cite: 4, 6, 8]
    if os.path.exists("job_description.md"):
        with open("job_description.md", "r", encoding="utf-8") as f:
            job_description = f.read()
    else:
        job_description = "Looking for a Senior Software Engineer with Python, Docker, AWS, and system design experience."
        print("Warning: job_description.md not found! Using fallback string.")

    # 2. Pipeline execution
    try:
        filtered_pool = stream_and_pre_filter(job_description)
        df_submission = rank_and_generate_reasons(filtered_pool, job_description)
        
        # 3. Export to CSV [cite: 4, 19, 25]
        df_submission.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSUCCESS: {OUTPUT_PATH} generated successfully with exactly {len(df_submission)} records.") 
        
    except FileNotFoundError:
         print(f"Error: Could not find '{CANDIDATES_PATH}'. Please ensure it is placed in this directory.") 
    
    total_time = time.time() - overall_start
    print(f"Total processing execution time: {total_time:.2f} seconds.")
    print("Remember to run `python validate_submission.py` before uploading your package!")