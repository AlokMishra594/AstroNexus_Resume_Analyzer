# AstroNexus_Resume_Analyzer
## AI-powered candidate ranking system that can intelligently analyze 100,000 resumes, ranks the top 100 candidates based on semantic matching and behavioral signals, and generates explainable hiring recommendations in under 5 minutes.

## Features

- Analyze 100,000 candidate profiles efficiently
- Semantic resume matching using Sentence Transformers
- BM25 lexical retrieval for fast candidate filtering
- Behavioral signal scoring using Redrob Signals
- Honeypot and trap candidate detection
- AI-generated 2-sentence explanations for every shortlisted candidate
- CPU-optimized pipeline with <5 minute execution
- Export ranked candidates in submission-ready CSV format

- ## Tech Stack

- Python
- Sentence Transformers
- BM25 (rank-bm25)
- Pandas
- NumPy
- Scikit-learn
- JSONL Streaming


## Workflow

Job Description
        │
        ▼
Load Candidates data file(json, jsonl)
        │
        ▼
Defensive Filtering
(Honeypot & Trap Detection)
        │
        ▼
BM25 Candidate Retrieval
        │
        ▼
Semantic Similarity Ranking
        │
        ▼
Behavioral Signal Scoring
        │
        ▼
Weighted Ranking
        │
        ▼
Top 100 Candidates
        │
        ▼
AI Explanation Generation
        │
        ▼
Submission CSV




