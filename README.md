# 🌐 Network Analysis of Israel–Palestine Reddit Discussions

This project analyzes user interactions on Reddit to explore how the Israel–Palestine conflict influences network structure and discourse patterns. By constructing interaction networks from comments and submissions, the study observes how online behavior and polarization evolved during key conflict events.

## 📁 Files

- `network_construction.ipynb` – Builds directed user interaction networks based on replies and mentions  
- `network_analysis.ipynb` – Computes network statistics (e.g., centrality, density, modularity), community detection, and visualization  
- `text_preprocessing.ipynb` – Cleans comment text and prepares it for token-level or topic-level analysis  
- `paper.pdf` – The explanatory report

## 📊 Summary

### 1. Data Collection  
- Collected ~600k Reddit comments and posts related to Israel–Palestine  
- Included metadata such as timestamps, user, subreddit, and parent relationship

### 2. Network Construction & Analysis  
- Built user-by-user interaction networks (replies/mentions)  
- Analyzed structural properties:  
  - **Polarization** increased immediately following major conflict events  
  - Detected distinct **communities** often aligned with ideological stance  
- Visualized how networks reorganized around peaks of discourse

### 3. Topic & Text Analysis  
- Preprocessed content to identify recurring themes and sentiment  
- Compared language across communities to highlight polarization

### 4. Key Findings  
- Conflict escalation correlates with denser, more insular network structures  
- Moderated communities show reduced polarization over time  
- Topic modeling reveals divergent thematic focus across ideological groups

## 🔧 Tools & Libraries

- **Python** (pandas, NetworkX, matplotlib, numpy)
- Reddit data via torrent
- Community detection using Louvain method
- Topic modeling via gensim

## 📌 Notes

- Focus is on structural and textual trends; network interactions are anonymized
- Results are observational and correlational
- Could be extended by adding sentiment classification or cross-platform analysis

---

📖 Completed as part of a course in Simulations and Modeling
