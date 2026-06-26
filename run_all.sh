#!/usr/bin/env bash
# Execute the full analysis pipeline in order.
# Run from the analysis/ directory: bash run_all.sh
# Requires: Python 3.10+, all packages in requirements.txt

set -e  # exit on first error

echo "[1/10] Collecting Reddit data..."
python3 01_collect_data.py

echo ""
echo "[2/10] Building interaction networks..."
python3 02_build_network.py

echo ""
echo "[3/10] Computing TF-IDF user vectors and cosine similarity..."
python3 03_tfidf_similarity.py

echo ""
echo "[4/10] Running QAP permutation test (content similarity)..."
python3 04_qap_test.py

echo ""
echo "[5/10] Running style robustness test (function-word similarity + QAP)..."
python3 10_style_robustness.py

echo ""
echo "[6/10] Running MRQAP with activity control..."
python3 11_mrqap_activity.py

echo ""
echo "[7/10] Running Louvain community detection..."
python3 05_community_detection.py

echo ""
echo "[8/10] Running LDA topic modelling per community..."
python3 06_lda_topics.py

echo ""
echo "[9/10] Sentiment analysis..."
python3 07_sentiment_analysis.py

echo ""
echo "[10/10] Generating paper figures..."
python3 09_figures_final.py

echo ""
echo "Pipeline complete. Results in results/, figures in figures/."
