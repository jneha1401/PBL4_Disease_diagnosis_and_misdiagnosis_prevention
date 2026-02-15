# Hybrid Rare‑Disease Diagnosis System

## Overview
This repository implements a hybrid AI system for **rare‑disease diagnosis** that treats diagnosis as a similarity‑based retrieval problem over a structured disease knowledge base.[web:6][web:12] Given a set of patient‑reported symptoms, the system returns a ranked list of candidate diseases, jointly leveraging neural disease embeddings and fuzzy logic over curated symptom frequencies.[web:7][web:15]

The approach supports **zero‑shot generalization** to rare diseases that are never seen during supervised training by embedding both common and rare diseases into a shared latent phenotypic space.[web:7]

---

## Core Ideas

- Formulate diagnosis as **similarity‑based retrieval** over a disease knowledge base with symptom‑frequency profiles.[web:15]
- Normalize noisy free‑text patient symptoms to standardized ontology concepts (e.g., HPO, Orphanet).[web:6][web:12]
- Learn a **neural disease embedding** from symptom‑frequency vectors of common diseases only.[web:7]
- Use **fuzzy logic** to interpret vague categorical frequency labels (“very frequent”, “occasional”, etc.).[web:2]
- Combine **cosine similarity** in embedding space with fuzzy compatibility into a hybrid ranking score.[web:2]
- Achieve **zero‑shot rare‑disease retrieval** by embedding rare diseases with frozen encoder weights and using a hybrid ranking function.[web:7]

---

## Knowledge Base

Each disease is associated with a **symptom‑frequency profile**, built from curated resources such as Orphanet and the Human Phenotype Ontology (HPO).[web:6][web:15]

- Symptoms are standardized ontology concepts (e.g., HPO terms).[web:15]
- Each symptom is labeled with one of:
  - Very frequent
  - Frequent
  - Occasional
  - Very rare
  - Excluded
- These categorical labels are later mapped to fuzzy membership degrees (low/medium/high).[web:2]

The knowledge base contains:
- **Common diseases**: used for supervised training of the neural encoder.[web:7]
- **Rare diseases**: held out from training and used for zero‑shot evaluation.[web:7]

---

## Pipeline Overview

The end‑to‑end pipeline:

1. Patient symptom input  
2. Hybrid symptom normalization  
3. Neural disease embedding encoder  
4. Fuzzy frequency interpretation  
5. Cosine similarity computation  
6. Hybrid ranking and top‑k disease output  

### 1. Patient Symptom Input
- Input: free‑text notes, EHR fields, or structured questionnaires with patient‑reported symptoms.[web:12]
- Symptoms may be noisy, colloquial, or misspelled; they must be normalized before downstream use.[web:12]

### 2. Hybrid Symptom Normalization

A three‑stage **hybrid normalization pipeline** maps each raw phrase to a standardized ontology concept.[web:12]

**2.1 Preprocessing**

- Lowercasing  
- Tokenization  
- Punctuation removal  
- Stopword filtering  

**2.2 Exact and Dictionary‑Based Matching**

- Use a curated dictionary of ontology labels and expert‑defined synonyms.
- Examples:  
  - “stomach pain” → “abdominal pain”  
  - “GI discomfort” → “gastrointestinal distress”  
- Exact and high‑confidence dictionary matches are accepted directly.

**2.3 Heuristic Token‑Overlap Matching**

For phrases not covered by the dictionary:

- Compute token‑level Jaccard similarity, character n‑gram overlap, or Levenshtein distance to ontology labels.[web:12]
- If similarity exceeds a tuned threshold, map the phrase to the best candidate concept.
- Captures:
  - Minor spelling errors  
  - Token reordering  
  - Morphological variants  

**2.4 Sentence‑Transformer Semantic Matching**

For remaining unmapped phrases:

- Encode the patient phrase and ontology concept descriptions using a biomedical sentence‑transformer (e.g., BERT/BioBERT variants).[web:12]
- Compute cosine similarity between phrase and concept embeddings.
- Select the highest scoring concept above a tuned confidence threshold.
- Preference:
  - If both dictionary and semantic candidates exist, prefer exact/dictionary matches.
  - If only semantic candidates exist, choose the highest cosine similarity.
- Phrases failing all steps are discarded as unmappable or ambiguous.

**Output:** a list of standardized symptom concepts compatible with the disease knowledge base.

---

## Neural Disease Embedding Network

Each disease is represented by a high‑dimensional vector encoding fuzzy‑scaled frequencies for all ontology symptoms.[web:7][web:15]

### Architecture

A feed‑forward encoder maps symptom‑frequency vectors to a low‑dimensional **disease embedding**:

- **Input layer:**  
  - Dimension = total number of ontology concepts.
  - Input: fuzzy‑scaled frequencies for each symptom.

- **Hidden layer 1:**  
  - 256 neurons, ReLU activation, dropout (0.3–0.5).  
  - Learns basic symptom co‑occurrence patterns.

- **Hidden layer 2:**  
  - 128 neurons, ReLU activation, dropout.  
  - Captures higher‑order interactions and organ‑system relationships.

- **Bottleneck layer:**  
  - 64 neurons, ReLU activation.  
  - Defines the final disease embedding (64‑D) used for similarity.

- **Classifier layer (training only):**  
  - Fully connected layer mapping 64‑D embeddings to logits over common diseases.  
  - Softmax activation for multi‑class classification.

### Training Protocol

- Train only on **common diseases** with labeled examples.[web:7]
- Loss: cross‑entropy between predicted and true disease class.
- Optimizer: Adam, learning rate ≈ 0.001 (tuned).
- Epochs: ~20, with potential early stopping based on validation loss/accuracy.
- Train/validation split: stratified by disease label to preserve class distribution.

### Inference and Embedding Space

- After training, discard the classifier layer.
- Use the 64‑D bottleneck output as the **fixed disease embedding**.
- Embed:
  - All common diseases (lookup table).
  - All rare diseases (using the same encoder weights on their symptom‑frequency profiles).
  - Patients (via their fuzzified symptom vectors).
- The resulting embedding space places phenotypically similar diseases close together, enabling similarity‑based retrieval.[web:7]

---

## Fuzzy Logic for Symptom Frequency

Symptom‑frequency labels are vague, population‑level descriptors; **fuzzy logic** is used to model them as graded degrees of presence.[web:2]

### 1. Fuzzy Membership Functions

Define three linguistic variables:

- Low frequency  
- Medium frequency  
- High frequency  

Each is represented by a triangular membership function over a normalized frequency axis.[web:2] Categorical labels are mapped to fuzzy membership degrees in each set, for example:

- Very frequent → high membership in “high”, moderate in “medium”, low in “low”  
- Frequent → moderate in “medium”, some in “high”  
- Occasional → more “low” and “medium” than “high”  
- Very rare → mostly “low”  
- Excluded → near‑zero membership in all sets  

Each symptom for each disease receives fuzzy weights (e.g., low = 0.2, medium = 0.5, high = 0.8), calibrated with clinical input.

### 2. Patient Symptom Fuzzification

- The patient’s symptom intensity (e.g., severity scores) is combined with the disease‑specific fuzzy weights.
- Fuzzy operators such as product or minimum are used to compute symptom‑level contributions.

### 3. Fuzzy Disease Score Aggregation

- Aggregate symptom‑level fuzzy contributions into a scalar **fuzzy compatibility score** \( F(p, d) \).
- Common strategies:
  - Weighted average (weights reflect symptom importance).
  - Max‑min composition rules.
- This score quantifies how well the patient’s presentation matches the disease’s expected symptom frequencies.

---

## Cosine Similarity in Embedding Space

The 64‑D embeddings encode **relative symptom patterns** rather than absolute magnitudes; cosine similarity is used as the geometric similarity measure.[web:2][web:7]

Given patient embedding \( e_p \) and disease embedding \( e_d \):

- Cosine similarity close to 1 → highly similar diseases.  
- Near 0 → unrelated.  
- Negative → opposing patterns (if present in practice).

A **patient pseudo‑embedding** is computed by feeding the patient’s fuzzified symptom vector through the trained encoder, and cosine similarity is computed w.r.t. all disease embeddings.

---

## Zero‑Shot Rare‑Disease Mapping

The model performs **zero‑shot** retrieval for rare diseases not used in supervised training.[web:7]

### 1. Medical Neighborhood Construction

For each rare disease:

- Compute cosine similarity between its embedding and all common disease embeddings.
- Store the top‑k (e.g., 3–5) most similar common diseases as its **medical neighborhood**.
- Provides interpretable context by linking rare diseases to more familiar common ones.

### 2. Inference and Retrieval

At test time:

- Compute the patient embedding.
- For each disease, compute:
  - Cosine similarity with the patient.
  - Fuzzy compatibility score.
- Rank diseases using the hybrid score (see below).
- A rare disease can be retrieved correctly if:
  - Its embedding lies near the patient embedding.
  - Its fuzzy score exceeds the compatibility threshold.

---

## Hybrid Ranking and Decision Fusion

Final ranking fuses **geometric similarity** and **fuzzy compatibility**.

### 1. Hybrid Score

For each disease \( d \):

\[
S(d) = \alpha \cdot \cos(e_p, e_d) + \beta \cdot F(p, d)
\]

- \( e_p \): patient embedding  
- \( e_d \): disease embedding  
- \( F(p, d) \): fuzzy compatibility score  
- \( \alpha, \beta \ge 0 \): weighting coefficients tuned on validation data  

Embedding norms are approximately constant and thus not explicitly modeled.

### 2. Fuzzy Thresholding

- Before ranking, filter out diseases with \( F(p, d) \) below a clinician‑defined threshold.
- This removes diseases that are incompatible with the observed symptom frequencies.

### 3. Ranking and Output

- Sort remaining diseases by \( S(d) \) in descending order.
- Return top‑k diseases as candidate diagnoses.
- Observations:
  - Fuzzy‑only model: often best Hit@1 for some rare diseases but limited diversity.  
  - Cosine‑only model: good global ranking but can be noisy.  
  - Hybrid model: best overall Hit@3/Hit@5, good balance between accuracy and diversity.

---

## Training, Tuning, and Evaluation

### Training Protocol

- Train encoder on common diseases with stratified train/validation split.[web:7]
- Tune:
  - Learning rate  
  - Dropout rate  
  - Number of epochs  
  - Fuzzy membership parameters  
  - Hybrid weights \( \alpha, \beta \)
- Use grid or random search with early stopping where beneficial.

### Evaluation Setup

- Evaluate on held‑out clinical cases including **rare diseases not in training**.[web:7]
- Pipeline steps:
  1. Symptom normalization  
  2. Fuzzification  
  3. Embedding computation  
  4. Fuzzy scoring  
  5. Hybrid ranking
- Metrics:
  - Hit@k (k = 1, 3, 5)  
  - Mean Reciprocal Rank (MRR)  
  - Precision/Recall@k  
- Report metrics separately for common vs. rare diseases.

### Qualitative Clinical Validation

- Clinicians review top‑k ranked diseases per case.
- Validate that non‑ground‑truth suggestions form a clinically plausible differential diagnosis set.

---

## Experimental Results (Summary)

- **Overall performance:**  
  - Strong Hit@5; correct diagnosis often within top few suggestions even if not rank 1.
- **Common diseases:**  
  - High Hit@1 and Hit@3 due to supervised training.
- **Rare diseases (zero‑shot):**  
  - Competitive Hit@3/Hit@5, demonstrating effective zero‑shot generalization.[web:7]
- **Model variants:**  
  - Cosine‑only: sensitive to noise from incomplete mappings.  
  - Fuzzy‑only: robust Hit@1, strong organ‑system filtering.  
  - Hybrid: best Hit@3/Hit@5, better differential diversity.
- **Fuzzy thresholding:**  
  - Modest gains in Hit@3/Hit@5, fewer implausible candidates.
- **Zero‑shot neighborhood behavior:**  
  - Rare diseases positioned near clinically similar common diseases in embedding space.[web:7]
- **Evolutionary weight optimization (ablation):**  
  - Genetic search over \( \alpha, \beta \) showed cosine similarity dominates; fuzzy term behaves like a low‑dynamic‑range offset; no consistent gains over manually tuned weights.
- **Training efficiency:**  
  - 20 epochs, batch size 32, convergence within minutes on a standard GPU.
- **Symptom normalization coverage:**  
  - 87–92% of patient phrases mapped; 8–13% discarded due to ambiguity or non‑medical content.[web:12]
- **Runtime performance:**  
  - <200 ms per case (CPU) for 5–10 symptoms and 500+ diseases; total memory footprint <50 MB.

---

## Implementation Sketch

### Dependencies (Typical)

- Python 3.9+
- Core:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `torch` or `tensorflow`
- NLP:
  - `transformers` (BERT/BioBERT)
  - `nltk` or `spaCy`
- String similarity / fuzzy matching:
  - `python-Levenshtein`, `rapidfuzz` or similar
- Optional:
  - `matplotlib`, `seaborn` for visualization

### Typical Workflow

1. **Build disease knowledge base**
   - Extract disease → symptom → categorical frequency from Orphanet/HPO or similar sources.[web:6][web:15]

2. **Train encoder**
   - Use only common diseases with labeled training data.
   - Save encoder weights and disease embedding table.

3. **Run inference**
   - Normalize patient symptom phrases to ontology concepts.
   - Fuzzify patient symptoms and compute patient embedding.
   - Compute fuzzy scores and cosine similarities vs. all diseases.
   - Apply fuzzy thresholding and hybrid scoring.
   - Return top‑k differential diagnoses.
