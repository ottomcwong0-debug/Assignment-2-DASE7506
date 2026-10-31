# DASE7506 Machine Learning Project 2: Hard Verifier for Mathematical Reasoning

## 1. Introduction and Problem Statement

The Machine Learning Project 2 (MP2) for DASE7506 focuses on developing a robust **"Hard Verifier"** for mathematical reasoning tasks using the GSM8K dataset. Unlike standard validation sets where simple majority voting (consensus) effortlessly yields high accuracy, the test set consists of 467 specially filtered "hard" questions. In this test split, all candidate solutions exhibit **disagreement** (no unanimous consensus among candidate answers), making naive majority voting inadequate and necessitating advanced verification models.

The primary objectives of this project are twofold:

1. **Accuracy Maximization**: Achieve a as high as possible test set accuracy on the 467 hard questions.

2. **Computational Efficiency**: Ensure the total prediction time remains strictly under the $30.00$-second limit while adhering to the $16\text{ MiB}$ inference asset constraint.

---

## 2. Comparative Evolution: Baseline Method vs. Final Optimized Method

To understand the trajectory of our improvements, we contrast our **Initial Baseline Method** with our **Final Optimized Method** across architecture, feature engineering, training strategy, and performance.

| Dimension | Initial Baseline Method | Final Optimized Method |
| :--- | :--- | :--- |
| **Model Architecture** | Standard Logistic Regression ($\text{L2}$ penalty) on raw $1024$-dim candidate embeddings. | Heterogeneous Ensemble: Dual $\text{LR}$ + $\text{HistGradientBoosting}$ + $\text{ExtraTrees}$ + $\text{MLP}$ on PCA-compressed features ($48$-dim) + scalar statistics. |
| **Training Distribution** | Trained on all training samples (including $>55\%$ unanimous "easy" questions). | **Disagreement-Focused Training**: Filtered exclusively to conflict/disagreement subsets. |
| **Label Formulation** | First matching candidate labeled as $y=1$; all other matching candidates incorrectly penalized as $y=0$ (Label Contamination). | **Sanitized Labels**: All candidate solutions matching the gold reference are marked as $y=1$. |
| **Feature Engineering** | Pure embedding vector distance and cosine similarity metrics against peers. | **Symbolic & Semantic Features**: AST-based multi-step equation verification, last-equation match, number coverage, relative Z-scores. |
| **Decision Rule** | Greedy argmax over individual candidate probabilities ($\arg\max_i P(y=1 \mid c_i)$). | **Answer-Cluster Probability Aggregation**: Softmax group-level aggregation with dynamic voting weights tailored to pattern structures (e.g., $(3,1)$, $(2,2)$). |
| **Inference Latency** | $\sim 15.23$ seconds (Sequential candidate loop). | $\sim 1.54 - 1.70$ seconds (**Batch Vectorized Matrix Inference**). |
| **Test Accuracy** | $\sim 74.00\% - 75.37\%$ | **$80.09\%$ ($374 / 467$ correct)** |

---

## 3. Detailed Methodological Breakdown of the Final System

Our final solution adopts a multi-stage hybrid pipeline comprising **Disagreement-Focused Training**, **Symbolic & Semantic Feature Engineering**, **PCA-Enhanced Heterogeneous Ensemble**, and **Answer-Cluster Probability Aggregation**.

### 3.1 Disagreement-Focused Training & Label Sanitization

* **Domain Adaptation**: Standard training splits contain over $55\%$ unanimous "easy" questions (Pattern $(4,)$, $(3,)$, etc.). Training directly on these biases classifiers toward simple majority consensus. We filtered our training set to focus exclusively on **disagreement samples** ($\text{len}(\text{unique valid answers}) > 1$ or incomplete candidates), aligning the training distribution with the hard test split.

* **Label Contamination Fix**: In traditional verifiers, only the first matching candidate is labeled as positive ($y=1$), while other valid candidates sharing the gold answer are incorrectly penalized as negative ($y=0$). Our pipeline marks **all** candidates yielding the correct gold answer as positive, resolving training contradictions.

### 3.2 Symbolic, Semantic, and Relative Feature Engineering

To bridge the gap between abstract vector embeddings and mathematical correctness, we engineered 30+ specialized features per candidate:

* **Symbolic AST Equation Verification**: Using Python's `ast` parser, we dynamically evaluate multi-step arithmetic expressions ($a \odot b = c$) in candidate texts to detect calculation errors (`eq_wrong`) and verify whether the final computed value matches the reported answer (`last_match`).

* **Multilingual & Numerical Coverage**: Regex extractors and multilingual number dictionaries (e.g., converting `"twenty"` to $20$) compute overlap ratios between question numbers and candidate text numbers, quantifying numerical hallucination.

* **Relative Z-Score Standardization**: To eliminate absolute score biases across different questions, candidate features are normalized relative to their peer group within the same question.

### 3.3 PCA Compression & Heterogeneous Ensemble

High-dimensional embedding features ($1024$-dim per candidate) are processed through a dual-path architecture:

1. **Dimensionality Reduction**: Principal Component Analysis (PCA) compresses the embedding space to 48 principal components, capturing core semantic variance.

2. **Heterogeneous Classifiers**: We integrate dual regularized `LogisticRegression` models, a `HistGradientBoostingClassifier`, an `ExtraTreesClassifier`, and an `MLPClassifier`. This combines the linear robustness of logistic regression with tree- and neural-network-based non-linear feature interactions.

### 3.4 Answer-Cluster Probability Aggregation

Instead of greedily selecting a single candidate via argmax, the system groups candidates by their parsed numerical values. Group-level scores are computed by aggregating normalized candidate probabilities (mean and max) combined with vote-share weights dynamically tuned across different voting patterns (e.g., $(3, 1)$, $(2, 1, 1)$, $(2, 2)$ tie-breaking).

---

## 4. Experimental Results and Analysis

### 4.1 Quantitative Performance

Evaluated on the official test protocol (`dase7506-mp2-open-hard-verifier-v5`), our final system achieved:

* **Test Accuracy**: $80.09\%$ ($374 / 467$ correct), representing a $+4.72\%$ absolute improvement over our initial baseline ($75.37\%$).

* **Prediction Time**: $1.54 - 1.70$ **seconds** (a $10\times$ speedup over the baseline's $15.23\text{s}$, well within the $30.00$-second constraint).

### 4.2 Pattern-Wise Breakdown Analysis

Analyzing performance across different test split voting patterns highlights the strength of our answer-cluster probability aggregation compared to baseline majority voting:

* **Pattern** $(3, 1)$ **(**$196$ **questions)**: Achieved $87.8\%$ accuracy, successfully bypassing minority traps where baseline majority voting failed.

* **Pattern** $(2, 2)$ **(**$39$ **questions)**: In strictly tied zero-consensus scenarios where baseline majority voting achieved only $38.5\%$ accuracy, our verifier achieved $82.1\%$ accuracy (recovering $+17$ correct questions).

* **Pattern** $(1, 1, 1, 1)$ **(**$38$ **questions)**: In highly chaotic disagreement splits, accuracy reached $65.8\%$, vastly outperforming baseline random guessing or naive majority heuristics.

---

## 5. Conclusion

This project demonstrates that high-performance mathematical verification on hard test distributions can be successfully achieved through rigorous feature engineering (symbolic verification and PCA compression), targeted disagreement training, and structured probability aggregation. By systematically addressing baseline limitations (label contamination, linear model bottlenecks, and sequential inference overhead), the final system satisfies all accuracy ($>80\%$) and latency ($<30\text{s}$) benchmarks while remaining lightweight ($<16\text{ MiB}$ assets).
