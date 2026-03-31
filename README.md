# CS197 Project: Exploring Biases in Healthcare Data
**Course Staff:** Kara Liu (contact: karaliu@stanford.edu)

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197/CS195 AT STANFORD UNIVERSITY**

In this project you will be explore how to detect and mitigate biases[^0] present in real-world tabular[^1] datasets. Real-world data, particularly in healthcare, are a rich source of information in research that can help us understand trends, test new methods, and support real people like clinicians and patients. However, this data is often messy and imperfect. The common phrase "garbage in, garbage out" emphasizes how important data quality is to learn useful and accurate machine learning (ML) models. If the data is incomplete or noisy, our models may draw wrong conclusions and can even lead to unfair outcomes.

In this project, we will be working with the [eICU Collaborative Research Database (eICU-CRD)](https://eicu-crd.mit.edu/), a large, multi-center critical care dataset containing de-identified patient data from intensive care units (ICUs) across the United States. 

[^0] Yes, "bias" is a loaded term. We will clarify what we mean by this later.

[^1] Tabular means any dataset that contains a mixture of both discrete (country of origin, has hypertension, ...) and continuous (age, height, ....) variables with separate columns.
---

## A. Course Structure & Timeline

This 10-week project is divided into two phases:

* **Weeks 1-3: Onboarding.** Everyone must apply for access to the data and will get familiar with real-world data, which many of you may have not worked with before. Then, Each project will provide its own specialized curriculum and students through the foundational materials with interactive notebooks and suggested readings.
* **Weeks 4-10: Specialized Project Track.** Armed with this knowledge, you will choose a specialized project to own for the rest of the quarter, culminating in a final presentation and report.

The projects are as follows, with the hyperlinks attached:

Project 1: Missingness and selection bias
Project 2: Fairness
Project 3: Proxy learning 

## B. Getting started 

### Data Access
(IMPORTANT) Please complete this step **as soon as possible**, as it will take a bit of time.

1. First, you must request approval for the eICU dataset. To do so, follow the instructions [here](https://eicu-crd.mit.edu/gettingstarted/access/) which consists of three steps: 
    * Complete required CITI training (needed to work with patient data, even if de-identified; takes ~30 minutes  - 2 hours)
    * Register for an account on PhysioNet
    * Submit an application to PhysioNet for access to eICU data. *Note: When it asks for your supervisor name and email, you can give my contact info: Kara Liu, karaliu@stanford.edu*
2. Data acess will take about 1-5 days. If it has been over 7 days since you submitted your PhysioNet application and you have yet to hear back, please email me. 
3. While you wait for access, you can explore the [**demo** dataset](https://physionet.org/content/eicu-crd-demo/2.0.1/) which consists of about ~2,000 patients subsampled from the full ~200,000. You can download it via the terminal: 
    ```
    wget -r -N -c -np https://physionet.org/files/eicu-crd-demo/2.0.1/
    ```
    You do not need to unzip the individual files, we will be reading the `*.csv.gz` files directly. 
4. Once you are granted access to the full eICU dataset (expected in Week 2), you can navigate to the [full dataset's page](https://physionet.org/content/eicu-crd/2.0/) to start downloading the full eICU dataset. You can download via the terminal command:
    ```
    wget -r -N -c -np --user {YOUR_PHYSIONET_USERNAME} --ask-password https://physionet.org/files/eicu-crd/2.0/
    ```
    which will prompt you for your PhysioNet password before starting the download. Note, downloading may take several hours (it took me about ~7) as some of the files are quite large. 
5. After downloading, your folder should look something like
    ```
    eicu-collaborative-research-database-demo-2.0.1/
    | -- patient.csv.gz
    | -- lab.csv.gz
    | -- ...
    ```

Congratulations! You now have access to a real-world clinical dataset.

If this process felt slower or more involved than expected, that’s completely normal. Working with real-world data often includes several time-consuming steps, but the reward is working with data that contains real patient information. This tradeoff is all part of the reserach process.


### Setting up the environment

1. If you haven't already, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or Anaconda, which will help manage your project's software packages.

2. Open your terminal and run the following commands. We will create a fresh environment to ensure your dependencies don't conflict with other projects.
    ```
    # Create an environment named 'cs197-bias' with Python 3.12
    conda create --name cs197-bias python=3.12

    # Activate the new environment
    conda activate cs197-bias
    ```
    Then you can install all necessary package requirements. 
    ```
    cd cs197-bias # if you are not already there 
    pip install -r requirements.txt
    ```

3. Start by launching Jupyter and opening the first notebook, `1_Introduction.ipynb`, to dive into the core concepts.
    ```
    jupyter notebook
    ```


## C. Project 1: 

In this project, we will be exploring what it means to develop "fair" machine learning models. As ML is increasingly deployed in real-world settings such as healthcare, it is more important than ever that practitioners think carefully about how these models may lead to unfair outcomes.

The field of algorithmic fairness often focuses on technical definitions of fairness, that is, metrics or constraints that a model should satisfy to be deemed "fair". However, as we will see throughout this project, *fairness is not one-size-fits-all*. What it means for a model to be fair depends heavily on the context: what the model is predicting, who it affects, and how it is used in practice.

The goal of this project is to critically evaluate existing fairness metrics and potentially design fairness methods that better fit the problem setting.

### Week 1: 
**Onboarding:** 

 - Follow the [instructions provided in section B.](#b-getting-started) to get setup with the code and data.

**Readings:**
- [An Empirical Characterization of Fair Machine Learning For Clinical Risk Prediction](http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/): Please read and use for Assignment 1. 
- [A brief review on algorithmic fairness](https://link.springer.com/article/10.1007/s44176-022-00006-z) 


**Notebook:** 
- Walk through the notebook `week1_data_explore.ipynb` to become familiar with the eICU dataset. 

*For Assignment 1:*  Please read [An Empirical Characterization...]((http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/)) (for Part A: Read a Paper) and turn in your outputs of `week1_data_explore.ipynb` as a pdf (for Part 2: Section Starter Task).


### Week 2: 
**Onboarding:** 
- By the end of the week, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 

**Readings:**

- [Peeking into a black box, the fairness and generalizability of a MIMIC-III benchmarking model](https://www.nature.com/articles/s41597-021-01110-7) 
- [Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342)
- [Ensuring Fairness in Machine Learning to Advance Health Equity](https://www.acpjournals.org/doi/epdf/10.7326/M18-1990)
- [Algorithmic fairness in computational medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463525/)

**Video:**
- Watch [this video](https://www.youtube.com/watch?v=MzuoWAk9_AQ) from 21:00 to 1:21:00

**Notebook:** 
- Walk through the notebook `week2_alg_fairness.ipynb` to become familiar with algorithmic fairness calculations. 

*For Assignment 2:* 

*For Progress Reprot I:*

### Week 3: 


**Readings:**

**Readings:**

- [Peeking into a black box, the fairness and generalizability of a MIMIC-III benchmarking model](https://www.nature.com/articles/s41597-021-01110-7) 
- [Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342)
- [Ensuring Fairness in Machine Learning to Advance Health Equity](https://www.acpjournals.org/doi/epdf/10.7326/M18-1990)
- [Algorithmic fairness in computational medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463525/)

**Video:**
- Watch [this video](https://www.youtube.com/watch?v=MzuoWAk9_AQ) from 21:00 to 1:21:00

**Notebook:** 
- Walk through the notebook `week2_alg_fairness.ipynb` to become familiar with algorithmic fairness calculations. 

### Weeks 4-10: 

* As part of Assignment 1, we will be delving into a few key related works in this space. Please read the following works: 
* Notebook: Explore how the data works 
Week 2: 
* More suggestions on what to read
* Notebooks: start testing what these fairness concepts mean 
Week 3: 
* more suggestsions on what to read 
* different outcomes, different fairness concepts
Week 4:
* fairness without awareness vs fairness with awareness nb? 
Week 5:
Week 6:
Week 7: 
Week 8: 
Week 9:
Week 10: 
## Week 1: Setup
This week, we'll

    <!-- ***Note: If you are unable to import cg_rag inside the notebook, ensure you ran the pip install -e . command successfully.*** -->
#### C. Brushing up on the material 

**Objective:** Grasp the concept of "Structure Learning" using Dynamic Programming. How do we turn a stream of tokens into meaningful atomic units?
* **Materials:**
    * `tutorials/1_The_Geometry_of_Coherence.ipynb`: A hands-on walkthrough of the L0-segmentation algorithm.
    * `pipelines -> OfflinePipeline`
    * `core/geometry.py:` The implementation of the DP solver.
    * `tutorials/math_for_RAG_system.pdf`: the Mathematics behind the project, start by reading through this document.
* **Assignment 1: (Fri, Jan 16, 10:30AM)** Setup your environment (See above: Getting Started). Go through the assignments in `1_The_Geometry_of_Coherence.ipynb`. Then download a small corpus from huggingface and try to run the OfflinePipeline. Visualize some of the examples. Then describe the core/geometry codebase (min. 200 words). PrtSc all your results, put them in a .pdf and share on the handin website. Moreover, you have to solve the paper reading task [here](https://web.stanford.edu/class/cs197/assignments/a1.html) of [Piktus 2020](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf). Note that for the dynamic programming, the math in `math_for_RAG_system.pdf` paper could be useful. 

**Week 2: Stage II - The Pricing Engine**
* **Objective:** Master the core innovation—Pricing-Guided Selection. Understand the math of "Reduced Cost" and "Orthogonal Projection."
* **Materials:**
    * `tutorials/math_for_RAG_system.pdf`: the Mathematics behind the project, start by reading through this document.
    * `tutorials/2_The_Pricing_of_Selection.ipynb`: A hands-on walkthrough of the L0-pricing algorithm.
    * `core/pricing.py`: Deep dive into `update_residual` (Schur complement).
* **Progress report 1: (Sun, Jan 18, 8PM)** Go through the assignments in `2_The_Pricing_of_Selection.ipynb`. Then download a small corpus from huggingface and try to run the RetrievalPipeline. Visualize some of the examples. Then describe the pricing algorithm and codebase (min. 200 words). PrtSc all your results, put them in a .pdf and share on the handin website. For the pricing algorithm, the math in `math_for_RAG_system.pdf` paper could be useful.

**Week 3: The Status Quo & The Flaw**
* **Objective:** Understand standard Dense Retrieval and identifying its critical failure modes (Redundancy, Variable K, Noise).
* **Materials:**
    * [SQuAD 1.1 and 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
    * `tutorials/3_The_Architecture_of_RAG.ipynb`: You will build your own RAG system! And benchmark it!
    * `main.py`
    * [sentence-transformers](https://pypi.org/project/sentence-transformers/) (you will need this to get embeddings).
* **Progress report 2: (Sun, Jan 25, 8PM, GROUP HANDIN)** Go through the assignments in `3_The_Architecture_of_RAG.ipynb`. Then redo the experiments for [NewsQA](https://aclanthology.org/W17-2623.pdf).

From here on, progress reports will follow the standard template as you work through your "research" part of the course, now skilled in RAG systems and the CG-ICO framework!
---

## Phase 2: Specialized Project Tracks (Weeks 4-10)

Choose one of the following projects for your deep dive. Each project includes a motivation, a list of key skills you will develop, and a detailed week-by-week plan.

### Project A: Stable Retrieval with Recursive Language Models 🔁
[Paper](https://arxiv.org/abs/2512.24601v1)

**Motivation**

Recursive Language Models (RLMs) perform multi-step reasoning by repeatedly generating intermediate outputs and re-querying context. However, naïve retrieval at each step causes **context drift**, contradictions, and abandoned reasoning paths. CG-ICO’s pricing and residual framework offers a principled way to stabilize retrieval across reasoning steps, ensuring that each retrieval adds genuinely new information without disrupting prior context.

**Key Skills**

Agentic AI, Optimization, Linear Algebra, Retrieval Evaluation, Experimental Design

**Week-by-Week Plan**

**Week 4 – Failure Analysis of Naïve Multi-Step Retrieval**

Implement a simple iterative reasoning loop (e.g., multi-step QA with retrieval at each step). Log retrieved segments across steps using standard dense retrieval. Identify where context shifts occur, where redundancy accumulates, and when reasoning paths destabilize or contradict earlier conclusions.

**Week 5 – Instrumenting CG-ICO Across Retrieval Steps**

Extend the existing PricingEngine to persist residuals across multiple retrieval calls. Simulate multi-step reasoning queries and visualize which segments are retrieved at each step, how much overlap exists, and how the residual evolves across steps.

**Week 6 – Context Anchoring Mechanism**

Implement context anchoring: segments retrieved in early steps are “locked” and persist in the context window. Modify residual updates so locked segments are treated as fully explained. Compare stability against the unmodified pricing engine.

**Week 7 – Coherence-Penalized Reduced Cost**

Extend the reduced cost function with a coherence penalty that discourages selecting segments that are topically distant from already-selected context. Tune the penalty weight and evaluate whether this further reduces destabilizing context shifts.

**Week 8 – Dynamic Sparsity Scheduling**

Experiment with dynamic λ scheduling: begin retrieval with high sparsity (focused context) and gradually relax it as reasoning progresses. Analyze how this affects reasoning depth and consistency.

**Week 9 – Evaluation on Multi-Step Reasoning Tasks**

Evaluate on a small reasoning benchmark or curated multi-step queries. Measure final accuracy and reasoning stability metrics (e.g., reversals of prior conclusions, contradictions).

**Week 10 – Final Report & Presentation**

Deliver an analysis showing how naïve retrieval destabilizes RLMs, what modifications improve stability, and when stable retrieval matters most.

---

### Project B: Hybrid Sparse–Dense Retrieval with Pricing 💡
**Motivation**

Dense retrieval struggles with rare terms, exact matches, and entity-heavy queries, while sparse retrieval (BM25) fails on semantic paraphrases. This project integrates sparse and dense signals directly into the pricing framework, moving beyond naïve score fusion toward principled hybrid selection.

**Key Skills**

Information Retrieval, Optimization, Feature Fusion, Experimental Analysis

**Week-by-Week Plan**

**Week 4 – Sparse vs. Dense Failure Characterization**

Run dense retrieval and BM25 separately on a query set. Identify 30–50 cases where one succeeds and the other fails. Categorize failure modes (rare terms, paraphrases, entities, keyword overload).

**Week 5 – Naïve Score Fusion Baseline**

Implement simple score fusion (normalized BM25 + dense similarity). Tune weights on a dev set and evaluate gains and regressions relative to single-method baselines.

**Week 6 – Pricing-Aware Hybrid Reduced Cost**

Extend the reduced cost computation to include a sparse relevance term. Experiment with requiring both dense and sparse reduced costs to be negative before selecting a segment.

**Week 7 – Dual Residual Tracking**

Implement separate residuals:

* a **semantic residual** (dense embeddings)
* a **lexical residual** (query term coverage)
Update each independently and observe how retrieval behavior changes.

**Week 8 – Comparative Analysis of Fusion Strategies**

Compare naïve fusion, hybrid reduced cost, and dual-residual approaches. Analyze when each helps or hurts, particularly on entity- and keyword-heavy queries.

**Week 9 – Benchmark Evaluation**

Evaluate all methods on a general QA benchmark plus the failure cases from Week 4. Measure accuracy and retrieval efficiency.

**Week 10 – Final Report & Presentation**

Present a structured analysis of sparse vs. dense failure modes, fusion tradeoffs, and whether pricing-based hybrid retrieval justifies its added complexity.

---

### Project C: Context-Aware Sentence Embeddings for RAG 🧠

**Motivation**

Most RAG systems embed sentences independently and in fixed-sized chunks, ignoring cross-sentence dependencies like pronouns (“this”, “they”), discourse markers (“however”, “in fact”), and implicit entity references. This causes semantically incomplete embeddings across arbitrary paragraph boundaries that degrade segmentation, retrieval, and downstream answer quality. This project investigates **lightweight, drop-in context-aware sentence embeddings** using our Dynamic Programming, and measures their impact end-to-end in a real RAG system, not just in isolation.

**Key Skills**

Sentence Embeddings, Representation Learning, Retrieval Evaluation, RAG Systems, Experimental Design

**Week-by-Week Plan**

**Week 4 – Failure Mode Dataset**

Construct a compact diagnostic set (≈40–60 examples) from Wikipedia or news articles where isolated sentence embeddings clearly fail. Categorize failures (pronouns, discourse markers, entity carryover). This dataset will be reused throughout the project.

**Week 5 – Baseline RAG System**

Set up a competent, low-friction RAG baseline, for example:

* **SentenceTransformers + FAISS** for retrieval
* **LlamaIndex** or **Haystack** with a local or API-based LLM
* Fixed chunking and top-k retrieval
* Our solution from week 1.

Evaluate baseline performance on:

* Retrieval relevance (does the retrieved context contain the needed information?)
* Answer correctness on a small QA set derived from your diagnostic corpus

This establishes the reference point.

**Week 6 – Context-Aware Embedding Method**

Extend to Part 6 in week one assignment.

**Week 7 – Embedding-Level Evaluation**

Evaluate embeddings directly on the diagnostic set:

* Similarity to their referent context
* Clustering coherence
* Nearest-neighbor retrieval quality

Compare against independent sentence embeddings to verify the method actually fixes the identified failures.

**Week 8 – End-to-End RAG Evaluation**

Swap the baseline embedder for your context-aware embedder in the **same RAG system** from Week 5.

Measure:

* Retrieval quality (recall@k of relevant sentences)
* Answer accuracy
* Context efficiency (how many retrieved sentences are actually useful)

Analyze where improvements appear and where they do not.

**Week 9 – Tradeoff & Stress Testing**

Vary context window size and measure:

* Accuracy vs. compute cost
* Degradation when context is noisy or irrelevant

Identify the **sweet spot** between contextuality and over-smoothing.

**Week 10 – Final Report & Presentation**

Deliver a clear taxonomy of sentence-level embedding failures in RAG, a compact context-aware embedding method, quantitative evidence of improved RAG performance, and a practical guidance on when this is worth using.

---

### Project D: Graph-Based Retrieval for Multi-Hop Reasoning 🕸️
**Motivation**

The current retrieval treats segments as completely independent units. This breaks down for multi-hop questions like "What university did the CEO of the company that acquired Instagram attend?" To answer this you need to chain information across multiple segments: Instagram leads to Meta leads to Zuckerberg leads to Harvard. Flat retrieval can't do this.The goal is to build a graph structure over segments that enables multi-hop traversal during retrieval. You'd extract named entities from segments and create edges between segments that share entities. You could also link segments through coreference chains where different mentions refer to the same entity, or add edges based on high embedding similarity which might indicate elaboration or continuation. Then during retrieval you'd start from query-matched segments and expand along edges to gather supporting context.This should integrate with the existing pricing framework. You could extend the reduced cost calculation to account for graph connectivity, preferring segments that are reachable from already-selected ones. The final product is a graph-augmented retrieval system evaluated on multi-hop QA benchmarks like HotpotQA.

**Key Skills**

Graph Algorithms, Information Extraction, Multi-Hop QA, System Design

**Week-by-Week Plan**

**Week 4 – Multi-Hop Reasoning Analysis**

Manually analyze 20–30 multi-hop questions (e.g., HotpotQA, 2WikiMultihopQA). Identify required segments and what connects them (entities, temporal links, causality).

**Week 5 – Entity Extraction and Graph Construction**

Run NER over segments and build a segment–entity bipartite graph. Visualize graph structure and density.

**Week 6 – Segment-to-Segment Graph Edges**

Create segment–segment edges based on shared entities and/or embedding similarity. Experiment with edge weighting.

**Week 7 – Graph-Expanded Retrieval Baseline**

Implement a two-stage retrieval: dense retrieval followed by one- or two-hop graph expansion. Compare against flat retrieval.

**Week 8 – Pricing-Aware Graph Integration**

Modify reduced cost to favor graph-connected segments to already-selected ones. Analyze how this changes selection order.

**Week 9 – Multi-Hop QA Evaluation**

Evaluate on a subset of HotpotQA or similar. Measure retrieval recall for required hops and final QA accuracy.

**Week 10 – Final Report & Presentation**

Discuss when graph expansion helps, when it hurts, and which edge types matter most.
