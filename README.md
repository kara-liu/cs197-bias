# CS197 Project: Exploring Biases in Healthcare Data
**Course Staff:** Kara Liu (contact: karaliu@stanford.edu)

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197/CS195 AT STANFORD UNIVERSITY**

In this project you will be explore how to detect and mitigate biases $^{[0]}$ present in real-world tabular $^{[1]}$ datasets. Real-world data, particularly in healthcare, are a rich source of information in research that can help us understand trends, test new methods, and support real people like clinicians and patients. However, this data is often messy and imperfect. The common phrase "garbage in, garbage out" emphasizes how important data quality is to learn useful and accurate machine learning (ML) models. If the data is incomplete or noisy, our models may draw wrong conclusions and can even lead to unfair outcomes.

In this project, we will be working with the [eICU Collaborative Research Database (eICU-CRD)](https://eicu-crd.mit.edu/), a large, multi-center critical care dataset containing de-identified patient data from intensive care units (ICUs) across the United States. 

$^{[0]}$ Yes, "bias" is a loaded term. We will clarify what we mean by this later.

$^{[1]}$  Tabular means any dataset that contains a mixture of both discrete (country of origin, has hypertension, ...) and continuous (age, height, ....) variables with separate columns.

---

## A. Course Structure & Timeline

This 10-week project is divided into two phases:

* **Weeks 1-2: Onboarding.** Everyone must apply for access to the data and will get familiar with real-world data through an interactive notebook
* **Weeks 3-10: Specialized Project Track.** In Weeks 3 and 4, each project will provide its own specialized curriculum through interactive notebooks and suggested readings. In Weeks 5-10, you will choose a specialized project to own for the rest of the quarter, culminating in a final presentation and report.

The projects are as follows, with the hyperlinks attached:

- [Project 1: Fair ML in healthcare](#c-project-1-fair-ml-in-healthcare)
- [Project 2: Improving selection bias with group DRO](#d-project-2-improving-selection-bias-with-group-dro)

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
5. After downloading, your folder (for me, it downloaded to the path `./physionet.org/files/eicu-crd/2.0/`) should look something like
    ```
    ./physionet.org/files/eicu-crd/2.0/
    | -- patient.csv.gz
    | -- lab.csv.gz
    | -- hospital.csv.gz
    | -- ...
    ```

Congratulations! You now have access to a real-world clinical dataset.

If this process felt slower or more involved than expected, that’s completely normal. Working with real-world data often includes several time-consuming steps, but the reward is working with data that contains real patient information. This tradeoff is all part of the reserach process.


### Setting up the environment

1. If you haven't already, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or Anaconda, which will help manage your project's software packages.

2. (Optional) Open your terminal and run the following commands. This step is optional but we recommend it as it is good practice to always create a fresh environment to ensure your dependencies don't conflict with other projects.  
    ```
    # Create an environment named 'cs197' with Python 3.12
    conda create --name cs197 python=3.12

    # Activate the new environment
    conda activate cs197
    ```
3. After downloading this repository (i.e. using `git clone`), cd into it and install all necessary package requirements. 
    ```
    cd cs197-bias # if you are not already there 
    pip install -r requirements.txt
    ```

3. Test that you can open the first notebook `week1_explore_eicu_data.ipynb` by launching (where if applicable, make sure you are in the correct conda environment): 
    ```
    jupyter notebook
    ```


## C. Project 1: Fair ML in healthcare


Consider the "standard" algorithmic fairness framework as defined by three key features:
* **A1:** Fairness is evaluated with respect to a small set of predefined sensitive attributes, typically race and sex.
* **A2:** Fairness is measured using a fixed set of standard metrics, such as demographic parity, equal opportunity, equalized odds, and calibration.
* **A3:** Missing data is treated as a preprocessing issue to be handled before evaluating mordel performance - typically, either by complete-case analysis (dropping patients with missing data) or by imputation. 

The research project for the remainder of the quarter is: *How can we improve this standard framework to better account for the information and structure contained in missingness?* Rather than treating missing data as something to "fix", this project investigates whether missingness itself reveals important patterns useable for constructing a better algorithmic fairness framework. 

This project can be broken down into several guiding questions:
* **Q1:** How informative is missingness in the eICU dataset?
* **Q2:** Where (and how) does the standard fairness framework break under missingness?
* **Q3:** Can we extend the fairness framework to incorporate missingness?

The (revised) schedule for the rest of the quarter is as follows, where each weeks is framed by the central questions your group will try to answer that week: 
* **Week 4**: Is missingness random, or informative, in the eICU dataset? 
* **Week 5**: Can missingness itself be a "sensitive attribute" to evaluate fairness over (an alternative to race / sex from A1)?
* **Week 6**: How do different missing data handling strategies (A3) affect fairness metrics?  
* **Week 7**: How can we extend the "standard" fairness frameworks listed in A1, A2, or A3 to better handle or incorporate missingness? 
* **Week 8-10**: Experiments / writing 

### Weeks 1 - 2: 
**Onboarding:** 

 - Follow the [instructions provided in section B.](#b-getting-started) to get setup with the code and data.

**Readings:**
- [An Empirical Characterization of Fair Machine Learning For Clinical Risk Prediction](http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/): Please read and use for Assignment 1.
- [A brief review on algorithmic fairness](https://link.springer.com/article/10.1007/s44176-022-00006-z): Review paper, feel free to skim parts that are already familiar to you. 

**Additional Readings:** Review papers on the general field of bias in healthcare data. Optional but highly encouraged. 
- [Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6347576/)
- [Unmasking bias in artificial intelligence: a systematic review of bias detection and mitigation strategies in electronic health record-based models](https://pubmed.ncbi.nlm.nih.gov/38520723/)
- [Bias in medical AI: Implications for clinical decision-making](https://pmc.ncbi.nlm.nih.gov/articles/PMC11542778/)
- [Lessons and tips for designing a machine learning study using EHR data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8057454/)

**Notebook:** 
- Walk through the notebook `week1_explore_eicu_data.ipynb` to become familiar with the eICU dataset. 

*For Assignment 1 (Due April 12):*  Please read [An Empirical Characterization...]((http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/)) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).

*For Progress Report 1 (Due April 12):* Meet with your project group and submit what you all want to accomplish for Week 3. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.

### Week 3: 
**Onboarding:** 
- By Wednesday, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 
- You will need to generate and save the full dataset using the notebook from Week 2. 

**Additional Readings:** In conjunction with the readings for Assignment 2. 

- [Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342)
- [Ensuring Fairness in Machine Learning to Advance Health Equity](https://www.acpjournals.org/doi/epdf/10.7326/M18-1990)
- [Algorithmic fairness in computational medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463525/)

**Video:**
- Watch [this video](https://www.youtube.com/watch?v=MzuoWAk9_AQ) from 21:00 to 1:21:00

**Notebook:** 
- Walk through and complete the notebook `proj1/week2_fairness_evaluation.ipynb` to become familiar with algorithmic fairness calculations.

*[For Assignment 2](https://web.stanford.edu/class/cs197/assignments/project.html#related-work) (Due April 16):* You will explore related work in this field. Please see the "nearest-neighbor" papers for this project [here](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing). 

*For Progress Report 2 (Due April 19):* Meet with your project group and submit what you all want to accomplish for Week 4. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.


### Week 4 (April 20 - 26): Is missingness informative?
**Goal:**
Is missingness random, or informative, in the eICU dataset?

**Readings:**
- [Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness](https://proceedings.mlr.press/v193/jeanselme22a/jeanselme22a.pdf) - read in-depth
- [Fairness in Missing Data Imputation](https://arxiv.org/pdf/2110.12002) - read in-depth
- [Exploring the Inequitable Impact of Data Missingness on Fairness in Machine Learning](https://ieeexplore.ieee.org/document/10920480) - skim
- [Adapting Fairness Interventions to Missing Values](https://arxiv.org/pdf/2305.19429) - skim
- [Missing data and multiple imputation in clinical epidemiological research](https://pmc.ncbi.nlm.nih.gov/articles/PMC5358992/) - skim


**Tasks**: 

(Note: The `week3_fairness_tradeoffs.ipynb` notebook is now entirely optional / not required.)

1. Examine the distribution of missing features (missingness frequency per feature and per patient, and then an overall histogram of missingness frequency across all features and all patients). Plot using seaborn to visualize your findings. 
2. Analyze how missingness varies across: sex, ethnicity, age, hospital characteristics (e.g., hospitalid, region, bed count). Do certain groups systematically have more or less complete data? Use a statistical test (i.e., two-sampled KS test, see [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC8327789/) if you are unfamiliar with statistical tests) and plot using seaborn to determine the answer. Furthermore, do missing values correlate across the features themselves? 
3. Construct a "missingness attribute". For example, you can define patient groups based on if they have low / medium / high rates of feature missingness, you can cluster using KNN based on binary missingness masks. Visualize and interpret these groups.
4. Train a classifier model using only missingness indicators (binary features) to predict mortality. Use interpretability tools (i.e., coefficients (for linear classifiers), or SHAP values (for tree-based including XGB classifiers)). Identify which missing features are most predictive and see if this makes sense given the task at hand.
5. Based on your findings, argue if missingness is random (uninformative), or structured (reflecting clinical processes, access, or data collection differences). Relate this to MAR, MCAR, and MNAR (you should have learned about these in your assigned readings!). 
6. Thoroughly discuss how missingness may impact (good or bad) the "standard" fairness evaluation framework.  

**Deliverables:**
* Progress Report 3 (Due April 26) - a minimum 2-page writeup plus a notebook of all the completed tasks above. As a reminder: $\checkmark+$ = 100% indicates you went above and beyond; $\checkmark$ = 95% indicates basic completeness. 
* Introduction (Due April 23) - see website.

### Week 5 (April 27 - May 3): Can missingness be a sensitive attribute?

**Goal**:
Can missingness itself be treated as a "sensitive attribute" for evaluating fairness, as an alternative or complement to race and sex?

**Tasks**: 
1. Decide on a way to categorize missingness levels (i.e., using your "missingness attribute" from Week 4). For a classifier trained to predict mortality, pick one imputaiton strategy (we will analyze this strategy further next week), and analyze the "standard" four metrics of fairness (see A2 above) using the following 3 sensitive attributes: (1) missingness category, (2) race, and (3) sex. Which missingness group perform the best and the worst? How does this performance compare to the best / worst groups defined by race and sex? 
2. Construct intersectional subgroups: e.g., (race × missingness level), (sex × missingness level), (sex x race x missingness level). Evaluate fairness performance across these groups, and report the bootstrapped variance of these fairness metrics (so we can see how small sample size affects the consistency of fairness metrics). Summarize your findings. 
3. Repeat these experiments where we look at a different "missingness attribute" based on a different set of feature missingness. For example, if your original category was based on missingness of ALL variables, evaluate over a new cateogry of missingness of JUST lab variables. 
4. Reflect: Should missingness be considered a fairness-relevant attribute? Why or why not? 

**Deliverables:**
* Progress Report 4 (Due May 3) - a minimum 2-page writeup plus a notebook of all the completed tasks above 
* (Optional) Related Works - based on this project direction, refine your current related works section to focus in on algorihtmic fairness specifically with respect to missing data. You will need to do this eventually, so might be a good idea to work on that this week. 

### Week 6 (May 4 - 10): Does missing data handling affect fairness evaluations?

**Goal**:
Do different missing data handling strategies lead to different fairness conclusions?  

**Tasks**:
1. Define a fairness evaluation protocol (i.e., fix a model, split, and evaluation metrics). 
2. Implement at least three strategies for handling missing data, such as (a) complete-case analysis - dropping all patients who have feature/s X missing, (b) simple imputation (mean/median), (c) using missingness indicators, (d) MICE or other advanced method. For each, track dataset size (how many patients remain), and track feature distributions (before vs after).
3. Measure fairness using the four standard metrics across all group definitions (i.e., race, sex, age, missingness attribute, and intersecitonal groups). Compare how the different strategies for handling missing data affect the fairness results. 
4. Introduce mild synthetic missingness (e.g., drop 10–20% of values randomly = MCAR or based on some attribute = MAR). Re-run preprocessing pipeline and train on this synthetically missing data, but when you evaluate, evaluate on the original dataset without the 10-20% dropped. Check whether fairness conclusions remain stable.
5. Reflect on how missing data preprocessing affected the fairness conclusions.  

**Deliverables:**
* Progress Report 5 (Due May 10) - a minimum 2-page writeup plus a notebook of all the completed tasks above 


### Week 7 (May 11 - 17): TBD
### Week 8 (May 18 - 24): TBD
### Week 9 (May 25 - 31): TBD
### Week 10 (June 1 - 7): TBD

## D. Project 2: Improving Selection Bias with Group DRO

In this project, we will explore how to handle **selection bias** in real-world data, with a focus on **cross-hospital generalizability**. Selection bias arises when the data we observe is not a random sample of the population we care about $^{[1]}$. 

For example, we may want to predict mortality risk across many hospitals in the U.S., but in practice we only have access to EHR data (e.g., eICU) from a subset of hospitals and patient populations. These datasets are often affected by *selection bias*, in that they are non-random subsets of the general population. For example, patients who interact more frequently with the healthcare system are more likely to be observed, and measurement practices differ across hospitals. As a result, the training data may not be representative of the target population, leading to models that fail to generalize to new clinical environments.

A common approach to improving robustness is **Group DRO**, which optimizes worst-case performance over predefined groups (e.g., race, sex, or hospital). However, standard Group DRO assumes that the relevant sources of distribution shift are fully captured by these observed groups. In settings with selection bias, this assumption may fail: the worst-case test environment may not correspond to any observed group in the training data.

The key question this project seeks to answer is: *How can we improve Group DRO-style robustness methods to better account for selection bias and generalization to new (hospital) environments?* We can further break down this project can be broken down into several guiding questions: 

* **Q1:** Does standard Group DRO improve generalization for cross-site (hospital) settings? (this would motivate the "bit-flip")
* **Q2:** How can we improve Group DRO to perform better on new hospital environments? Can we extend Group DRO by defining better groups or objective functions based on selection risk? 

With respect to **Q2**, we may propose to extend Group DRO by incorporating a notion of *selection risk*. You have learned roughly that selection bias is a common source of dataset shift whereby a **selection mechanism** essentially "samples" if a patient will be included in a dataset or not (i.e., imagine a function for Stanford Hospital data that will more likely accept patients sampled from the general U.S. population if they live in the Bay Area, etc.) If we can approximate this selection risk, then we might be able to build a better DRO method. 

One idea is as follows: Define proxy variables for the selection mechanism, such as missing data patterns, hospital / site, simple covariates (e.g., age, severity proxies). Then define groups based on these variables (e.g., clusters or bins). Next, we can estimate a **selection-risk score** $s_g$ for each group: e.g., using a classifier to distinguish training vs. target environments, or using distance / support-based measures. Finally, we can train a model that prioritizes high-risk groups:
$$
\min_\theta \max_g \; s_g \cdot E[\ell(f_\theta(x), y) \mid g]
$$

where $s_g$ captures how poorly group \(g\) is represented under selection.

* **Week 4:** How does model performance change across hospitals? What evidence is there of selection bias? And does regular group DRO help / hinder this performance?
* **Week 5:** Can we construct better groupings using proxy selection variables or selection risk scores? 
* **Week 6:** TODO
* **Week 7:** TODO
* **Week 8-10:** TODO

---
$^{[1]}$: Many different fields have names for this phenomenon. It may also be called *distribution shift*, *data shift*, *dataset bias*, *sample* selection bias, and in some cases, *covariate shift*. In ML, you might see that the field of domain adaptation is relevant for selection bias. The term *selection bias* I am using has its origins in the field of causal inference.



### Weeks 1-2: 
**Onboarding:** 

 - Follow the [instructions provided in section B.](#b-getting-started) to get setup with the code and data.

**Readings:**
- [Sample Selection Bias in Machine Learning for Healthcare](https://dl.acm.org/doi/pdf/10.1145/3761822)
- (News article) [Research suggests Epic Sepsis Model is lacking in predictive power](https://www.healthcareitnews.com/news/research-suggests-epic-sepsis-model-lacking-predictive-power)
- [A Unified Framework on Generalizability of Clinical Prediction Models](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.872720/): Feel free to skim. 



**Additional Readings:** Review papers on the general field of bias in healthcare data. Optional but highly encouraged. 
- [Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6347576/)
- [Unmasking bias in artificial intelligence: a systematic review of bias detection and mitigation strategies in electronic health record-based models](https://pubmed.ncbi.nlm.nih.gov/38520723/)
- [Bias in medical AI: Implications for clinical decision-making](https://pmc.ncbi.nlm.nih.gov/articles/PMC11542778/)
- [Lessons and tips for designing a machine learning study using EHR data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8057454/)


**Notebook:** 
- Walk through the notebook `week1_explore_eicu_data.ipynb` to become familiar with the eICU dataset. 

*For Assignment 1 (Due April 12):*  Please read [Sample Selection Bias in Machine Learning for Healthcare](https://dl.acm.org/doi/pdf/10.1145/3761822) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).

*For Progress Report 1 (Due April 12):* Meet with your project group and submit what you all want to accomplish for Week 3. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.

### Week 3: 
**Onboarding:** 
- By Wednesday, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 
- You will need to generate and save the full dataset using the notebook from Week 2. 

**Additional Readings:** In conjunction with the readings for Assignment 2. 
- [A Review of Domain Adaptation without Target Labels](https://pubmed.ncbi.nlm.nih.gov/31603771/): Feel free to skim the more "math-y" parts, but you should understand the general aim of the equations. 
- [Selection Mechanisms and Their Consequences: Understanding and Addressing Selection Bias](https://www.researchgate.net/publication/343541124_Selection_Mechanisms_and_Their_Consequences_Understanding_and_Addressing_Selection_Bias): **Optional** overview of selection bias in causal inference, for those curious. 

**Video:**
- Watch this quick [video](https://www.youtube.com/watch?v=MvS_wYtT7Yw). 

**Notebook:** 
- Walk through and complete the notebook `proj2/week2_selection_bias.ipynb` to become familiar with how to evaluate and correct for selection bias in our dataset. Note: We will not be requiring you to submit this notebook, but strongly encourage you to go through it. 

*[For Assignment 2](https://web.stanford.edu/class/cs197/assignments/project.html#related-work) (Due April 16):* You will explore related work in this field. Please see the "nearest-neighbor" papers for this project [here](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing). 

*For Progress Report 2 (Due April 19):* Meet with your project group and submit what you all want to accomplish for Week 4. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.



### Week 4 (April 20 - 26): Does standard Group DRO improve cross-hospital generalization?

(Note: The `week3_missingness.ipynb` notebook is now entirely optional / not required.)

**Goal:**  
How does model performance degrade under cross-hospital distribution shift, and does standard Group DRO help? 

**Readings:**
- [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) - read in-depth  
- [The Impact of Multi-Institution Datasets on the Generalizability of Machine Learning Prediction Models in the ICU](https://journals.lww.com/ccmjournal/fulltext/2024/11000/the_impact_of_multi_institution_datasets_on_the.4.aspx) - read in-depth  
- [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434) - skim  
- [Generalization in Clinical Prediction Models: The Blessing and Curse of Measurement Indicator Variables](https://journals.lww.com/ccejournal/fulltext/2021/08000/generalization_in_clinical_prediction_models__the.5.aspx) - skim  
- [Targeted Validation: Validating Clinical Prediction Models in Their Intended Population and Setting](https://diagnprognres.biomedcentral.com/articles/10.1186/s41512-022-00136-8) - skim  

**Tasks:**  

1. Train a baseline mortality prediction model using standard ERM (which is standard training; ERM = empirical risk minimiazation) on one subset of hospital/s and evaluate it on held-out hospitals. Compare performance (AUROC, AUPRC, or even fairness metrics if you want) across internal validation, external hospital, and worst-hospital performance. Plot these results clearly using seaborn. Repeat this whole process at least 5 times to get a sense of generalization trends. 

2. Identify evidence of selection bias / distribution shift across hospitals.**  
   Compare patient and feature distributions across hospitals, including:
   - demographics (e.g., age, sex, ethnicity)  
   - outcome prevalence  
   - hospital characteristics  
   - missingness summaries  
   
   Use statistical tests or visualizations to assess whether hospitals differ meaningfully. Briefly interpret what kinds of selection mechanisms might explain these differences.

<!-- 3. **Test whether standard group definitions capture worst-case failure.**  
   Evaluate baseline model performance across:
   - sex  
   - ethnicity  
   - sex × ethnicity  
   
   Then compare these disparities to cross-hospital disparities. Are the largest failures captured by standard demographic groups, or are hospital-level failures larger?

4. **Implement standard Group DRO under different group definitions.**  
   Train Group DRO models using at least two different choices of groups:
   - demographic groups (e.g., sex, ethnicity, or sex × ethnicity)  
   - hospital groups  
   
   Compare them to ERM. Does standard Group DRO improve:
   - worst-group training performance?  
   - worst-group test performance?  
   - held-out hospital generalization?  

5. **Evaluate alternative proxy group definitions.**  
   Define at least one alternative grouping intended to better reflect the selection mechanism, such as:
   - missingness level (low / medium / high)  
   - hospital × missingness groups  
   - clusters based on missingness + simple covariates  
   
   Evaluate whether these proxy groups reveal larger or more meaningful failures than race / sex alone.

6. **Interpret what standard Group DRO is and is not solving.**  
   Based on your experiments, discuss:
   - whether Group DRO helps with cross-hospital generalization  
   - whether the worst deployment environment seems to be captured by observed groups in training  
   - whether hospital generalization appears to be more a problem of **selection bias / unseen environments** than simple imbalance across known groups  

   Your write-up should end with a clear answer to the week’s central question:  
   **Is standard Group DRO enough for cross-site generalization, or do we need a more selection-aware notion of groups?** -->

**Deliverables:**
* Progress Report 3 (Due April 26) - a minimum 2-page writeup plus a notebook of all the completed tasks above. As a reminder: $\checkmark+$ = 100% indicates you went above and beyond; $\checkmark$ = 95% indicates basic completeness. 
* Introduction (Due April 23) - see website.

### Week 5 (April 27 - May 3): 

**Goal**:

**Tasks**: 


**Deliverables:**
* Progress Report 4 (Due May 3) - a minimum 2-page writeup plus a notebook of all the completed tasks above 
* (Optional) Related Works - based on this project direction, refine your current related works section.... You will need to do this eventually, so might be a good idea to work on that this week. 

### Week 6 (May 4 - 10): 

**Goal**:

**Tasks**:

**Deliverables:**
* Progress Report 5 (Due May 10) - a minimum 2-page writeup plus a notebook of all the completed tasks above 


### Week 7 (May 11 - 17): TBD
### Week 8 (May 18 - 24): TBD
### Week 9 (May 25 - 31): TBD
### Week 10 (June 1 - 7): TBD