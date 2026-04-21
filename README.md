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
- [Project 2: Exploring selection and missing data biases](#d-project-2-exploring-selection-and-missing-data-biases)

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


Consider the “standard” algorithmic fairness framework as defined by three key features:
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
* **Week 7**: How can we extend either A1, A2, or A3 to better handle or incorporate missingness? 
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

## D. Project 2: Exploring selection and missing data biases


In this project, we will explore different aspects of selection bias in real-world data and how selection bias affects the generalization of ML models. Selection bias arises when the data we observe is not a random sample of the population we care about $^{[1]}$. For example, we may want to predict mortality risk across all adults in the U.S.. However, EHR data (as in our eICU case) typically only includes patients from certain geographical regions and oversamples for people who frequently interact with the healthcare system. As a result, the data available to us may not be representative of the target population, which can lead to models that do not generalize well.


Project Goal: In this project, you will investigate how different forms of selection bias affect ML prediction models trained on the eICU dataset, with the goal of proposing a strategy to mitigate these biases. We will consider several types of bias, including temporal bias, bias due to missing data, and geographic bias. During Weeks 1–4, you will explore ways to detect and reason about selection bias, and examine how it impacts model performance. During Weeks 5–10, you will focus on developing and evaluating methods to address these biases. We provide a more thorough description for developing your project under Weeks 7-8; please refer to this section often to guide your experiments in Weeks 1-6. 

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



### Week 4: 

**Readings:**
- [Moving Beyond Medical Statistics: A Systematic Review on Missing Data Handling in Electronic Health Records
](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615160/pdf/hds.0176.pdf)
- [Assessing Missing Data Assumptions in EHR-Based Studies: A Complex and Underappreciated Task](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2776905)
- [Methods for Addressing Missingness in Electronic Health Record Data for Clinical Prediction Models: Comparative Evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12617989/)

**Notebook:** 
- Walk through the notebook `proj2/week3_missingness.ipynb` to explore why missing data may be a concern in ML, and why simple imputation methods may not work. 


*For Progress Report 3 (Due April 26):* Meet with your project group and submit what you all want to accomplish for Week 5. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.



## Weeks 5-10: 
In the next few weeks, you will expand upon what you have learned so far to propose a novel research project on fairness using the eICU dataset. We will provide a suggested outline for Weeks 5–10 to help you brainstorm new fairness evaluation strategies, metrics, and considerations that are particularly relevant in healthcare settings. 
<!-- For example, in Week 4, we will explore how missing data may itself be an important dimension of fairness that is not captured by standard metrics. -->

While these weekly themes are meant to guide your thinking, you are not limited to them. You are also welcome to explore alternative directions, including:
- Berkson's bias. In the context of eICU data, this bias points out that we are only observing a subset of patients who are sick enough to be admitted to the ICU.
- Improving on the weaknesses identified form the paper from Week 2's assigned reading

*Note*: We will no longer list what assignments are due every week. It is up to you to reference the CS197 website for deadlines. 


### Week 5: 

In this week, you will explore the limitations of reweighting methods as a solution to selection bias. As we touched on in Week 2's notebook, reweighting approaches, including density ratio estimation, aim to correct for differences between training and target distributions by assigning higher importance to certain samples. However, these methods come with challenges.

One key issue is that the estimated weights can be unstable and high-variance, especially when there is limited overlap between the training and target distributions. In these cases, a small number of samples may receive very large weights. 

You can explore this by visualizing the histogram of $P(S=1 | Y_i, X_i)$ for data in the training set. If there are values near 0 or 1, then the training and testing distribution lack overlap and the weights will be unstable. How does model performance change as weights become more extreme?


Finally, consult the literature and propose at least one alternative approach to address these limitations of reweighting. This could include methods such as domain adaptation or model-based methods. 


### Week 6: 

In this week, you will investigate Berkson’s bias. Berkson’s bias arises when we condition on being in the hospital (or ICU), which can distort the relationship we want to measure (i.e., how covariates predict an outcome). For instance, patient admitted to the ICU tend to have higher disease severity, so our mortality prediction model might overinflate this risk on the non-ICU population. 


Start by choosing a proxy for patient severity (for example, number of lab measurements, number of ICU visits within 24 hours, or length of ICU visits within 24 hours). Then:
- Divide patients into groups based on severity (e.g., low, medium, high)
- Train a model on one group (e.g., high-severity patients)
- Evaluate it on another group (e.g., lower-severity patients)

Compare both performance and calibration across these groups. Does the model trained on sicker patients generalize well to less severe patients? Does it systematically over- or under-predict risk?

 <!-- ### Week 6:  -->


### Weeks 7–8:  

In these weeks, you will begin designing your own approach to addressing selection bias in healthcare data. Building on the knowledge you have observed in earlier weeks (e.g., reweighting instability, bias induced by missing data, Berkson’s bias, and types of dataset shift), your goal is to propose a method or evaluation strategy that improves how models generalize beyond the observed data.


Your approach may take several forms. For example, you might:
- Modify an existing method (e.g., reweighting, imputation, or recalibration) to make it more stable or realistic  
- Propose a new way of defining or approximating the target population that is more realistic for our healthcare setting  
- Introduce a strategy that explicitly accounts for missingness or partial observability for accurate prediction of some $Y$ 
- Develop a diagnostic to detect when a model is likely to fail under selection bias  

You should clearly define your proposed approach, including both its intuition and how it would be implemented. If applicable, describe any assumptions your method relies on, and how those assumptions relate to the realities of EHR data.

Apply your method to your existing models and compare it to standard approaches (such as no correction, simple imputation, or reweighting). Reflect on when your method provides improvements, and when it may fail or introduce new tradeoffs (e.g., variance, instability, or reliance on untestable assumptions).

These weeks are less about finding a perfect solution and more about developing and testing a principled approach to handling selection bias in a realistic healthcare setting.

At the beginning of Week 7, you and your team should also critically examine what assumptions are reasonable when correcting for selection bias in healthcare settings. Specifically, we first consider the scenario where we train a mortality prediction model using eICU data and aim to deploy it across all hospitals in the U.S.

Is this simply a case of covariate shift, or are there other types of dataset shift at play? What factors might differ between the training and deployment populations (e.g., hospital resources, patient demographics, access to care, socioeconomic status)? Are there important variables—such as income or structural determinants of health—that are not captured in the dataset but may influence outcomes?

Next, think carefully about what target data we would realistically have access to at deployment time. Many methods for addressing dataset shift assume access to samples from the target distribution, or even access to both features and outcomes in that distribution. In practice, are these assumptions reasonable?

Discuss with your group:
- Where would we obtain external target data in a real healthcare setting? How might limited or partial access to the target distribution affect the methods we can use?
- Given the solutions you've seen so far, which are reasonable in our setting? 
- Is there a gap in what is realistic and solutions that address that setting? 

### Week 9-10:  

In this week, you will evaluate your proposed method on the eICU dataset across one or more prediction tasks (such as mortality). Your goal is to assess whether your approach improves generalization compared to baseline methods. We recommend looking at generalization across hospital regions or hospitalsids, but you are open to explore other variables such as temporal shift. 

You should:
- Compare your method against at least one standard approach (e.g., no adjustment, imputation, reweighting)  
- Evaluate performance across a training and testing shift (i.e., we train on eICU data from all regions but the South, and evaluate on the South eICU data). 

In addition to overall performance, consider whether your method is:
- Stable (e.g., does it rely on extreme weights or sensitive parameters?)  
- Realistic (does it require access to data we would not have in practice?)  

Reflect on whether your approach meaningfully improves selection bias, and what tradeoffs it introduces. This week should help you refine your method and prepare for your final project, where you will present and justify your approach in a real-world deployment setting.

