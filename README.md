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

* **Weeks 1: Onboarding.** Everyone must apply for access to the data and will get familiar with real-world data through an interactive notebook
* **Weeks 2-10: Specialized Project Track.** In Weeks 2 and 3, each project will provide its own specialized curriculum through interactive notebooks and suggested readings. In Weeks 4-10, you will choose a specialized project to own for the rest of the quarter, culminating in a final presentation and report.

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

3. Test that you can open the first notebook `week1_explore_eicu_data.ipynb` by launching:
    ```
    jupyter notebook
    ```


## C. Project 1: Fair ML in healthcare

In this project, we will be exploring what it means to develop "fair" machine learning models. As ML is increasingly deployed in real-world settings such as healthcare, it is more important than ever that practitioners think carefully about how these models may lead to unfair outcomes.

The field of algorithmic fairness often focuses on technical definitions of fairness, that is, metrics or constraints that a model should satisfy to be deemed "fair". However, as we will see throughout this project, *fairness is not one-size-fits-all*. What it means for a model to be fair depends heavily on the context: what the model is predicting, who it affects, and how it is used in practice.

Project Goal: In this project, you will examine ML prediction models on the eICU dataset and consider how a "fair" model should be defined and evaluated in each healthcare-inspired setting. Although you will first assess your models using existing fairness metrics (such as demographic parity), your group should explore or propose new metrics as part of your research contribution, either heavily inspired by existing literature or of your own design. During weeks 4 - 10, you will build toward a novel fairness contribution by following a set of suggested experiments to brainstorm new evaluation strategies, metrics, and considerations that are especially relevant in healthcare. These themes are meant to guide your thinking towards a new fairness evaluation metric, but your group may also propose new interventions to train or adjust models in order to improve fairness.

We provide a more thorough description of developing your project under Weeks 7-8; please refer to this section often to guide your experiments in Weeks 1-6. 

### Week 1: 
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

*For Assignment 1:*  Please read [An Empirical Characterization...]((http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/)) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).


### Week 2: 
**Onboarding:** 
- By the end of the week, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 

**Additional Readings:** In conjunction with the readings for Assignment 2. 

- [Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342)
- [Ensuring Fairness in Machine Learning to Advance Health Equity](https://www.acpjournals.org/doi/epdf/10.7326/M18-1990)
- [Algorithmic fairness in computational medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463525/)

**Video:**
- Watch [this video](https://www.youtube.com/watch?v=MzuoWAk9_AQ) from 21:00 to 1:21:00

**Notebook:** 
- Walk through and complete the notebook `proj1/week2_fairness_evaluation.ipynb` to become familiar with algorithmic fairness calculations. Note: We will not be requiring you to submit this notebook, but strongly encourage you to go through it. 

*[For Assignment 2](https://web.stanford.edu/class/cs197/assignments/project.html#related-work):* You will explore related work in this field. Please see the "nearest-neighbor" papers for this project [here](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing). 

*For Progress Report I:* Go through the assignments in `proj1/week2_fairness_evaluation.ipynb`. Either submit your final output of the notebook as a pdf or write up a condensed summary of your findings from the notebook. 

### Week 3: 

**Readings:**
- [Evaluating and mitigating bias in machine learning models for cardiovascular disease prediction](https://www.sciencedirect.com/science/article/pii/S1532046423000151?via%3Dihub)
- [The accuracy, fairness, and limits of predicting recidivism](https://www.science.org/doi/10.1126/sciadv.aao5580): If you've heard of the famous COMPAS fairness example, this is the original paper.  


**Notebook:** 
- Walk through the notebook `proj1/week3_fairness_tradeoffs.ipynb` to become familiar with fairness–performance tradeoffs when you try to invertene on the model to improve fairness. 

**Bonus Notebook:** 
- Feel free to check out the fairness evaluation on EHR data done [in this notebook](https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/fairness-in-AI-systems-student.ipynb).

**Bonus Readings:** 

These works provide deeper insight into why fairness tradeoffs arise. They are more theory-heavy, but helpful if you're curious about the limitations of fairness metrics.
- [On Fairness and Calibration](https://arxiv.org/abs/1709.02012)
- [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807)
- [A comparison of approaches to improve worst-case predictive model performance over patient subpopulations](https://www.nature.com/articles/s41598-022-07167-7)


*For Progress Report II:* Go through the assignments in `proj1/week3_fairness_tradeoffs.ipynb`. Submit a condensed summary of your findings from the notebook for the progress report. 

## Weeks 4-10: 
In the next few weeks, you will expand upon what you have learned so far to propose a novel research project on fairness using the eICU dataset. We will provide a suggested outline for Weeks 4–10 to help you brainstorm new fairness evaluation strategies, metrics, and considerations that are particularly relevant in healthcare settings. For example, in Week 4, we will explore how missing data may itself be an important dimension of fairness that is not captured by standard metrics.

While these weekly themes are meant to guide your thinking, you are not limited to them. You are also welcome to explore alternative directions, including:
- Proposing new interventions to train or adjust models in order to improve fairness. 
- Exploring the advanced but under-developed field of causality-based algorithmic fairness, such as path-specific fairness
- Exploring fair representation learning of tabular or NLP data (note: the eICU dataset does not have free text for its note table)
- Examining the role of using race in prediction tasks; this is similar to the idea of proximal learning whereby ML uses a variable that is only spurriously correlated with the variables we actually want to use 

### Week 4: 

In this week, you will move beyond the standard fairness metrics introduced earlier and explore additional notions of fairness from the literature. Your goal is to select at least three alternative metrics, implement them, and evaluate how they behave in the context of predicting mortality. As you do this, reflect on whether these metrics align with what you believe fairness should mean in this setting. Consider how your conclusions might change if the prediction target Y were different—for example, predicting readmission or length of stay instead of mortality. You should also think carefully about the choice of sensitive attribute A: which attributes are most appropriate for defining fairness in this context, and which are actually available in the dataset?

As one example, you may revisit calibration, which we introduced earlier. A calibrated model produces risk scores that correspond to true outcome probabilities, which is often desirable in clinical settings where decisions are made based on predicted risk. However, you should question whether calibration alone captures what we mean by fairness. Is a well-calibrated model necessarily fair? Conversely, can a model be considered fair under other metrics while still being poorly calibrated? Use your experiments to explore these tensions and discuss your findings with your group.


### Week 5: 

In this week, you will investigate how missing data may act as a source of bias that could drive unfairness in machine learning models trained on EHR data. Missingness in healthcare is often not random, and may reflect underlying structural factors such as access to care, insurance status, or clinical decision-making patterns. As a result, it may encode information about sensitive attributes even when those attributes are not explicitly included in the model.

One direction is to treat a patient’s missingness pattern itself as a sensitive attribute A. For example, patterns of missing lab values may correlate with differences in care or patient populations. You can explore how fairness metrics behave when defined over these missingness-based groupings, and whether this reveals disparities that are not captured by traditional attributes such as race or gender.

You can also examine how modeling choices interact with missingness. For instance, what happens to fairness metrics if you remove patients with high levels of missing data? Why might this change the results, and what are the potential consequences of excluding these patients? More broadly, reflect on whether a model can realistically be considered "fair" if it is trained on data with systematic missingness, and what this implies for fairness evaluation in real-world healthcare settings.

### Week 6:

In this week, you will investigate how fairness behaves under distribution shift. In real-world healthcare settings, models are often deployed in environments that differ from the data they were trained on, such as different hospitals, time periods, or patient populations. Your goal is to explore whether a model that appears fair on one dataset remains fair when evaluated on a different distribution.

You can use hospital-level information - such as `hospital_region` variable - to trigger distribution shift by training a model on one hospital region (or all but one hospital region) and evaluting on the heldout hospital region. Similarly, you could also analyze temporal shift by splitting based on the `hospitaldischargeyear`. Measure both overall performance when trained on the full data and subgroup performance on a biased dataset (i.e., only trained on data from one hospital region). Do the fairness metrics remain stable? Are Consider whether fairness should be evaluated only on the training distribution, or whether robustness to shift should be part of the definition of fairness. Reflect on the implications of deploying a model that is fair in one setting but may be unfair in another.


### Weeks 7-8:

In these weeks, you will begin designing your own notion of fairness. Building on the limitations you have observed in existing metrics, your goal is to propose a new fairness metric or evaluation strategy that better captures what you believe fairness should mean in this healthcare setting. This may involve modifying an existing metric, combining multiple metrics, or introducing a new way of measuring disparities, such as focusing on worst-case subgroup performance, conditioning on clinical context, or incorporating missingness or uncertainty.

You should clearly define your proposed metric, including its mathematical form and intuition, and explain what types of disparities it is intended to capture. Apply your metric to your existing models and compare it to standard fairness metrics. Reflect on when your metric provides additional insight, and when it may fail or produce unintended consequences. This week is less about finding a perfect metric and more about articulating and testing a principled definition of fairness.


### Week 9: 

In this week, you will evaluate your proposed fairness metric on the eICU dataset for the sensitive attributes of your choice and several prediction tasks (such as mortality). Your goal is to test whether your metric meaningfully distinguishes between models and captures important tradeoffs that standard metrics may miss. You should also apply your metric to multiple prediction models (i.e., logistic regression, XGBoost, etc.), including those you have explored in earlier weeks, and compare how models rank under your metric versus existing ones.

You should also examine how your metric behaves under different conditions, such as subgroup stratification, missingness, or distribution shift. Consider whether your metric is stable, interpretable, and useful for decision-making. Reflect on whether optimizing for your metric would lead to better outcomes in practice, and what tradeoffs it introduces. This week should help you refine your proposal and prepare for your final project, where you will present and justify your definition of fairness in a real-world setting.



## D. Project 2: Exploring selection and missing data biases


In this project, we will explore different aspects of selection bias in real-world data and how selection bias affects the generalization of ML models. Selection bias arises when the data we observe is not a random sample of the population we care about $^{[1]}$. For example, we may want to predict mortality risk across all adults in the U.S.. However, EHR data (as in our eICU case) typically only includes patients from certain geographical regions and oversamples for people who frequently interact with the healthcare system. As a result, the data available to us may not be representative of the target population, which can lead to models that do not generalize well.


Project Goal: In this project, you will investigate how different forms of selection bias affect ML prediction models trained on the eICU dataset, with the goal of proposing a strategy to mitigate these biases. We will consider several types of bias, including temporal bias, bias due to missing data, and geographic bias. During Weeks 1–3, you will explore ways to detect and reason about selection bias, and examine how it impacts model performance. During Weeks 4–10, you will focus on developing and evaluating methods to address these biases. We provide a more thorough description for developing your project under Weeks 7-8; please refer to this section often to guide your experiments in Weeks 1-6. 

$^{[1]}$: Many different fields have names for this phenomenon. It may also be called *distribution shift*, *data shift*, *dataset bias*, *sample* selection bias, and in some cases, *covariate shift*. In ML, you might see that the field of domain adaptation is relevant for selection bias. The term *selection bias* I am using has its origins in the field of causal inference.


### Week 1: 
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

*For Assignment 1:*  Please read [Sample Selection Bias in Machine Learning for Healthcare](https://dl.acm.org/doi/pdf/10.1145/3761822) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).


### Week 2: 
**Onboarding:** 
- By the end of the week, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 

**Additional Readings:** In conjunction with the readings for Assignment 2. 
- [A Review of Domain Adaptation without Target Labels](https://pubmed.ncbi.nlm.nih.gov/31603771/): Feel free to skim the more "math-y" parts, but you should understand the general aim of the equations. 
- [Selection Mechanisms and Their Consequences: Understanding and Addressing Selection Bias](https://www.researchgate.net/publication/343541124_Selection_Mechanisms_and_Their_Consequences_Understanding_and_Addressing_Selection_Bias): **Optional** overview of selection bias in causal inference, for those curious. 

**Video:**
- Watch this quick [video](https://www.youtube.com/watch?v=MvS_wYtT7Yw). 

**Notebook:** 
- Walk through and complete the notebook `proj2/week2_selection_bias.ipynb` to become familiar with how to evaluate and correct for selection bias in our dataset. Note: We will not be requiring you to submit this notebook, but strongly encourage you to go through it. 

*[For Assignment 2](https://web.stanford.edu/class/cs197/assignments/project.html#related-work):* You will explore related work in this field. Please see the "nearest-neighbor" papers for this project [here](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing). 

*For Progress Report I:* Go through the assignments in `proj2/week2_selection_bias.ipynb`. Either submit your final output of the notebook as a pdf or write up a condensed summary of your findings from the notebook. 

### Week 3: 

**Readings:**
- [Moving Beyond Medical Statistics: A Systematic Review on Missing Data Handling in Electronic Health Records
](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615160/pdf/hds.0176.pdf)
- [Assessing Missing Data Assumptions in EHR-Based Studies: A Complex and Underappreciated Task](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2776905)
- [Methods for Addressing Missingness in Electronic Health Record Data for Clinical Prediction Models: Comparative Evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12617989/)

**Notebook:** 
- Walk through the notebook `proj2/week3_missingness.ipynb` to explore why missing data may be a concern in ML, and why simple imputation methods may not work. 


*For Progress Report II:* Go through the assignments in `proj2/week3_missingness.ipynb`. Submit a condensed summary of your findings from the notebook for the progress report. 

## Weeks 4-10: 
In the next few weeks, you will expand upon what you have learned so far to propose a novel research project on fairness using the eICU dataset. We will provide a suggested outline for Weeks 4–10 to help you brainstorm new fairness evaluation strategies, metrics, and considerations that are particularly relevant in healthcare settings. For example, in Week 4, we will explore how missing data may itself be an important dimension of fairness that is not captured by standard metrics.

While these weekly themes are meant to guide your thinking, you are not limited to them. You are also welcome to explore alternative directions, including:
- Berkson's bias In the context of eICU data, this bias points out that we are only observing a subset of patients who are sick enough to be admitted to the ICU.



### Week 4: 

In this week, you will explore the limitations of reweighting methods as a solution to selection bias. As we touched on in Week 2's notebook, reweighting approaches, including density ratio estimation, aim to correct for differences between training and target distributions by assigning higher importance to certain samples. However, these methods come with challenges.

One key issue is that the estimated weights can be unstable and high-variance, especially when there is limited overlap between the training and target distributions. In these cases, a small number of samples may receive very large weights. 

You can explore this by visualizing the histogram of $P(S=1 | Y_i, X_i)$ for data in the training set. If there are values near 0 or 1, then the training and testing distribution lack overlap and the weights will be unstable. How does model performance change as weights become more extreme?


Finally, consult the literature and propose at least one alternative approach to address these limitations of reweighting. This could include methods such as domain adaptation or model-based methods. 


### Week 5: 

In this week, you will investigate Berkson’s bias. Berkson’s bias arises when we condition on being in the hospital (or ICU), which can distort the relationship we want to measure (i.e., how covariates predict an outcome). For instance, patient admitted to the ICU tend to have higher disease severity, so our mortality prediction model might overinflate this risk on the non-ICU population. 


Start by choosing a proxy for patient severity (for example, number of lab measurements, number of ICU visits within 24 hours, or length of ICU visits within 24 hours). Then:
- Divide patients into groups based on severity (e.g., low, medium, high)
- Train a model on one group (e.g., high-severity patients)
- Evaluate it on another group (e.g., lower-severity patients)

Compare both performance and calibration across these groups. Does the model trained on sicker patients generalize well to less severe patients? Does it systematically over- or under-predict risk?

 ### Week 6: 
In this week, you will critically examine what assumptions are reasonable when correcting for selection bias in healthcare settings. Specifically, we first consider the scenario where we train a mortality prediction model using eICU data and aim to deploy it across all hospitals in the U.S.

Is this simply a case of covariate shift, or are there other types of dataset shift at play? What factors might differ between the training and deployment populations (e.g., hospital resources, patient demographics, access to care, socioeconomic status)? Are there important variables—such as income or structural determinants of health—that are not captured in the dataset but may influence outcomes?

Next, think carefully about what target data we would realistically have access to at deployment time. Many methods for addressing dataset shift assume access to samples from the target distribution, or even access to both features and outcomes in that distribution. In practice, are these assumptions reasonable?

Discuss with your group:
- Where would we obtain external target data in a real healthcare setting? How might limited or partial access to the target distribution affect the methods we can use?
- Given the solutions you've seen so far, which are reasonable in our setting? 
- Is there a gap in what is realistic and solutions that address that setting? 

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


### Week 9:  

In this week, you will evaluate your proposed method on the eICU dataset across one or more prediction tasks (such as mortality). Your goal is to assess whether your approach improves generalization compared to baseline methods. We recommend looking at generalization across hospital regions or hospitalsids, but you are open to explore other variables such as temporal shift. 

You should:
- Compare your method against at least one standard approach (e.g., no adjustment, imputation, reweighting)  
- Evaluate performance across a training and testing shift (i.e., we train on eICU data from all regions but the South, and evaluate on the South eICU data). 

In addition to overall performance, consider whether your method is:
- Stable (e.g., does it rely on extreme weights or sensitive parameters?)  
- Realistic (does it require access to data we would not have in practice?)  

Reflect on whether your approach meaningfully improves selection bias, and what tradeoffs it introduces. This week should help you refine your method and prepare for your final project, where you will present and justify your approach in a real-world deployment setting.

