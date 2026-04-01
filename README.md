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

### Week 1: 
**Onboarding:** 

 - Follow the [instructions provided in section B.](#b-getting-started) to get setup with the code and data.

**Readings:**
- [An Empirical Characterization of Fair Machine Learning For Clinical Risk Prediction](http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/): Please read and use for Assignment 1.
- [A brief review on algorithmic fairness](https://link.springer.com/article/10.1007/s44176-022-00006-z): Review paper, feel free to skim parts that are already familiar to you. 


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


**Bonus Readings:** 

These works provide deeper insight into why fairness tradeoffs arise. They are more theory-heavy, but helpful if you're curious about the limitations of fairness metrics.
- [On Fairness and Calibration](https://arxiv.org/abs/1709.02012)
- [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807)
- [A comparison of approaches to improve worst-case predictive model performance over patient subpopulations](https://www.nature.com/articles/s41598-022-07167-7)


*For Progress Report II:* Go through the assignments in `proj1/week3_fairness_tradeoffs.ipynb`. Submit a condensed summary of your findings from the notebook for the progress report. 

## Weeks 4-10: 
In the next few weeks, you will expand upon what you have learned so far to propose a novel research project on fairness using the eICU dataset. We will provide a suggested outline for Weeks 4–10 to help you brainstorm new fairness evaluation strategies, metrics, and considerations that are particularly relevant in healthcare settings. For example, in Week 4, we will explore how missing data may itself be an important dimension of fairness that is not captured by standard metrics.

While these weekly themes are meant to guide your thinking, you are not limited to them. You are also welcome to explore alternative directions, including proposing new interventions to train or adjust models in order to improve fairness. 

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



## Project 2: Exploring selection and missing data biases