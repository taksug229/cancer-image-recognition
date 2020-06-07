# Cancer Image Recognition

![Cover](img/cover.jpg)

## Table of Contents
1. [Introduction](#introduction)
    - [Background](#background)
    - [AWS Setup](#aws-setup)
    - [Data](#data)
        - [Web scrapping](#web-scrapping)
        - [Data Cleaning](#data-cleaning)


2. [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Sample images](#sample-images)
    - [Unbalanced Data](#unbalanced-data)


3. [Pipeline](#pipeline)
    - [Base Model](#baseline-model)
    - [Basic Pipeline](#basic-pipeline)
    - [Transfer Learning](#transfer-learning)


4. [Model Evaluation](#model-evaluation)
    - [ROC AUC Curve](#roc-auc-curve)
    - [Precision Recall Curve](#precision-recall-curve)
    - [Confusion Matrix](#confusion-matrix)
    - [Application](#application)


5. [Future works](#future-works)
    - [Possible ways to improve pipeline](#possible-ways-to-improve-pipeline)
    - [Build App](#build-app)


- [Built with](#built-with)
- [Author](#author)

---

## Introduction

### Background


### AWS Setup

### Data

#### Web scrapping



You can find the code I used for web scrapping [here](1.&#32;Data&#32;generation&#32;(Webscrapping).ipynb).  



#### Data Cleaning


- **Train Set: 2000-2018 Seasons (8780 rows, 33 columns)**
- **Test Set: 2019 Season (530 rows, 33 columns)**

---

## Exploratory Data Analysis

### Sample images


![Insert sample images]()

### Unbalanced Data

![Unbalanced Data](img/unbalanced.png)



---

## Pipeline

I scored my predictions based on root mean squared error (RMSE). 

### Base Model


![Insert Baseline Model Confusion Matrix]()


### Basic Pipeline


![Insert Photo of Model]()

### Transfer Learning

![Insert Photo of VGG 16 Model]()


## Model Evaluation


### ROC AUC Curve

![Insert image]()

### Precision Recall Curve

![Insert image]()

### Confusion Matrix

![Insert image]()

### Application


---

## Future works

### Possible ways to improve pipeline

### Build App


---


## Built With

* **Software Packages:**  [Python](https://www.python.org/),  [Pandas](https://pandas.pydata.org/docs/), [Numpy](https://numpy.org/), [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/), [Scikit-Learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), [Scipy](https://docs.scipy.org/doc/), [Seaborn](https://seaborn.pydata.org/)
* **Prediction Methods:** Gradient Boosting, Random Forest, XGBoost, Ada Boost, Random Forest.
## Author

* **Takeshi Sugiyama** - *Data Scientist*
  * [Linkedin](https://www.linkedin.com/in/takeshi-sugiyama/)
  * [Tableau](https://public.tableau.com/profile/takeshi.sugiyama)