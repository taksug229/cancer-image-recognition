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
    - [Base Model](#base-model)
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
Cancer is one of the leading causes of death in the world. The World Health Organization (WHO) [estimates](https://www.who.int/news-room/fact-sheets/detail/cancer) that cancer was responsible for 9.6 million deaths globally in 2018. Globally, about 1 in 6 deaths is due to cancer. 
Studies consistently [show](https://www.who.int/news-room/detail/03-02-2017-early-cancer-diagnosis-saves-lives-cuts-treatment-costs) that early cancer diagnosis saves lives and cuts treatment costs . 

I wanted to be a part of the solution in helping people detect cancer early. So I decided to build a skin cancer recognition model using 23.9K images from the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery). The images were all moles in the skin that were labeled either melanoma or not. 

### AWS Setup
Since I was dealing with large sets of image data, I decided to set up AWS EC2 instances to process and store the images. I decided to set up 2 [t3.2xlarge](https://aws.amazon.com/ec2/instance-types/t3/) server instances with 150 GB of storage. One instance was for building the neural network pipeline and another was for training the pipeline on the image dataset. By having 2 server instances, I could build and run the pipeline simultaneously. 

### Data

#### Web scrapping
I’ve scraped the 23.K images from the [ISIC](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery) website. You can find the code I used for web scrapping [here](0.&#32;Data&#32;extraction&#32;and&#32;cleaning.ipynb).  

After scrapping the images, I ended up having 53 GB worth of images with a [metadata](data/metadata.csv) that had the labeled information. With my AWS setup, it took about 2 hours to download all the images.

#### Data Cleaning
The images had inconsistent sizes so I resized them all to 100 x 100 pixel dimensions. 
I split my training and test set with a 80:20 split and ended up with the below ratio.

There were 249 images (roughly 1% of the entire dataset) that didn’t have labels to indicate if the patient had cancer or not.  So I decided to drop those images and ended up with a total of 23,653 images.  



Insert Before & After 
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