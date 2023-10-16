# Supervised Learning. Classification.

Summary: this project is an introduction to classification problems and related ML algorithms.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Logistic Regression](#logistic-regression) \
        2.1.1. [Formulation](#formulation) \
        2.1.2. [Limitations and strong sides](#limitations-and-strong-sides) \
    2.2. [Naive Bayes](#naive-bayes) \
        2.2.1. [Formulation](#formulation) \
        2.2.2. [Limitations and strong sides](#limitations-and-strong-sides) \
    2.3. [KNN](#knn) \
        2.3.1. [Formulation](#formulation) \
        2.3.2. [Limitations and strong sides](#limitations-and-strong-sides) \
    2.4. [SVM](#svm)
3. [Chapter III. Goal](#chapter-iii-goal) 
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

You have already studied regression problems. We can try to predict values such as: human age, stock prices, 
tomorrow's weather (temperature), etc. But what if we want to predict **categorical** characteristics: 
if the patient has a particular illness, what digits are on the street sign? 
These kinds of problems are called **classification problems**. They are similar to regression ones, 
but have a different flavor.

We will observe 3 kinds of classification models: Logistic Regression 
(which is some sort of alternation of Linear Regression), 
Naive Bayes Classifier (model designed specifically for classification problems) and KNN 
(can be applied to regression and to classification tasks).


## Chapter II. Introduction

Classification problems are similar to regression ones. In classification problems, the output space is a set of **C** 
unordered and **mutually exclusive** labels (classes); we want to invent function *f* such:

$$f: X \rightarrow Y, \text{ where } Y \in \lbrace 1, 2, \dots, C \rbrace, X - \text{ training set}$$
, where _Y__= {1, 2, . . . , C}_, _X_ – training set.

If we have only two classes (if we set $$Y \in \lbrace 0, 1 \rbrace$$) the problem is called **binary classification**. We will focus on binary problems in this lesson.

Real-life examples of binary classification include:
* If a patient has a particular illness.
* If a given transaction is a fraud.
* Predict if a customer would click on an advertisement banner in a mobile app.
* Predict if the price of a particular stock would go up or down in the next few seconds (useful for trading).
* Determination if a client would return credit (probability of default, PD; useful for banking).

From probabilistic point of view binary classification problem could be viewed as estimating probability for given instance x that this instance has label 1:

$$p(y=1|\mathbf{x}, \boldsymbol{\theta}), \mathbf{x} - \text{instance, } \boldsymbol{\theta} - \text{parameters of the model}$$ - parameters of the model 

Binary problems are only one particular case of classification problems 
(if ***C*** is greater than 2 we call that a **multiclass problem**). 
Approaches for binary problems can be used to solve multiclass tasks.  
Solve ***C*** different binary problems in one-vs-all fashion, i.e.: train 
***C*** models where for the **i**’s binary problem you define target as 1 when y = i, and define target as 0 when y ≠ i. 
To make prediction in such case we can simply take *argmax* of ***C*** received scores:

$$\arg \max_{i \in \lbrace 1, 2, \dots, C \rbrace} p(y=i|\mathbf{x}, \boldsymbol{\theta})$$

Sometimes it is more convenient (for solving business problems) to translate regression-like tasks into binary ones: 
predict if a person has higher income than the average income in his/her country or forecast stock’s price movements 
(like in the example above). Sometimes classification problems can be converted into regression problems, when labels 
are ordered. In the definition above we defined a set of labels C to be unordered but sometimes it is convenient 
to solve our task as classification problem even if our labels are ordered.

### Logistic Regression

#### Formulation

Logistic Regression is model of the following form:

$$p(y=i|\mathbf{x}, \boldsymbol{\theta}) = \sigma (a) = \frac{1}{1 + e^{-a}}, \text{where } a = \mathbf{w}^T \mathbf{x} + b \text{ and } \boldsymbol{\theta} = (\mathbf{w}, b)$$

Function $$\sigma$$ is called **sigmoid**. Logistic Regression is equivalent to linear combination of 
features of matrix ***X*** and adding bias term ***b***. Sometimes (when doing the math proofs) we absorb ***b*** and use an additional 
column of ones in ***X*** representing bias inside the ***w*** vector. To compute the output of Logistic Regression for particular 
instance ***x*** we need to multiply each component of ***x*** to the corresponding component of ***w*** and add bias number ***b***; 
after that squash your result with sigmoid function. You can plot the sigmoid function and assure that its output 
is between numbers 0 and 1. “Regression” is only a part of the name of the algorithm, it has nothing to do with the 
regression problem.

Logistic Regression is not the best algorithm in terms of accuracy (on average) but it is an interpretable algorithm: 
it is easy to explain how your model works to any manager or user. 
There is a simple relationship between parameters and data.

How can we decide which parameters $$\boldsymbol{\theta} = (\mathbf{w}, b)$$ to use? 
There is a convenient way to train models – use the maximum likelihood estimation (MLE). 
We want to figure out parameters that maximize the probability of receiving a given dataset ***X*** with ***Y***. 
It can be formalized in the following way:

$$p(y|\mathbf{x}, \boldsymbol{\theta}) = \sigma (a) = \frac{1}{1 + e^{-a}}$$

is equivalent to 

$$p(y|\mathbf{x}, \boldsymbol{\theta}) =\text{Ber}\left(y|\sigma(\mathbf{w}^T \mathbf{x} + b)\right)$$

where Ber is [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).

We take the product of probability density functions for all different instances in our dataset, 
then take the logarithm (this is correct because we believe that all instances are independent). 
Resulting function taken with a negative sign is called **Negative Log Likelihood (NLL)**. 
Using MLE in this context means minimizing NLL. Derivation is given as bonus material.

We can use any gradient-based optimization algorithm to figure out the best parameters for minimizing NLL. 
It is easy to compute the gradient and the hessian for NLL and apply any gradient-based method involving gradient and hessian. 
You have to derive NLL gradient by hand (you will need it to solve programming assignments). 
If you are stuck check out this [book](https://probml.github.io/pml-book/book1.html). One of the best ways for optimizing NLL is **Stochastic Gradient Descent (SGD)** 
but it is possible to use second-order methods like **Newton’s method**. 
Different software realizations propose different methods to use. 
For training on thousands or millions of training samples SGD algorithm is more suitable than 
Newton’s method (do you research and figure out why). 
Stochastic Gradient Descent is a powerful method used not only for learning LR models: 
it is a basic method for optimizing parameters of **Deep Learning** models.

For LR models we usually apply the same preprocessing techniques like in case of Linear Regression models: 
scaling, one-hot-encoding for categorical features, etc. For NaN features you can use imputing with mean/mode value 
and setting an additional feature – flag of NaN or not NaN value in a given feature.

#### Limitations and strong sides

Logistic Regression is an interpretable algorithm: this fact makes it appealing for usage in industries with strong 
regulations (for example banking) or need for 100% explainability (some scientific fields). 
Simple form of the model guarantees explainability but leads to less accurate models. 
Roughly speaking, you trade capacity of the model for its simplicity. Logistic Regression has only

*number-of-numeric-features + number-of-all-categories-in-catl-features + 1*

features so this algorithm is less likely to overfit (than more complicated algorithms).

But how can we improve the baseline logistic model? Logistic Regression is a linear model with only the non-linear 
part being the sigmoid activation function. So we will have a hard time trying to fit Logistic Regression 
to non-linear dependency between ***X*** and ***Y***. Geometrically, Logistic Regression model is a hyperplane in the feature space: 
it is able to separate classes only in relatively easy cases.

However, we can transform our input ***X*** using non-linear transformations (formally, it is applying some non-linear function ***φ*** on ***X***). 
Sometimes we can solve challenging tasks using this particular hack. Check the following picture:

![](misc/images/transform_figure.png)

On the picture below we transform quadratic decision boundary into a linear one by calculating quadratic features:
$`x_1 \rightarrow x_1^2`$ and $`x_2 \rightarrow x_2^2`$

> Source: [Kevin Murphy, Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)

Common versions of this trick include:
* Logarithm of features (particularly features with fat tales).
* Polynomial features.
* Division of features: feature# 1/feature #2.
* **sin(x)** or **cos(x)** for catching periodic components.
* Tree-output-like features: comparisons “is feature #1 is greater than the number **k**”.

Adding features has drawbacks: on average, more features can lead to more overfitting.

Usage of complex features makes your final model less interpretable. However, sometimes we can have a balance 
between explainability and accuracy. Classic example are credit score cards:

![](misc/images/credit_score_cards.png)

> Source: https://www.lancaster.ac.uk/stor-i-student-sites/katie-howgate/2021/02/07/how-to-build-a-credit-scorecard/

Continuous features (like age) are **binned**, it is a non-linear operation but the final model is easy to explain for 
anyone, you just need to look at features and corresponding weights (“scores” on the picture below).

### Naive Bayes

#### Formulation

There is another simple approach for classification problems called **Naive Bayes algorithm** (NB algorithm). The core of NB is the next assumption: we assume that all features are conditionally independent given the class label (that is why it is called Naive). This assumption is too strong to be true in the real world scenarios but often NB classifiers give good results even when NB assumption is violated.

The main idea of the method is to calculate posterior probability using the Bayes formula:

$$\text{posterior} = \frac{\text{prior} \cdot \text{likelihood}}{\text{evidence}}$$

$$p(y=c|\mathbf{x}) = \frac{p(y=c) p(\mathbf{x}|y=c)}{p(\mathbf{x})} = \frac{p(y=c) p(\mathbf{x}|y=c)}{\sum_{c'} p(y=c') p(\mathbf{x}|y=c')}$$

Thus our goal will be to Maximize A Posterior (MAP estimation). But if we take a closer look at the denominator, we can understand that it is constant since the values of the features [latex] - `x_i` are fixed for the dataset. Moreover, when NB assumption holds we can derive probability of receiving feature vector $$\mathbf{x}$$, given that label _y_ for that vector equals class _c_:

$$p(\mathbf{x} | y=c) =\prod_{d=1}^D p \left( x_d | y=c \right)$$

![](RackMultipart20230823-1-8wl8w6_html_c66d13329d78e68c.png)

where $$x_d$$– feature _d_ for instance $$\mathbf{x}$$.

Combining all statements together, we get that our problem of maximizing the Posterior is equivalent to:

$$p(y=c) p(\mathbf{x}|y=c) \rightarrow \max$$

To use this formula for training the classifier we need:

1. to specify the form of probability distributions for prior
2. to specify the form of probability distributions of our features
3. and estimate both of them

As for prior, for classification it is modeled with Bernoulli or Multinomial distribution. The good estimate for the class probability in this case would be the share of this class in the dataset. As for distributions of our features:

- We use [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) for binary features
- We can use Normal distribution for continuous features or binarize them and stick to Bernoulli.
- In case of categorical features, Categorical (or Multinomial) [distribution](https://en.wikipedia.org/wiki/Categorical_distribution) is appropriate.

Again, in the real-life problems distributions of features may be different from Bernoulli/Multinoulli/Normal but, 
like with NB assumption,  sometimes we still are capable of training accurate NB classifiers.

Maximum likelihood estimation (MLE) could be used (again) to calculate optimal parameters for the NB model. 
It could be shown (check section 9.3.2 of [book](https://probml.github.io/pml-book/book1.html)) that learning 
NB classifier involves only calculations with simple counts of features (binary or categorical features case) 
or square deviations from mean (continuous features case). So we don’t have any sophisticated optimization procedure, 
only simple and fast calculations based on training data.

#### Limitations and strong sides

Obviously, NB assumption is a main limitation: NB algorithm has poor quality in most real-life scenarios. 
Simplicity of training NB and native way to handle missing values are main advantages of this algorithm.

### KNN

#### Formulation

Both LR model and NB model have associated **parameters**. We use data to estimate optimal parameters, 
after the training stage we can throw all datasets away. Non-parametric models don’t have a fixed set of parameters: 
the effective number of parameters of the model can grow with the size of the dataset. One example of such a model is KNN.

The idea is simple: to classify a new input ***x***, we find the ***K*** closest examples to ***x*** in the training set 
(using some metric such as L2 or L1 distance), then we average labels of ***K*** received neighbors to receive prediction.

![](misc/images/knn.png)

> Source: [Kevin Murphy, Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)

There are no optimization procedures: we find closest neighbors and store them in memory. 

#### Limitations and strong sides

The main difficulty with KNN is the choice of the distance metric. L1/L2 are suitable variants in case when we have 
homogeneous and continuous data. In other cases distances between samples are not so informative. 
Another problem is the Dimensionality Curse: instances tend to be too far away if the number of features is big. 
Training KNN demands a lot of memory, to perform inference we have to keep all our training data.

However, there are different approaches for solving (or at least partially solving) all these problems: 
there are ways to accelerate training speed, find better distance metrics, handle and query training data. 
KNN today is used as a building block for other ML algorithms (for example, recommender systems).

### SVM

Another model for classification problems – [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (SVMs). SVMs could be linear or non-linear 
(in other words, to have a non-linear kernel). In the linear case SVM tries to separate classes with hyperplane but 
this hyperplane is constructed in a little bit different way (using *margins*). 
Non-linear case involves transforming feature space with *kernels* before constructing the hyperplane to make classes 
linearly separable.

SVM optimization includes optimization with constraints.

Linear SVM has difficulties with non-linear dependencies and non-linear SVMs could be viewed as a particular case 
of Neural Networks, so we won’t describe SVM with details here. [This article](https://en.wikipedia.org/wiki/Support_vector_machine) 
has a good base description of SVMs.

#### Metrics

The most common loss function used in classification problems is logloss. However, this function is not the best 
choice as a quality metric for business applications.

There are some commonly used metrics[^1]:
1. Precision/Recall
2. F1 score
3. Fbeta score
4. AUC PR
5. AUC ROC

AUC metrics are supposed to be calculated for binary problems, Precision/Recall/F1/Fbeta can be generalized 
for multiclass setups. We will stick with binary problems for simplicity.

[^1]:  We still use logloss as minimization for our algorithms when it is appropriate, but make decisions with model based on other metrics.

##### Precision/Recall

Precision and recall are metrics for hard labels: not probabilities, but prediction of the class. Hard labels are received from estimated probabilities using thresholding:

$$\mathbb{I}(y=1|x) \> \text{threshold}$$

_ **I** __(p(y = 1|x) \> threshold)_

where $$\mathbb{I}$$ - indicator function (one or zero, based on underlying statement, one if it is true, otherwise zero).

Hard labels allow us to construct a **confusion matrix**. Definition and the details of the confusion matrix 
are described in this [source](https://en.wikipedia.org/wiki/Confusion_matrix), study it before reading the next paragraph.

![](misc/images/precision.png)

> Source: https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5

Precision and recall are defined as

```math
Precision = {tp \over {tp + fp}} \\
Recall = {tp \over {tp + fn}}
```

From formulas we can derive some interpretations:
* Precision is *what fraction of predicted ones are actually ones*.
* Recall is *what fraction of actual ones are predicted by our model*.

We use combinations of precision and recall to validate the quality of your model: one can maximize 
Recall given a minimal needed level of Precision or vice versa.

In some tasks recall is more important than precision:
* medicine cases when further investigation is safe and cheap (you want to detect as many diseases as you can, FP errors result in limited costs/harm.
* fraud detection
* recommendation of relevant content (online courses, videos, etc.)

In other cases Precision is preferred over Recall:
* medicine: deep investigation or treatment of sickness is too harmful or costly.
* recommendations for VIP-segment clients (you don't want to bother your best clients with non-relevant recommendations).
* Security systems in airports.

##### F1 score

Precision and Recall can be combined with appropriate averaging. Harmonic mean between Precision and Recall is called F1 score.

$`F = 2 \cdot {precision \cdot recall \over {precision + recall}}`$

F1 creates a balance between precision and recall.

##### Fbeta score

If we want to pay more attention to Recall or to Precision, we use Fbeta score, which is generalization of F1 score: 

$`F_{\beta} = (1 + \beta^2) \cdot {precision \cdot recall \over {\beta^2 \cdot precision + recall}}`$

*What value of parameter puts more weight on Recall than on Precision?*

##### AUC PR

When optimizing Fbeta score we still have to search for an optimal threshold. Average Precision (AP) 
could be viewed as a way to aggregate Precision and Recall for all possible thresholds. 
[Sklearn’s documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) 
gives a good overview of AP.

##### AUC ROC

[This source](https://alexanderdyakonov.wordpress.com/2017/07/28/auc-roc-%D0%BF%D0%BB%D0%BE%D1%89%D0%B0%D0%B4%D1%8C-%D0%BF%D0%BE%D0%B4-%D0%BA%D1%80%D0%B8%D0%B2%D0%BE%D0%B9-%D0%BE%D1%88%D0%B8%D0%B1%D0%BE%D0%BA/) 
has a comprehensive explanation of AUC ROC. AUC ROC can be interpreted as probability of the following event:

*[predicted probability for random instance of class 0 < predicted probability for random instance of class 1]*

Note that the baseline value (score of random noise) for AUC ROC metric is 0.5, not 0!

**Gini score** can be viewed as “undirectional” AUC ROC score and can be computed using following formula: *|2\*ROC AUC-1|*

## Chapter III. Goal

The goal of this task is to get a deep understanding of the base models for classification (mainly Logistic Regression, NB and KNN). 

## Chapter IV. Instructions

* This project will only be evaluated by humans. You are free to organize and name
your files as you desire.
* Here and further we use Python 3 as the only correct version of Python.
* For training deep learning algorithms you can try [Google Colab](https://colab.research.google.com). It offers kernels
(Runtime) with GPU for free that is faster than CPU for such tasks.
* The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
* Store the datasets in the subfolder data


## Chapter V. Task

1. Download data from Don’tGetKicked competition. <br/><br/>
2. Design train/validation/test split.
Use “PurchDate” field for splitting, test must be later in time than validation, the same goes for validation and train: train.PurchDate < valid.PurchDate < test.PurchDate.
Use the first 33% of dates for train, last 33% of dates for test, and middle 33% for validation set.
*Don’t use the test dataset until the end!* <br/><br/>
3. Use LabelEncoder or OneHotEncoder from sklearn to preprocess categorical variables. Be careful with data leakage (fit Encoder on train and apply on validation & test). Consider another encoding approach if you meet new categorical values in valid & test (unseen in the training dataset), for example: https://contrib.scikit-learn.org/category_encoders/count.html <br/><br/>
4. Train LogisticRegression, GaussianNB, KNN from sklearn on the training dataset and check the quality of your algorithms on the validation dataset. The dependent variable (IsBadBuy) is binary. Don’t forget to normalize your datasets before training models.
<br/><br/>You must receive at least **0.15 Gini score** (the best of all four). Which algorithm performs better? Why?
<br/><br/>
5. Implement Gini score calculation. You can use 2\*ROC AUC - 1 approach, so you need to implement ROC AUC calculation. Check if your metric approximately equals `abs(2\*sklearn.metrcs.roc_auc_score - 1)`.
<br/><br/>
6. Implement your own versions of LogisticRegression, KNN and NaiveBayes classifiers. For LogisticRegression compute gradients with respect to the loss and use stochastic gradient descent.
Are you able to reproduce results from step 4?
<br/><br/>Guide for this task:
Your model must be represented by class with *fit*, *predict* (*predict_proba* with 0.5 *threshold*), *predict_proba* methods.
For LR moder compute gradient of loss with respect to parameters **w** and parameter **b** in fit function. Use a simple SGD approach for estimating optimal values of parameters.<br/><br/>
7. Try to create non-linear features, for example:
<br/><br/>
fractions: feature1/feature2
groupby features: `df[‘categorical_feature’].map(df.groupby(‘categorical_feature’)[‘continious_feature’].mean())`
<br/><br/>
Add new features into your pipeline, repeat step 4. Did you manage to increase your Gini score (you should!)?
<br/><br/>
8. Detect the best features for the problem using coefficients of the Logistic model. Try to eliminate useless features by hand and by using L1 regularization. Which approach is better in terms of Gini score?
<br/><br/>
9. *Try to apply non-linear variants of SVM, use the RAPIDS library if you have access to GPU. In other cases, use sklearn SVC with a non-linear kernel. If the training process needs too much time or memory try to subsample training data. Are you able to receive a better Gini score (on valid dataset) with this approach?
<br/><br/>
10. Select your best model (algorithm + feature set) and tweak its hyperparameters to increase Gini score on the validation dataset. Which hyperparameters are the most impactful?
<br/><br/>
11. Check Gini scores on all three datasets for your best model: train Gini, valid Gini, test Gini. Can you see any drop in performance when comparing valid quality vs test quality? Is your model overfitted or not? Explain.
<br/><br/>
12. Implement calculation or Recall, Precision, F1 score and AUC PR metrics.
<br/>Compare your algorithms on the test dataset using AUC PR metric.
<br/><br/>
13. Which hard label metric do you prefer for the task of detecting “lemon” cars?

## Chapter VI. Bonus part

* Bonus exercises marked with * in [Chapter V. Task](#chapter-v-task)
* Derivation: Using MLE in this context means minimizing NLL.

💡 [Tap here](https://forms.yandex.ru/cloud/646b472b6938722a69a85da0/) **to leave your feedback on the project**. Pedago Team really tries to make your educational experience better.


