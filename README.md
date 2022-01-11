# Portfolio

**First name**: Adrien  
**Last name**: Lucbert  
**Student number**: 21132356  
**Academic year**: 2021-2022  
**Project group**: IMPutation    

## Table of contents

- [Introduction](#introduction)
- [Datacamp certificates](#datacamp-certificates)
- [Reflection on my own contribution to the project](#reflection-on-my-own-contribution-to-the-project)
- [Reflection on my own learning objectives](#reflection-on-my-own-learning-objectives)
- [Reflection on the group project as a whole](#reflection-on-the-group-project-as-a-whole)
- [Subject #1: Research project](#subject-1-research-project)
  - [Task definition](#task-definition)
  - [Evaluation](#evaluation)
  - [Conclusions](#conclusions)
  - [Planning](#planning)
- [Subject #2: Predictive analysis](#subject-2-predictive-analysis)
  - [Selecting a Model](#selecting-a-model)
  - [Configuring a Model](#configuring-a-model)
  - [Training a model](#training-a-model)
  - [Evaluating a model](#evaluating-a-model)
  - [Visualizing the outcome of a model](#visualizing-the-outcome-of-a-model)
- [Subject #3: Domain knowledge](#subject-3-domain-knowledge)
  - [Introduction of the subject field](#introduction-of-the-subject-field)
  - [Literature research](#literature-research)
  - [Explanation of Terminology, jargon and definitions](#explanation-of-terminology-jargon-and-definitions)
- [Subject #4: Data preprocessing](#subject-4-data-preprocessing)
  - [Data exploration](#data-exploration)
  - [Data cleansing](#data-cleansing)
  - [Data preparation](#data-preparation)
  - [Data explanation](#data-explanation)
  - [Data visualization](#data-visualization)
- [Subject #5: Communication](#subject-5-communication)
  - [Presentations ](#presentations)
  - [Writing paper](#writing-paper)
- [Feedback](#feedback)
  - [Feedback from others](#feedback-from-others)
  - [Feedback for others](#feedback-for-others)

## Introduction



[Back to the table of contents](#table-of-contents)

## Datacamp certificates

You can find here below each DataCamp assignments and the corresponding
certificate obtained after completion.

| Course                                              | Certificate                                                                                         |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Introduction to Python                              | [certificate](assets/datacamp/certificates/introduction-to-python.pdf)                              |
| Intermediate Python                                 | [certificate](assets/datacamp/certificates/intermediate-python.pdf)                                 |
| Python Data Science Toolbox (Part 1)                | [certificate](assets/datacamp/certificates/python-data-science-toolbox-part-1.pdf)                  |
| Python Data Science Toolbox (Part 2)                | [certificate](assets/datacamp/certificates/python-data-science-toolbox-part-2.pdf)                  |
| Statistical Thinking in Python (Part 1)             | [certificate](assets/datacamp/certificates/statistical-thinking-in-python-part-1.pdf)               |
| Statistical Thinking in Python (Part 2)             | [certificate](assets/datacamp/certificates/statistical-thinking-in-python-part-2.pdf)               |
| Supervised Learning with scikit-learn	              | [certificate](assets/datacamp/certificates/supervised-learning-with-scikit-learn.pdf)               |
| Linear Classifiers in Python                        | [certificate](assets/datacamp/certificates/linear-classifiers-in-python.pdf)                        |
| Introduction to Data Visualization with Matplotlib  | [certificate](assets/datacamp/certificates/introduction-to-data-visualization-with-matplotlib.pdf)  |
| Model Validation in Python                          | [certificate](assets/datacamp/certificates/model-validation-in-python.pdf)                          |
| Exploratory data analysis in Python               	| [certificate](assets/datacamp/certificates/exploratory-data-analysis-in-python.pdf)                 |
| Cleaning data in Python	                            | [certificate](assets/datacamp/certificates/cleaning-data-in-python.pdf)                             |
| Data Manipulation with pandas                       | [certificate](assets/datacamp/certificates/data-manipulation-with-pandas.pdf)                       |
| Machine Learning for Time Series Data in Python     | [certificate](assets/datacamp/certificates/machine-learning-for-time-series-data-in-python.pdf)     |
| Manipulating Time Series Data in Python             | [certificate](assets/datacamp/certificates/manipulating-time-series-data-in-python.pdf)             |
| Joining Data with pandas                            | [certificate](assets/datacamp/certificates/joining-data-with-pandas.pdf)                            |
| Time Series Analysis in Python                      | [certificate](assets/datacamp/certificates/time-series-analysis-in-python.pdf)                      |

[Back to the table of contents](#table-of-contents)

## Reflection on my own contribution to the project

*Situation:*

*Task:*

*Action:*

*Result:*

*Reflection:*

[Back to the table of contents](#table-of-contents)

## Reflection on my own learning objectives

*Situation:*

*Task:*

*Action:*

*Result:*

*Reflection:*

[Back to the table of contents](#table-of-contents)

## Reflection on the group project as a whole

*Situation:*

*Task:*

*Action:*

*Result:*

*Reflection:*

[Back to the table of contents](#table-of-contents)

## Subject #1: Research project

### Task definition

The IMP project aims to provide guidelines for choosing imputation methods for
Building Management System (BMS) time series data. Indeed, sensors often fail to
either collect or send data, which results in gaps in datasets, that can have an
impact on whatever use is made of this data.

The project owner was the research group Energy in Transition from THUAS, and Mr.
Baldiri Salcedo Rahola from this research group accompanied us through the whole
project. The research group works in collaboration with Factory Zero, a company
which, among other things, produces and monitors energy-neutral houses in the
Netherlands. Factory Zero provided us with the data of around 120 houses over
the year 2019.

In previous work done by the research group and Factory Zero, the method chosen
to impute missing data proved to have an impact on the quality of their results.
Depending on the type of data and the gaps sizes, some methods performed better
than others at predicting trends or exact values, depending on the end-use.
For instance, to analyse a household water consumption, imputing showers at
some point in the day would suffice, even if they are not imputing at the right
time, whereas to accurately predict power usage at a specific point in time,
imputing values accurately would be important.

Considering all these parameters, providing guidelines for each situation would
then make it easier for the research group and Factory Zero to choose the
appropriate imputation method. This led us to the following research questions:

**Main question:**

*Which imputation techniques should be applied for data imputation in building
energy time series data?*

**Sub questions:**

1. *What imputation methods are known for imputing time series data?*
2. *Which imputation techniques are best suited for what gap sizes?*
3. *What imputation techniques are best suited for which types of data?*

### Evaluation

### Conclusions

### Planning 

[Back to the table of contents](#table-of-contents)

## Subject #2: Predictive analysis

### Selecting a model

I started off the project with the simplest imputation strategy possible, in
order to quickly have a base to compare other methods with. When searching for
widely used imputation methods, [I stumbled upon an article](https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779)
which compares a few methods. The simplest ones were imputation using constant
values (zero, mean, median) and interpolation. However using a constant value
didn't seem interesting, so chose [2nd order spline interpolation](https://github.com/thuas-imp-2021/thuas-imp-2021/blob/pipeline/imputers/interpolate.ipynb).

Later in the project, I joined Jesús to try a machine learning method. Baldiri
wanted us to include at least one machine learning method in the research paper,
and according to [several papers](https://www.nature.com/articles/s41598-018-24271-9),
Recurrent Neural Networks seemed to be the state-of-the-art for imputing
multivariate time series with missing values.

### Configuring a model

?

### Training a model

Prevent {over,under}fitting:
- early stopping
- feature selection
- HP optimization

### Evaluating a model

### Visualizing the outcome of a model

[Back to the table of contents](#table-of-contents)

## Subject #3: Domain knowledge

### Introduction of the subject field

### Literature research

### Explanation of Terminology, jargon and definitions

[Back to the table of contents](#table-of-contents)

## Subject #4: Data preprocessing

### Data exploration

### Data cleansing

### Data preparation

### Data explanation

### Data visualization

[Back to the table of contents](#table-of-contents)

## Subject #5: Communication

### Presentations 

| Date       | Implication                                                                   | Presentation                                                                           |
| ---------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 04-10-2021 | Created presentation support, presented slides 7-8, 10-13, answered questions | [internal presentation week 6](assets/presentations/internal-presentation-week6.pdf)   |
| 08-10-2021 | Created presentation support, presented slides 9-12, answered questions       | [external presentation week 6](assets/presentations/external-presentation-week6.pdf)   |
| 10-12-2021 | Created presentation support, presented slide 6, answered questions           | [external presentation week 14](assets/presentations/external-presentation-week14.pdf) |

### Writing paper

RNN part  
reviews + comments  
help generating data and graphs  

[Back to the table of contents](#table-of-contents)

## Feedback

### Feedback from others

### Feedback for others

[Back to the table of contents](#table-of-contents)
