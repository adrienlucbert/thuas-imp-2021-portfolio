# Portfolio

**First name**: Adrien  
**Last name**: Lucbert  
**Student number**: 21132356  
**Academic year**: 2021-2022  
**Project group**: IMPutation    

## Table of contents

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

When trying out Recurrent Neural Networks, I started with trying to reproduce
the example from Jeroen's lectures. I recreated it, and played a bit with it,
trying to get a good grasp of how RNNs work, and how each hyper-parameter
influences the model's performance.
Aside from the lecture, I read about different architectures that could be used
for imputing missing data for building energy data, mainly from two papers
([[1]](https://booksc.eu/book/81720800/7b299c) and [[2]](https://www.nature.com/articles/s41598-018-24271-9)).
These papers discussed two promising architectures: a variation of GRU, and
bi-directional LSTM. I tested these two configurations and eventually observed
slight performance improvements on the lecture example dataset.

After these experimentations, Jesús and I moved on to adapting RNN to Factory
Zero's house data. From my experimentations, we chose to use one-directional GRU
architecture.

### Training a model

When training our first models, we faced overfitting after some epochs.

<figure style="max-width:600px;text-align:center;margin:.8em auto">
  <img src="assets/RNN/validation-curve-poor-features.png" alt="Validation curve with poor features"/>
  <figcaption><i>Validation curve that shows overfitting</i></figcaption>
</figure>

> Note that at epoch 120, we reset the model to the state where it's validation
loss is the lowest, before continuing the training a bit further with a smaller
learning rate. This explains the strange shape at the end of this curve. We did
the same in some trainings later.

We then realized that this could be avoided by selecting correlated features to
help predicting the target. Indeed, so far we tried to predict the heat pump's
flow temperature using only the flow temperature and timestamp as features.  

[I plotted a heatmap](https://github.com/thuas-imp-2021/thuas-imp-2021/blob/pipeline/corr.ipynb)
showing the correlation between each field of Factory Zero data to help us
choose the most correlated columns.

<figure style="max-width:600px;text-align:center;margin:.8em auto">
  <img src="assets/RNN/correlation-matrix.png" alt="Correlation matrix of Factory Zero sensors data"/>
  <figcaption><i>Correlation matrix of the Factory Zero sensors data</i></figcaption>
</figure>

After adding the two most correlated columns (`alklimaHeatPump return_temp` and
`energyHeatpump power`), we obtained much better results.

<figure style="max-width:600px;text-align:center;margin:.8em auto">
  <img src="assets/RNN/validation-curve-better-features.png" alt="Validation curve with better features"/>
  <figcaption><i>Validation curve that no longer overfits, thanks to better feature selection</i></figcaption>
</figure>

these results were satisfactory, but we wanted to improve them further, by
tuning the model's hyper-parameters. To do that, I used a technique I learnt
during an internship last summer: hyper-parameters optimization using a genetic
algorithm.  
I used a genetic algorithm very much similar to [this one](https://github.com/thuas-imp-2021/Learning-Lab/blob/main/genetic-algorithm.ipynb),
which generated random set of hyper-parameters and evaluated them using
[this script](sources/GA/rnn-fzero.py). The evaluation fitness was the max
`r2_score` obtained while training with the given hyper-parameters. This way,
I run the genetic algorithm for 40 generations and 21500 indivuals, to improve
our `r2_score` from 0.85 to 0.963 with the following parameters:

| Parameter     | Description                                                               | Bounds     | Optimized value |
| ------------- | ------------------------------------------------------------------------- | ---------- | --------------- |
| `window_size` | size of the input window to feed to the RNN                               | 2-12       | 12              |
| `batch_size`  | number of training samples provided to the trainer in one training pass   | 5-32       | 5               |
| `num_layers`  | number of hidden layers                                                   | 1-5        | 1               |
| `hidden_size` | size of the hidden layers                                                 | 2-100      | 95              |
| `loss`        | loss function to use for the training                                     | MSE, Huber | MSE             |
| `rnn`         | RNN architecture to use                                                   | GRU, LSTM  | GRU             |

### Evaluating a model

For the final paper, we wanted imputation results for different targets. For
that reason, I trained models for each target. Each model uses different features,
according to the correlation matrix. Also, I wanted to evaluate the impact that
adding timestamp or timedelta (difference between each observation timestamp and
the previous observation timestamp) to the features had on the model's performance.

To do this, I trained models on each target using [this script](https://github.com/thuas-imp-2021/thuas-imp-2021/blob/main/rnn-trainer.ipynb),
adding the timetamp alone, the timedelta alone, both, or none. The results are
presented in the table below. As a matter of fact, using none resulted in an
overall slight improvement.

| Source      | Target                        | r2_score       | r2_score       | r2_score                    | r2_score  |
| ----------- | ----------------------------- | -------------- | -------------- | --------------------------- | --------- |
|             |                               | with timestamp | with timedelta | wih timestamp and timedelta | with none |
| KNMI        | Temperature                   | 0.96648        | 0.96844        | 0.9674                      | 0.9689    |
| KNMI        | Relative atmospheric humidity | 0.83019        | 0.86689        | 0.86335                     | 0.86642   |
| KNMI        | Global Radiation              | 0.74355        | 0.90357        | 0.87708                     | 0.90472   |
| FactoryZero | alklimaHeatPump flow_temp     | 0.94248        | 0.40943        | 0.39286                     | 0.94214   |
| FactoryZero | alklimaHeatPump op_mode       | 0.90225        | -0.10337       | -0.06115                    | 0.90194   |
| FactoryZero | smartMeter power              | 0.82267        | 0.42267        | 0.34906                     | 0.83316   |
| FactoryZero | co2sensor co2                 | 0.97538        | -2.27796       | -2.32368                    | 0.97715   |

### Visualizing the outcome of a model

Once the models were trained on training data, it was necessary to verify their
ability to impute multi-step gaps. To do this, I ran the model through our
pipeline, to load the data, create gaps of different sizes, impute them and then
evaluate the results.  
After that, I used scripts written by Albert and Juliën to compare RNN results
to other methods, plotting a specific gap, or calculating the variance error of
imputations using different methods.

<figure style="text-align:center;margin:.8em auto">
  <img src="assets/pipeline/results/method-comparison-temperature-gap-3.png" alt="Method comparison on one gap of the KNMI Temperature target"/>
  <figcaption><i>Method comparison on one gap of the KNMI Temperature target</i></figcaption>
</figure>

<figure style="max-width:600px;text-align:center;margin:.8em auto">
  <img src="assets/pipeline/results/average-variance-error-per-gap-temperature.png" alt="Average variance error per gap on the KNMI Temperature target"/>
  <figcaption><i>Average variance error per gap on the KNMI Temperature target</i></figcaption>
</figure>

The variance error is an indicator of how well the trend of the original dataset
was followed in the imputed data.

> RNN models were trained to minimize the prediction error (using the MSE loss
function), however they also performed well in most cases at predicting the trend.

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

Juliën was mainly responsible for writing the paper, but he asked each one of
us to write the part concerning what we worked on during the project.  
So I wrote [the part about RNN](writings/research-paper-rnn.md), and asked for
Jesús' and Juliën's feedback which were useful to shorten some paragraphs and
better rephrase sentences.

I also gave feedback on the whole paper and took part in many meetings to help
finishing the paper, by helping to draw conclusions from results, or by providing
tables and graphs.  
Some of them were [correlation matrices, correlation values](https://github.com/thuas-imp-2021/thuas-imp-2021/blob/pipeline/corr.ipynb)
and [kurtosis and skewness comparison between datasets](https://github.com/thuas-imp-2021/thuas-imp-2021/blob/pipeline/houses-comparison.ipynb)
to support conclusions drawn in the paper.

[Back to the table of contents](#table-of-contents)

## Feedback

### Feedback from others

### Feedback for others

[Back to the table of contents](#table-of-contents)
