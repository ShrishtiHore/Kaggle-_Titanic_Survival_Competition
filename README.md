# Kaggle Titanic Survival Competition

### Code and Resources

Language: Python 3.8
Modules and Libraries: numpy, pandas, matplotlib, os, Tensorflow, time, date
Keywords: data analysis, plots, graphs, data cleaning, data science, kaggle
Dataset:  Cornell Movie-Dialogs Corpus 

Usually a Data Science Problem is solved by these 7 stages :

1. Question or problem definition.
2. Acquire training and testing data.
3. Wrangle, prepare, cleanse the data.
4. Analyze, identify patterns, and explore the data.
5. Model, predict and solve the problem.
6. Visualize, report, and present the problem solving steps and final solution.
5. Supply or submit the results.

Data science solutions workflow solves problem with 7 major goals:

1. Classifying
2. Correlating
3. Converting
4. Completing
5. Correcting
6. Creating
7. Charting

**Step 1:** Import required libraries and modules

**Step 2:** Connect Kaggle with google colab and extract the dataset from Kaggle

**Step 3:** Accquire Data and Analyze by decribing data

**Step 4:** preview the data like 

- Which features are mixed data types? Numerical and alphanumeric?? - Cabin, Ticket
- Which features may contain errrors and typos? - Name
- Which feautures contain blank, null, or empty values ? - Cabin>Age>Embarked features

What is the distribution of numerical feature values across the samples?

This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.

Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224). Survived is a categorical feature with 0 or 1 values. Around 38% samples survived representative of the actual survival rate at 32%. Most passengers (> 75%) did not travel with parents or children. Nearly 30% of the passengers had siblings and/or spouse aboard. Fares varied significantly with few passengers (<1%) paying as high as $512. Few elderly passengers (<1%) within age range 65-80.

**Step 5:** Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
Review Parch distribution using `percentiles=[.75, .8]`
SibSp distribution `[.68, .69]`
Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`


What is the distribution of categorical features?

- Names are unique across the dataset (count=unique=891)
- Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
- Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
- Embarked takes three possible values. S port used by most passengers (top=S)
- Ticket feature has high ratio (22%) of duplicate values (unique=681).

**Step 6:** Analyze by Visualizing Data

Correlating numerical features

**Observations:**

1. Infants (Age <=4) had high survival rate.
2. Oldest passengers (Age = 80) survived.
3. Large number of 15-25 year olds did not survive.
4. Most passengers are in 15-35 age range.

**Decisions:**

1. We should consider Age (our assumption classifying #2) in our model training.
2. Complete the Age feature for null values (completing #1).
3. We should band age groups (creating #3).

**Step 7:** Correlating numerical and ordinal features

**Observations:**

1. Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
2. Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
3. Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
4. Pclass varies in terms of Age distribution of passengers.

**Decisions:**

1. Consider Pclass for model training

**Step 8:** Correlating categorical features

**Observations:**

1. Female passengers had much better survival rate than males. Confirms classifying (#1).
2. Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct 3. correlation between Embarked and Survived.
4. Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
5. Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).

**Decisions:**

1. Add Sex feature to model training.
2. Complete and add Embarked feature to model training.

**Step 9:** Wrangle data

1. This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
2. Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
3. Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

**Step 10:** Creating new feature extracting from existing

**Step 11:** Completing a numerical continuous feature

1. A simple way is to generate random numbers between mean and standard deviation.
2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.

**Step 12:** Create new feature combining existing features

New feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets

**Step 13:** Completing a categorical Feature:

embarked feature S,Q,C values based on port of embarkation. Our training dataset has two missiong values. We will simplify fill these with most common occurance.

**Step 14:** Quick completiing and converting a numeric feature

We can now complete the fare feature for single misssing values in test dataset using mode to get the value that occcurs most frequently for this feature.

Note: we are not creating an inmedtiate new feature or doing any further analysis for correlation to guesss missing feature as we are replacing only a single value.

Also round off decimals the fare to to decimals as it represents currency.

**Step 15**: Model, Predict and Solve:

There are 60+ predictive modelling algorithms to choose from. We must understand the type of models which we can evaluate. Our problem is a classification and regression problem.

We want ot identify relationship between output ie; Survived or Not with other variables or features ie; Gender, Age or Port etc... . We are also performing supervised learning as we are training our models with a given dataset.

With these two criteria- Supervised Learning + Classification & Regression.

Now we can select from the the below models:

Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine

Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution.

We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.

Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
Inversely as Pclass increases, probability of Survived=1 decreases the most.
This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
So is Title as second highest positive correlation.


Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier.

Note: The model generates a confidence score which is higher than Logistics Regression model.

In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

Note: KNN confidence score is better than Logistics Regression but worse than SVM.


In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem.


The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time

This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees

Note: The model confidence score is the highest among models evaluated so far.

The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

Note: The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.

Model Evaluation:

We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.

**Step 15:** Export the output csv file and submit to Kaggle !

**Results**

