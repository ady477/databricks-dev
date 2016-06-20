# Databricks notebook source exported at Mon, 20 Jun 2016 23:33:13 UTC
# MAGIC %md
# MAGIC ## Subcontractor Score
# MAGIC  Author: ** Aditya Tiwari **

# COMMAND ----------

from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split

allData = sqlContext.sql("select * from xak_jd_base where SubQualityRate is not null")
df = allData.toPandas()
columns_name = (df.axes[1])
row_numbers = (df.axes[0])

df = df.fillna(0)
train, test = train_test_split(df, test_size = 0.2)

# COMMAND ----------

features = list(df.columns)
print(features)

# COMMAND ----------

f1 = ['SubContractAmt','SubSafetyRate','SubPunchlistRate','SubChangeOrderRate','SubWorkCompRate','SubResponseRate','SubFieldSupRate','SubManPowerRate','SubManOfficeRate','SubContWorkRate','SubRevTotalRate','SubDuration','SubNumObservers','SubNumTotObs', 'SubNumTotIssues', 'SubNumQAQC', 'SubNumSafe', 'SubNumQAQCHighNComplete', 'SubNumQAQCMediumNComplete', 'SubNumQAQCLowNComplete', 'SubNumQAQCNCompleteIssues', 'SubNumQAQCHighNManuf', 'SubNumQAQCMediumNManuf', 'SubNumQAQCLowNManuf', 'SubNumQAQCNManufIssues', 'SubNumQAQCHighDef', 'SubNumQAQCMediumDef', 'SubNumQAQCLowNDef', 'SubNumQAQCNDefIssues', 'SubNumQAQCHighNConDoc', 'SubNumQAQCMediumNConDoc', 'SubNumQAQCLowNConDoc', 'SubNumQAQCNConDocIssues', 'SubNumQAQCHigh', 'SubNumQAQCMedium', 'SubNumQAQCLow', 'SubNumQAQCIssues', 'SubNumSafeHigh', 'SubNumSafeMedium', 'SubNumSafeLow', 'SubNumSafeIssues', 'SubAvgQAHighComplDays', 'SubAvgQAMedComplDays', 'SubAvgQALowComplDays', 'SubAvgQAComplDays', 'ProjNumObservers', 'ProjNumSubs', 'ProjNumTotObs', 'ProjNumTotIssues', 'ProjNumQAQC', 'ProjNumSafe', 'ProjNumQAQCHighNComplete', 'ProjNumQAQCMediumNComplete', 'ProjNumQAQCLowNComplete', 'ProjNumQAQCNCompleteIssues', 'ProjNumQAQCHighNManuf', 'ProjNumQAQCMediumNManuf', 'ProjNumQAQCLowNManuf', 'ProjNumQAQCNManufIssues', 'ProjNumQAQCHighDef', 'ProjNumQAQCMediumDef', 'ProjNumQAQCLowNDef', 'ProjNumQAQCNDefIssues', 'ProjNumQAQCHighNConDoc', 'ProjNumQAQCMediumNConDoc', 'ProjNumQAQCLowNConDoc', 'ProjNumQAQCNConDocIssues', 'ProjNumQAQCHigh', 'ProjNumQAQCMedium', 'ProjNumQAQCLow', 'ProjNumQAQCIssues', 'ProjNumSafeHigh', 'ProjNumSafeMedium', 'ProjNumSafeLow', 'ProjNumSafeIssues', 'ProjAvgQAHighComplDays', 'ProjAvgQAMedComplDays', 'ProjAvgQALowComplDays', 'ProjAvgQAComplDays', 'ProjDuration', 'ProjContractAmt', 'RatioDuration', 'RatioSeverity', 'RatioAvgDaysToComplete']
#'ContractType', ProjectType,ProjectZip
#f2 = ['ProjectZip']
#df[f2]
print(len(f1))

# COMMAND ----------

#from sklearn.tree import DecisionTreeClassifier
y_train = train["SubQualityRate"]
x_train = train[f1]
y_test = test["SubQualityRate"]
x_test = test[f1]
#dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
#clf = dt.fit(x_train, y_train)

# COMMAND ----------

CLASSIFICATION = 'rf'
################################################CLASSIFICATION#########
if CLASSIFICATION == 'svm':
  from sklearn import svm
  clf = svm.SVC(kernel='linear')
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'mnb':
  from sklearn.naive_bayes import MultinomialNB
  clf = MultinomialNB()
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'bnb':
  from sklearn.naive_bayes import BernoulliNB
  clf = BernoulliNB()
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'gnb':
  from sklearn.naive_bayes import GaussianNB
  clf = GaussianNB()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'rf':
  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'lda':
  #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  #clf = LinearDiscriminantAnalysis()
  from sklearn.lda import LDA
  clf = LDA()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'qda':
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  clf = QuadraticDiscriminantAnalysis()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'dt':
  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier(max_depth=5)
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'knn':
  from sklearn.neighbors import KNeighborsClassifier
  clf = KNeighborsClassifier(n_neighbors= 10)
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'ada':
  from sklearn.ensemble import AdaBoostClassifier
  clf = AdaBoostClassifier()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
print("RMSE: ", mean_squared_error(y_test, result))
from sklearn.metrics import accuracy_score
print("ACCURACY :", accuracy_score(y_test, result))
#Creating +- count
countDiversions = {}
diff = y_test - result
for i in diff:
  if i in countDiversions.keys():
      countDiversions[i] += 1
  else:
      countDiversions[i] = 1
    
print("Diversions : ", countDiversions)

# COMMAND ----------

imp = clf.feature_importances_
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
ax.plot(range(len(imp)),imp)
#plt.xticks(range(len(f1)), f1)
display(fig)

# COMMAND ----------

print(imp)

# COMMAND ----------

featureImp = {}
for i in range(0, len(imp)):
  featureImp[f1[i]] = imp[i]
#print(featureImp)
sortedFeatureNames = sorted(featureImp, key=featureImp.get, reverse=True)
sortedFeatureValues = []
#print(len(sortedFeatureImp))
for w in sortedFeatureImp:
  print(w, "\t : \t", featureImp[w],"\n")
  sortedFeatureValues.append(featureImp[w])

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
ax.plot(range(0, len(featureImp)),sortedFeatureValues)
#plt.xticks(range(len(f1)), sortedFeatureNames, rotation='vertical')
plt.figure(figsize=(120,500))
display(fig)

# COMMAND ----------

