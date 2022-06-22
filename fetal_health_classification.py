# * Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
    f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

np.random.seed(0)

# * LOADING DATA
data = pd.read_csv("fetal_health.csv")
data.head()

data.info()

describe = data.describe().T

# * DATA ANALYSIS
# ! firstly, evaluate the target and find out if data is imbalanced or not
colours = ["#37E2D5", "#FBCB0A", "#C70A80"]
sns.countplot(data=data, x="fetal_health", palette=colours)

# Correlation matrix
corrmat = data.corr()
plt.figure(figsize=(15, 15))

cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)
sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)

# Accelerations Vs Fetal Movement by Fetal Health
sns.lmplot(data=data, x="accelerations", y="fetal_movement", palette=colours, hue="fetal_health", legend_out=False)
plt.show()
# Prolongued Decelerations Vs Fetal Movement by Fetal Health
sns.lmplot(data=data, x="prolongued_decelerations", y="fetal_movement", palette=colours, hue="fetal_health",
           legend_out=False)
plt.show()
# Abnormal Short Term Variability Vs Fetal Movement by Fetal Health
sns.lmplot(data=data, x="abnormal_short_term_variability", y="fetal_movement", palette=colours, hue="fetal_health",
           legend_out=False)
plt.show()
# Mean Value Of Long Term Variability Vs Fetal Movement by Fetal Health
sns.lmplot(data=data, x="mean_value_of_long_term_variability", y="fetal_movement", palette=colours, hue="fetal_health",
           legend_out=False)
plt.show()

# columns for detailed visualisation
cols = ['baseline value', 'accelerations', 'fetal_movement',
        'uterine_contractions', 'light_decelerations', 'severe_decelerations',
        'prolongued_decelerations', 'abnormal_short_term_variability',
        'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability']

for i in cols:
    sns.swarmplot(x=data["fetal_health"], y=data[i], color="black", alpha=0.5)
    sns.boxenplot(x=data["fetal_health"], y=data[i], palette=colours)
    plt.show()

shades = ["#f7b2b0", "#c98ea6", "#8f7198", "#50587f", "#003f5c"]
plt.figure(figsize=(20, 10))
sns.boxenplot(data=data, palette=shades)
plt.xticks(rotation=90)
plt.show()
# !The above plot shows the range of our feature attributes. All the features are in different ranges.
# !To fit this in a model we must scale it to the same range.
# !In the model building, we will preprocess the features to do the same.

# * MODEL SELECTION AND BUILDING
X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"]

# Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df = s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)
X_df.describe().T

# looking at the scaled features
plt.figure(figsize=(20, 10))
sns.boxenplot(data=X_df, palette=shades)
plt.xticks(rotation=90)
plt.show()

# spliting test and training sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

algorithms = [
    LogisticRegression(random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    SVC()
]
for a in algorithms:
    a.fit(X_train, y_train)
    pred = a.predict(X_test)
    print(a, accuracy_score(y_test, pred))

for d in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    clf = RandomForestClassifier(max_depth=d, min_samples_leaf=20)  # , criterion = 'entropy'
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("Score", clf.score(X_test, y_test), d)

#?---------------------
RF_model = RandomForestClassifier(max_depth=9, min_samples_leaf=20)
RF_model.fit(X_train, y_train)
#Testing the Model on test set
predictions=RF_model.predict(X_test)
acccuracy= accuracy_score(y_test,predictions)
acccuracy

acccuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average="weighted")
precision = precision_score(y_test, predictions, average="weighted")
f1_score = f1_score(y_test, predictions, average="micro")

print("********* Random Forest Results *********")
print("Accuracy    : ", acccuracy)
print("Recall      : ", recall)
print("Precision   : ", precision)
print("F1 Score    : ", f1_score)

print(classification_report(y_test, predictions))

# confusion matrix
plt.subplots(figsize=(12,8))
cf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap=cmap,annot = True, annot_kws = {'size':15})