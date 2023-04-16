# -----------------------------Dataset----------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('weather.csv')
# convert nominal data to numeric
from sklearn import preprocessing

my_label = preprocessing.LabelEncoder()
data['Precip Type'] = my_label.fit_transform(data['Precip Type'])
print(data['Precip Type'].unique())
print(my_label.inverse_transform(data['Precip Type'].unique()))
feature_names = ['Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover',
                 'Pressure (millibars)']  # x variable names
X = data[feature_names]  # setting the col names
y = data['Summary']  # setting the col names
target_names = ['Partly Cloudy',
                'Mostly Cloudy',
                'Overcast',
                'Foggy',
                'Breezy and Mostly Cloudy',
                'Clear',
                'Breezy and Partly Cloudy',
                'Breezy and Overcast',
                'Humid and Mostly Cloudy',
                'Humid and Partly Cloudy',
                'Windy and Foggy',
                'Windy and Overcast',
                'Breezy and Foggy',
                'Windy and Partly Cloudy',
                'Breezy',
                'Dry and Partly Cloudy',
                'Windy and Mostly Cloudy',
                'Dangerously Windy and Partly Cloudy',
                'Dry',
                'Windy',
                'Humid and Overcast',
                'Light Rain',
                'Drizzle',
                'Windy and Dry',
                'Dry and Mostly Cloudy',
                'Breezy and Dry',
                'Rain']  # potential classes
data.head(10)
# print(data[feature_names])

# ----------------------------------Detect Outliers-------------------------------------------
# box and whisker plots
data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
pyplot.show()

# histograms
data.hist()
pyplot.show()

# # Remove Outliers and clean the data
# from scipy import stats
# import numpy as np
# print(123)
# z_scores = stats.zscore(data[feature_names])
# print(123)
# abs_z_scores = np.abs(z_scores)
# print(123)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# print(abs_z_scores)
# clean_df = data[filtered_entries]
# X = clean_df[feature_names]
# y = clean_df['Summary']
# clean_df.head(10)
# box and whisker plots
# print(clean_df)
# clean_df.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
# pyplot.show()
#
# # histograms
# clean_df.hist()
# pyplot.show()

# # Feature Engg
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature min max scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Spot Check Algorithms
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Evaluate predictions
print(accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# ClassificationReport
from yellowbrick.classifier import ClassificationReport

# viz = ClassificationReport(model, classes=np.unique(predictions), support="percent")    #good
viz = ClassificationReport(model, classes=np.unique(predictions), ax=None,
                           cmap='YlOrRd',
                           support=None,
                           encoder=None,
                           is_fitted='auto',
                           force_model=False,
                           colorbar=True,
                           fontsize=None)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

from yellowbrick.classifier import ConfusionMatrix

cmPercent = ConfusionMatrix(
    model, classes=np.unique(predictions),
    percent=True
    # label_encoder={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
)
cmVal = ConfusionMatrix(
    model, classes=np.unique(predictions),
    # label_encoder={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
)
cmPercent.fit(X_train, y_train)
cmPercent.score(X_test, y_test)
cmPercent.show()
cmVal.fit(X_train, y_train)
cmVal.score(X_test, y_test)
cmVal.show()

# ClassPredictionError
from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(model, classes=np.unique(predictions))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# FeatureImportances
from yellowbrick.model_selection import FeatureImportances

visualizer = FeatureImportances(model)
visualizer.fit(X_train, y_train)
visualizer.show()
