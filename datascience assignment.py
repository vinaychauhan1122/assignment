import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('1000_Companies (1).csv')

data.head()
data.tail()
data.info()
data.describe()

data.columns
data.isna().sum()
##########
sns.heatmap(data.loc[:,['R&D Spend','Administration','Marketing Spend','Profit']]
            .corr(),annot=True)
            annot=True)annot=True)
##########
plt.scatter(x=data['R&D Spend'],y=data['Profit'])
sns.scatterplot(x=data['Administration'],y=data['Profit'])
sns.scatterplot(x=data['Marketing Spend'],y=data['Profit'])
sns.scatterplot(x=data['Administration'],y=data['Profit'])
sns.barplot(x=data['State'],y=data['Profit'])

sns.boxplot(x=data['State'],y=data['Profit'])

data['State'].unique()
data['State'].value_counts()
##################

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

data['State']=encoder.fit_transform(data['State'])
###########
#seggregate input output
x=data.drop(["Profit"],axis=1)
y=data["Profit"]
#####################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)
##########
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)

regressor.coef_
regressor.intercept_
################
y_pred=regressor.predict(x_test)
y_pred
#########
from sklearn import metrics
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
metrics.mean_absolute_error(y_test,y_pred)
metrics.r2_score(y_test, y_pred)
