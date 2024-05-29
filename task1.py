import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('homeprices2.csv')


'the null values it the bedrooms data is filled with the median of all the elements in bedrooms'
median_bed = math.floor(df.bedrooms.median())
print('The median of bedroom data is', median_bed)
df.bedrooms = df.bedrooms.fillna(median_bed)

median_bath = math.floor(df.bathrooms.median())
print('The median of bathroom data is ' , median_bath)
df.bathrooms = df.bathrooms.fillna(median_bath)


x=df.drop(['price'],axis=1).values
y=df['price'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

ml = LinearRegression()
ml.fit(x_train,y_train)

y_pred = ml.predict(x_test)

print('The R2 score is:-')
xx = r2_score(y_test,y_pred)
print(xx)

print('the predicted values of few data are as follows:-')
pred_y_df=pd.DataFrame({'ActualValue':y_test,'PredictedValue':y_pred,'Difference':y_test-y_pred})
print(pred_y_df[0:20])


plt.figure(figsize=(15, 10))
plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label='Predicted vs Actual')


max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')


plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Actual vs Predicted', fontsize=18)
plt.legend()


plt.gca().set_facecolor('lightgrey')

plt.show()

