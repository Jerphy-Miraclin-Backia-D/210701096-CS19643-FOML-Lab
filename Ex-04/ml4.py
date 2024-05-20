import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("default.csv")
x=df['balance']
y=df['default']
x=x.values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
logr= LogisticRegression()
logr.fit(x_train,y_train)
logr.predict(x_test)
x1=df['balance']
y1=df['default']
sns.regplot(x=x1,y=y1,data=df, logistic=True,ci=None)

