import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
feature_names=iris.feature_names
target_names=iris.target_names
print("Feature names: ",feature_names)
print("Target names: ",target_names)
print("\nFirst 10 rows of X: \n",x[:10])
df=pd.DataFrame(x,columns=feature_names)
print(df.head())
df.plot()
import seaborn as sns
iris=sns.load_dataset('iris')
sns.set()
sns.pairplot(iris,hue='species',height=2)
