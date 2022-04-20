import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train  = pd.read_csv('train.csv', parse_dates = ['datetime'])
# train = train.drop(columns = ['casual' , 'registered'])
# train = train.head().to_string()
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train = train.drop(columns = ['casual' , 'registered','datetime'])
# train = train.head().to_string()

cor = train.corr()

mask = np.array(cor)
mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots()
fig.set_size_inches(20,10)
hmap = sns.heatmap(cor, annot=True, square=True, mask=mask)
hmap.set_xticklabels(hmap.get_xticklabels(), rotation=30)
plt.show()