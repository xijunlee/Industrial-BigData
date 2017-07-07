import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import savefig

file_paths = ['12_feature_processing_result.csv','23_feature_processing_result.csv','29_feature_processing_result.csv']

dfs = []

for file_path in file_paths:
	dfs.append(pd.read_csv(file_path))

attrs0 = dfs[0].columns
attrs1 = dfs[1].columns
attrs2 = dfs[2].columns

for i in xrange(len(attrs0)):
	save_name = attrs0[i][3:] + '.png'
	series0 = dfs[0][attrs0[i]]
	series1 = dfs[1][attrs1[i]]
	series2 = dfs[2][attrs2[i]]
	df = pd.DataFrame({attrs0[i]:series0,
					   attrs1[i]:series1,
					   attrs2[i]:series2,
		})
	df.plot(subplots=True, figsize=(8,8),sharex=False,layout=(3,1))
	savefig(save_name)

'''
file_path = './data/train/23/23_data.csv'
data_df = pd.read_csv(file_path)
time_series = data_pd.time
data_df = data_df.drop(['time'],axis=1)
rename_labels = ['f'+str(i) for i in xrange(1,28)]
#data = normalize(data_pd.values,norm='l2',axis=0)
data_pd.columns = rename_labels

print data_pd.head()

corrmat = data_pd.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,square=True)
plt.show()
'''
'''
pca = PCA(n_components=3)
pca.fit(data)
data = pca.transform(data)
'''
'''
est = KMeans(n_clusters=4)
est.fit(data)
fig = plt.figure(1,figsize=(8,6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
plt.cla()
labels = est.labels_
ax.scatter(data[:,0],data[:,1],data[:,2], c=labels.astype(np.float))
plt.show()


#est = KMeans(n_clusters=4)
#est.fit(data)
#labels = est.labels_

result = pd.DataFrame({'time':time_series,'class':labels})
result.to_csv('result.csv',index=False)
print 'saved result!'
'''
