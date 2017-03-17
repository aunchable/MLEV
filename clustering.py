import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def getRawData(fname):

	with open(fname) as f:
		content = f.readlines()
		content = content = [x.split(',') for x in content]

	# Now get dictionary of each type of fit. Each fit
	# includes fit parameters for that fit, exclude fit error
	# and new line character
	fit_dict = {}

	for i in range(len(content)):
		fit_type = content[i][0]
		if fit_type in fit_dict:
			temp_array = []
			for j in range(1, len(content[i]) - 2):
				temp_array.append(content[i][j])
			fit_dict[fit_type].append(temp_array)
		else:
			temp_array = []
			for j in range(1, len(content[i]) - 2):
				temp_array.append(content[i][j])
			fit_dict[fit_type] = [temp_array]

	return fit_dict

# Implement K means clustering on each type of fit:
# Make number of cluster centers proportional to log of number
# of curves in each type of fit

# Get all the k means models
fit_params =  getRawData('best_fit_data.csv')
kmeans_fit_models = {}
for fit_type in fit_params:
	print (fit_type, len(fit_params[fit_type]))
	K_value = int(np.log(len(fit_params[fit_type])) )
	kmeans_fit_models[fit_type] = KMeans(n_clusters=K_value).fit(fit_params[fit_type]  )

print kmeans_fit_models

# Plot for linear data
clust_centers =  kmeans_fit_models['lin'].cluster_centers_
print clust_centers
lin_array =  np.array(fit_params['lin'])
print lin_array
plt.plot(lin_array[:, 0], lin_array[:, 1], '.')
plt.plot(clust_centers[:, 0], clust_centers[:, 1], '*', color = 'red')
plt.show()
