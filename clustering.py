import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def getFirstPartParams(fit_type, temp_array):
	if fit_type == 'exp_lin' or fit_type == 'exp_exp':
		return fit_type + '_first', [temp_array[1] * np.exp(temp_array[2] * temp_array[0]), temp_array[2]]
	elif fit_type == 'lin_lin' or fit_type == 'lin_exp':
		return fit_type + '_first', [temp_array[1] - temp_array[2] * temp_array[0], temp_array[2]]
	else:
		return 'error', []

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
				temp_array.append(float(content[i][j]))
			fit_dict[fit_type].append(temp_array)
			first_part_name, first_part_params = getFirstPartParams(fit_type, temp_array)
			if first_part_name != 'error':
				fit_dict[first_part_name].append(first_part_params)
		else:
			temp_array = []
			for j in range(1, len(content[i]) - 2):
				temp_array.append(float(content[i][j]))
			fit_dict[fit_type] = [temp_array]
			first_part_name, first_part_params = getFirstPartParams(fit_type, temp_array)
			if first_part_name != 'error':
				fit_dict[first_part_name] = [first_part_params]

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
# clust_centers =  kmeans_fit_models['lin'].cluster_centers_
# print clust_centers
# lin_array =  np.array(fit_params['lin'])
# print lin_array
# plt.plot(lin_array[:, 0], lin_array[:, 1], '.')
# plt.plot(clust_centers[:, 0], clust_centers[:, 1], '*', color = 'red')
# plt.show()

# lin_array =  np.array(fit_params['lin'])
# plt.plot(lin_array[:, 0], lin_array[:, 1], '.', color = 'red')
# lin_lin_first_array =  np.array(fit_params['lin_lin_first'])
# plt.plot(lin_lin_first_array[:, 0], lin_lin_first_array[:, 1], '.', color = 'green')
# lin_exp_first_array =  np.array(fit_params['lin_exp_first'])
# plt.plot(lin_exp_first_array[:, 0], lin_exp_first_array[:, 1], '.', color = 'blue')
# plt.show()

exp_array =  np.array(fit_params['exp'])
plt.plot(exp_array[:, 0], exp_array[:, 1], '.', color = 'red')
exp_lin_first_array =  np.array(fit_params['exp_lin_first'])
plt.plot(exp_lin_first_array[:, 0], exp_lin_first_array[:, 1], '.', color = 'green')
exp_exp_first_array =  np.array(fit_params['exp_exp_first'])
plt.plot(exp_exp_first_array[:, 0], exp_exp_first_array[:, 1], '.', color = 'blue')
plt.show()
