from sklearn import datasets
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as plt

digits = datasets.load_digits()



randomized_pca = RandomizedPCA(n_components=2)
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

#pca = PCA(n_components=2)

#reduced_data_rpca = pca.fit_transform(digits.data)

colors = ['black', 'blue', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
	x = reduced_data_rpca[:,0][digits.target==i]
	y = reduced_data_rpca[:,1][digits.target==i]
	plt.scatter(x,y,c=colors[i])

plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Princiapl component')
plt.title("PCA Scatter plot")
plt.show()


	