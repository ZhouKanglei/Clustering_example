print(__doc__)

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn import decomposition 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import seaborn as sns 

import xlrd

def loadData():
	plt.rcParams['font.sans-serif'] = ['Times New Roman']
	plt.rcParams['font.size'] = 14

	data = xlrd.open_workbook(r'data/data.xlsx')
	table = data.sheet_by_name('data_en') 



	cities = []
	X = np.zeros((table.nrows - 1, table.ncols - 1)) 
	for row in range(1, table.nrows):
	    row_value = []
	    cities.append(table.cell_value(row, 0))
	    for col in range(1, table.ncols):
	        X[row - 1, col - 1] = table.cell_value(row, col)

	cities = np.array(cities)

	return X, cities

def problemOne(X, cities, plot):
	silhouette_avg_all = []
	cluster_labels_all = []
	score_all = []
	range_n_clusters = [2, 3, 4, 5, 6]

	subplot = 0
	plt.figure(figsize = [8, 5])
	
	for n_clusters in range_n_clusters:
	    # 3行2列
	    subplot = subplot + 1
	    plt.subplot(2, 3, subplot)
	 
	    # 轮廓系数范围
	    plt.xlim([-0.1, 1])
	    # 在轮廓系数中插入空格(n_clusters+1)*10
	    plt.ylim([0, len(X) + (n_clusters + 1) * 10])

	    # 使用随机种子random_state使程序具备可再生性
	    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
	    cluster_labels = clusterer.fit_predict(X)
	    cluster_labels_all.append(cluster_labels)

	    # 给出了所有样本的平均值, 这提供了一个透视到形成的密度和分离的视角集群
	    silhouette_avg = silhouette_score(X, cluster_labels)

	    silhouette_avg_all.append(silhouette_avg)
	    # 计算每个样本的轮廓分数
	    sample_silhouette_values = silhouette_samples(X, cluster_labels)

	    score = metrics.calinski_harabasz_score(X, cluster_labels)  
	    score_all.append(score)

	    print("For n_clusters =", n_clusters,
	          "The average silhouette_score is :", silhouette_avg, 
	          "The calinski-harabaz score is : ", score)

	    y_lower = 10
	    for i in range(n_clusters):
	        # 汇总样本的轮廓分数并排序
	        ith_cluster_silhouette_values = \
	            sample_silhouette_values[cluster_labels == i]

	        ith_cluster_silhouette_values.sort()

	        size_cluster_i = ith_cluster_silhouette_values.shape[0]
	        y_upper = y_lower + size_cluster_i

	        color = cm.nipy_spectral(float(i) / n_clusters)
	        plt.fill_betweenx(np.arange(y_lower, y_upper),
	                          0, ith_cluster_silhouette_values,
	                          facecolor = color, edgecolor = color, alpha = 0.7)

	        # 标记轮廓系数
	        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

	        # 为下一个图计算新的y下标
	        y_lower = y_upper + 10  

	    plt.title("n_clusters = %d" % n_clusters)
	    plt.xlabel("Silhouette coefficient values\n (%d)"%subplot)
	    plt.ylabel("Cluster label")

	    # 所有值的平均轮廓线的垂直线
	    plt.axvline(x = silhouette_avg, color = "red", linestyle = "--")

	    plt.yticks([])  # 清除y轴坐标刻度
	    plt.xticks([ 0, 0.2, 0.4, 0.6, 0.8, 1])


	subplot = subplot + 1
	plt.subplot(2, 3, subplot)
	err = np.zeros((len(range_n_clusters), 1)) + np.std(np.array(score_all)) 
	plt.errorbar(range_n_clusters, score_all, yerr = err,
		fmt = 'o--', ecolor = 'r', color = 'b', elinewidth = 2, capsize = 4)
	plt.title("Scores error plot")
	plt.xlabel("Cluster No.\n (%d)"%subplot)
	plt.ylabel("Calinski-harabaz scores")

	if plot:
		plt.suptitle("Plots of Problem 1", fontweight = "bold")
		plt.tight_layout(pad = 2.0, w_pad = 0.5, h_pad = 1.0)
		plt.show()

	silhouette_avg_max = max(silhouette_avg_all)
	idx_max = silhouette_avg_all.index(silhouette_avg_max)

	print("Max silhouette score %d = %f, for n_clusters = %d" 
		% (idx_max, silhouette_avg_max, range_n_clusters[idx_max]))

	cluster_labels_max = cluster_labels_all[idx_max]
	for i in range(range_n_clusters[idx_max]):
		
		clusters_cities = cities[cluster_labels_max == i]
		print("Cluster %d: "%i, end = '')
		for j in range(len(clusters_cities)):
			print(clusters_cities[j], end = ", ")
		print()

	return cluster_labels_max

def plot_dendrogram(model, cities):
    # 创建链接矩阵，然后绘制树状图
    # 创建每个节点下的样本计数
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶节点
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    label = cut_tree(linkage_matrix, height = 0.8)

    # 绘制相应的树状图
    dendrogram(linkage_matrix, labels = cities, orientation = "right")


def problemTwo(X, cities, method, metric, height, plot):
    # 设置distance_threshold=0可以确保计算完整的树。
	model = AgglomerativeClustering(linkage = "single", affinity ="euclidean",
		n_clusters = None, distance_threshold = 0)
	model = model.fit(X)

	plt.rcParams['font.size'] = 12

	plt.figure(figsize = [8, 6])
	plt.title('Hierarchical Clustering Dendrogram (%s)'%method)
	# 标出树木图的前三个层次
	Z = linkage(X, method = method, metric = metric)
	dendrogram(Z, labels = cities, orientation = "right")

	# plot_dendrogram(model, cities)
	plt.xlabel("Height")

	if plot:
		plt.show()
		sns.clustermap(X, method = method, metric = metric, figsize = [8, 6])
		plt.show()

	label = cut_tree(Z, height = height)
	label = label.reshape(label.size,)

	PCA(X = X, label = label, cities = cities, method = method, height = height)


def PCA(X, label, cities, method, height):
	#根据两个最大的主成分进行绘图
	#选择方差95%的占比
	pca = decomposition.PCA(n_components = 0.95)    
	pca.fit(X)   #主城分析时每一行是一个输入数据
	result = pca.transform(X)  #计算结果
	plt.figure(figsize = [10, 6])  #新建一张图进行绘制
	plt.rcParams['font.size'] = 14
	n_clusters = len(set(label.tolist()))
	print("When Height = %d, n_clusters = %d." % (height, n_clusters))
	for i in range(result[:,0].size):
		color = cm.nipy_spectral(float(label[i]) / n_clusters)
		plt.plot(result[i, 0], result[i, 1], 
			c = color, marker = 'o', markersize = 10) 
		plt.text(result[i, 0], result[i, 1], cities[i])
	x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0] * 100.0), 2)   #x轴标签字符串
	y_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[1] * 100.0), 2)   #y轴标签字符串
	plt.xlabel(x_label)    #绘制x轴标签
	plt.ylabel(y_label)    #绘制y轴标签
	plt.title('Height = %d (%s)'%(height, method))
	plt.show()

if __name__ == '__main__':
	X, cities = loadData()
	labels = problemOne(X = X, cities = cities, plot = True)
	
	method = ["average", "complete", "ward", "single"]
	for met in method:	
		problemTwo(X = X, cities = cities, method = met, 
			metric = "euclidean", height = 1000,  plot = True)
