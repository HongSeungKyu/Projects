
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

def measureing(data, y_hap):
    average_score = silhouette_score(data, y_hap)

    print(average_score)
    print('Silhouette Analysis Score:',average_score)
    return average_score

from sklearn.cluster import AgglomerativeClustering

def cluster_hierarchy(data):
    lst = []
    if len(data)==1:
        pass
    for i in range(5):
        try:
            hc = AgglomerativeClustering(n_clusters=i , linkage='average')
            
            y_hc = hc.fit_predict(data)
            
            lst.append(measureing(data,y_hc))
        except:
            pass
    if lst:
        hipher_para = lst.index(max(lst))+2
    else:
        hipher_para = 2
    print("hipher_parameter : ",hipher_para)

    hc = AgglomerativeClustering(n_clusters=hipher_para)
    
    y_hc = hc.fit_predict(data)
    print(3)
    a = y_hc.reshape(-1, 1)

    return a , y_hc

#data = pd.read_csv('/Users/jeong-wonlyeol/Desktop/data.csv')
#data = data.drop(['Unnamed: 0'],axis = 1)

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

data['word_size'] = data['width']/data['word_count']
# ====

slid_lst = data['slid_num'].unique()
print(slid_lst)
lst_result = np.ndarray([])

for i in range(1,6):

    try:
        data_for_test = data[data['slid_num'] == i]
    except KeyError:
        print("end =============")
        break

    
    data_for_train = data_for_test[['left','word_size','height']]

    linked = linkage(data_for_train)
    a, y_hap = cluster_hierarchy(data_for_train)
    
    lst_result = np.append(lst_result, a)
    print(a,y_hap)
    """
    plt.figure(figsize=(10, 7))
    plt.title("paragraph hierarchy ")
    dendrogram(linked,
                orientation='top',
                labels=data_for_test.index.values,
                distance_sort='descending')
    plt.show()
    """
    x1 = data_for_train.values
    #data_for_test.to_csv('/Users/jeong-wonlyeol/Desktop/data_for_test.csv')
    """
    x_0 = x1[x1[:, -1]==0, :]
    x_1 = x1[x1[:, -1]==1, :]
    x_2 = x1[x1[:, -1]==2, :]
    """
    # 시각화
    #plt.scatter(x_0[:, 0], x_0[:, 1], cmap=mglearn.cm3)
    #plt.scatter(x_1[:, 0], x_1[:, 1], cmap=mglearn.cm3)
    #plt.scatter(x_2[:, 0], x_2[:, 1], cmap=mglearn.cm3)
    #plt.legend(['level1', 'level2', 'level3'], loc=2)
    #plt.show()


print("=====================")
print(data.shape)
print(lst_result[1:].reshape(-1,1).shape)
print("=====================")
data['cluster'] = lst_result[1:].reshape(-1,1)

data['cluster'] = data['cluster'].apply(lambda x: int(x))




#data.to_csv('/Users/jeong-wonlyeol/Desktop/data_clustered.csv',encoding="utf-8")
