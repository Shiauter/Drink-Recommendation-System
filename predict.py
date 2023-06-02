import csv
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree

def get_data():
    header = None
    names = []
    attributes = []
    with open("data.csv", "r", encoding="utf-8") as file:
        rows = csv.reader(file, delimiter=',')
        for row in rows:
            if header is None:
                header = row
            else:
                names.append(row[0])
                attributes.append(list(map(int, row[1:])))

    header = header[1:]
    return header, names, attributes

def recommend_similar_items(items, method):
    similar_items = []
    
    if method == "KMeans":
        kmeans = KMeans(n_clusters=8, n_init='auto') # 要找這個的值
        kmeans.fit(items)
        labels = kmeans.labels_
        cluster_label = labels[-1]
        cluster_items = np.where(labels == cluster_label)[0]
        similar_items = cluster_items.tolist()
        
    elif method == "DBSCAN":
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan.fit(items)
        labels = dbscan.labels_
        cluster_label = labels[-1]
        cluster_items = np.where(labels == cluster_label)[0]
        similar_items = cluster_items.tolist()
        
    elif method == "NearestNeighbors":
        nn = NearestNeighbors(n_neighbors=len(items))
        nn.fit(items)
        _, indices = nn.kneighbors([items[-1]])
        similar_items = indices.flatten().tolist()
        
    elif method == "KDTree":
        kdtree = KDTree(items)
        _, indices = kdtree.query([items[-1]], k=len(items))
        similar_items = indices.flatten().tolist()
        
    elif method == "BallTree":
        balltree = BallTree(items)
        _, indices = balltree.query([items[-1]], k=len(items))
        similar_items = indices.flatten().tolist()
        
    return similar_items


def recommendation(form_input):
    header, names, attributes = get_data()
    custom_item = list(map(lambda x: 1 if x in form_input else 0, header))
    num_recommendations = 10
    attributes_with_custom_item = np.array(attributes + [custom_item])

    methods = ["NearestNeighbors", "KDTree", "BallTree", "KMeans", "DBSCAN"]
    res = {}
    for i, m in enumerate(methods):
#         print(f"{i + 1}. {m}")
        similar_items = recommend_similar_items(attributes_with_custom_item, m)
        # print(len(similar_items))
#         print(f"Recommendation for custom_item: ")
        res[m] = []
        for idx in similar_items:
            if len(res[m]) >= num_recommendations: break
            if idx < len(names):
#                 print(f"- {names[idx]} ({idx})")
                res[m].append(names[idx])
#         print()
    
    return res