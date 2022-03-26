import pickle
from feature_extractor import FeatureExtractor
from sklearn.neighbors import NearestNeighbors

filenames = pickle.load(open('data/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('data/features-caltech101-resnet.pickle', 'rb'))

query_features = FeatureExtractor.extract('query.jpg')

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
distances, indices = neighbors.kneighbors([query_features])

for i in range(0,5):
    print(filenames[indices[0][i]])
    print("\n")