import pickle
from feature_extractor import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Loading stored features and filenames
filenames = pickle.load(open('data/filenames.pickle', 'rb'))
feature_list = pickle.load(open('data/features.pickle', 'rb'))

# Loading Query Image and Extracting Features
query_path = 'Query_Images/' + input("Enter FileName with extension: ")
query_feature = FeatureExtractor.extract(query_path)
print("Query Image Feature Length: " + str(len(query_feature)))

# Searching
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
distances, indices = neighbors.kneighbors([query_feature])

# Displaying Results
search_results = []
for i in range(0, 5):
    filePath = filenames[indices[0][i]]
    img = mpimg.imread(filePath)
    search_results.append(img)

query_img = mpimg.imread(query_path)
plt.imshow(query_img)
plt.title("Query Image")
plt.show()

plt.figure(figsize=(20, 10))
columns = 3
for i, image in enumerate(search_results):
    ax = plt.subplot(2, columns, i + 1)
    ax.set_title("\nSimilar Image - " + str(i + 1) + "\nDistance: " + str(float("{0:.2f}".format(distances[0][i]))))
    plt.imshow(image)

