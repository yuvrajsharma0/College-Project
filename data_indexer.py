import os
import pickle
import tqdm as tqdm
from feature_extractor import FeatureExtractor
import sys

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list


# path to the datasets
path = 'dataset'

if len(sys.argv) == 2:
    path = sys.argv[1]

root_dir = path
filenames = sorted(get_file_list(root_dir))
print("Total Files: " + str(len(filenames)))

# Extracting Images features
feature_list = []
for i in tqdm.tqdm(range(len(filenames))):
    feature_list.append(FeatureExtractor.extract(filenames[i]))


# Saving data in pickle file
pickle.dump(feature_list, open('data/features.pickle', 'wb'))
print("Pickle Dump success on location: data/features.pickle")

# Saving Filenames in pickle File
pickle.dump(filenames, open('data/filenames.pickle','wb'))
print("Pickle Dump success on location: data/filenames.pickle")
