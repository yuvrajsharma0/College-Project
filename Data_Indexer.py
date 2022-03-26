import os
import pickle
import tqdm as tqdm

from feature_extractor import FeatureExtractor
import tensorflow

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
root_dir = 'caltech101'
filenames = sorted(get_file_list(root_dir))


feature_list = []
for i in tqdm.tqdm(range(len(filenames))):
    feature_list.append(FeatureExtractor.extract(filenames[i]))

pickle.dump(feature_list, open('data/features-caltech101-resnet.pickle', 'wb'))
print("Pickle Dump success on location : data/features-caltech101-resnet.pickle")

pickle.dump(filenames, open('data/filenames-caltech101.pickle','wb'))
print("Pickle Dump success on location : data/filenames-caltech101.pickle")