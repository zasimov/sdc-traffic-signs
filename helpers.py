"""
This Module implements data structures and functions to help me
to build Signs Classifier
"""

import collections
import csv
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


class Samples(collections.namedtuple('Samples', ['features', 'labels'])):
    """Samples struct describes a set of feature-label pairs

         - features - train, valid or test features (numpy.ndarray)
         - labels - train, valid or test labels (numpy.ndarray)
    """

    def __len__(self):
        return len(self.features)

    @property
    def labels_set(self):
        """labels_set returns a set of labels
        """
        return frozenset(self.labels)

    @property
    def classes(self):
        return len(self.labels_set)

    def shuffle(self):
        features, labels = shuffle(self.features, self.labels)
        return Samples(features, labels)

    def append(self, samples):
        return Samples(np.append(self.features, samples.features, axis=0),
                       np.append(self.labels, samples.labels, axis=0))

    def batches(self, batch_size):
        for start in range(0, len(self), batch_size):
            features = self.features[start:start+batch_size]
            labels = self.labels[start:start+batch_size]
            yield Samples(features, labels)

    def map(self, fn):
        features = []
        for feature in self.features:
            features.append(fn(feature))
        features = np.array(features)
        return Samples(features, self.labels)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            dataset = {
                'features': self.features,
                'labels': self.labels,
            }
            pickle.dump(dataset, f)


def load_samples(filename):
    """load_samples loads the data set from pickle file

    Pickle file should contain dictionary with following keys

      - features - list of features
      - labels - list of labels

    load_samples returns Samples struct
    """
    with open(filename, 'rb') as infile:
        dataset = pickle.load(infile)

    return Samples(dataset['features'], dataset['labels'])


def unique_examples(samples, n_classes=None):
    """unique_examples returns one image per class

    Returns map: class -> image
    """
    examples = {}
    for i, image in enumerate(samples.features):
        cls_ = samples.labels[i]
        if cls_ in examples:
            continue
        examples[cls_] = image
        if n_classes and len(examples) == n_classes:
            break
    return examples


def slashed(path):
    if not path:
        return os.path.sep
    if path[-1] != os.path.sep:
        return path + os.path.sep
    return path


def show_all_signs(samples, n_classes):
    r = 5
    c = 9

    plt.figure(figsize=(10, 5))

    examples = unique_examples(samples, n_classes)

    for cls_ in range(n_classes):
        plt.subplot(r, c, cls_ + 1)
        image = examples[cls_]
        plt.title(cls_)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(image)

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def load_labels(filename):
    """load_labes loads map label_id => label_name from file

    file should be CSV file with header [ClassId, SignName]

    Returns dictionary label_id (int) => label_name (str)
    """
    labels = {}

    with open(filename, 'r') as f:
        reader = csv.DictReader(f=f)
        for row in reader:
            labels[int(row['ClassId'])] = row['SignName']

    return labels


def show_images(images, c=3):
    r = len(images) // c + 1
    n = 1

    for n, image in enumerate(images):
        plt.subplot(r, c, n + 1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(image)

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def show_softmax_predictions(samples, predictions, label_names, n_classes):
    features = samples.features
    labels = samples.labels

    fig, axies = plt.subplots(nrows=len(features), ncols=2, figsize=(16, 11))
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 5
    margin = 0.05
    ind = np.arange(n_predictions)
    bar_width = (2. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, labels, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]

        axies[image_i][0].imshow(feature)
        axies[image_i][0].set_title(label_names[label_id])
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], bar_width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])

    plt.show()