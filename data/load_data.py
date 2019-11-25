import spacy
import pandas as pd
import numpy as np
import torch
import numpy as np
import os
import random


def get_texts(sampled_character_folders, text_vectors, labels, texts, nb_samples=None, shuffle=False):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    texts_labels = [(i, text_vectors[idx]) for i in range(len(sampled_character_folders)) for idx in sampler(list(np.where(np.array(labels) == i)[0]))]
    if shuffle:
        random.shuffle(texts_labels)
    return texts_labels
    
class TextGenerator(object):
    """
    Data Generator capable of generating batches of text.
    """
    def __init__(self, df, num_classes, num_samples_per_class, num_meta_test_classes, num_meta_test_samples_per_class):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            num_meta_test_classes: Number of classes for classification (K-way) at meta-test time
            num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes
        self.dim_input = 768
        self.dim_output = 32
        self.texts = df['text'].tolist()
        self.labels = df['target'].tolist()
        #self.nlp = spacy.load('en_trf_bertbaseuncased_lg')
        class_list = np.unique(np.array(df['target'].tolist()))
        random.seed(1)
        random.shuffle(class_list)
        num_val = 6
        num_train = 8
        self.metatrain_character_folders = class_list[: num_train]
        self.metaval_character_folders = class_list[num_train:num_train + num_val]
        self.metatest_character_folders = class_list[num_train + num_val:]
    def sample_batch(self, batch_type, batch_size, text_vectors, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: meta_train/meta_val/meta_test
            shuffle: randomly shuffle classes or not
            swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "meta_train":
            text_classes = self.metatrain_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "meta_val":
            text_classes = self.metaval_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            text_classes = self.metatest_character_folders
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class
        all_text_batches, all_label_batches = [], []
        for i in range(batch_size):
            sampled_character_folders = random.sample(list(text_classes), num_classes)
            labels_and_texts = get_texts(sampled_character_folders, text_vectors, self.labels, self.texts, nb_samples=num_samples_per_class, shuffle=False)
            labels = [li[0] for li in labels_and_texts]
            texts_ = [li[1] for li in labels_and_texts]
            texts = np.stack(texts_)
            labels = np.array(labels)
            labels = np.reshape(labels, (num_classes, num_samples_per_class))
            labels = np.eye(num_classes)[labels]
            texts = np.reshape(texts, (num_classes, num_samples_per_class, -1))
            batch = np.concatenate([labels, texts], 2)
            if shuffle:
                for p in range(num_samples_per_class):
                    np.random.shuffle(batch[:, p])
            labels = batch[:, :, :num_classes]
            texts = batch[:, :, num_classes:]
            if swap:
                labels = np.swapaxes(labels, 0, 1)
                texts = np.swapaxes(texts, 0, 1)
            all_text_batches.append(texts)
            all_label_batches.append(labels)
        all_text_batches = np.stack(all_text_batches)
        all_label_batches = np.stack(all_label_batches)
        return all_text_batches, all_label_batches