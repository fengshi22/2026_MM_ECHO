import scipy.io
import hdf5storage
import random
import numpy as np
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
import os
import torch
from loadRelation import load_relation_knn


def DatasetLoader(args, load_from_file=True):

    # load dataset
    data_dir = os.path.join("./data/" + args.dataset_name)
    exist_k = os.path.exists(os.path.join(data_dir, "data_" + str(args.knn) + ".pth"))

    if load_from_file and os.path.exists(data_dir) and exist_k:
        print("Loading Existing adjacency matrix.")

        load_data = torch.load(os.path.join(data_dir, "data_" + str(args.knn) + ".pth"), weights_only=False)
        features = load_data["features"]
        labels = load_data["labels"]
        labeled_id = load_data["labeled_id"]
        unlabeled_id = load_data["unlabeled_id"]
        relation_list = load_data["relation_list"]
        adj_list = load_data["adj_list"]
        num_views = len(adj_list)
    else:
        print("Generating adjacency matrix.")
        load_path = args.load_path
        features, labels = loadMatData(os.path.join(load_path, args.dataset_name))
        features = feature_normalization(features)

        labeled_id, unlabeled_id = generate_partition(labels, args.label_ratio, args.seed)

        relation_list, adj_list = load_relation_knn(features, k=args.knn)
        num_views = len(adj_list)

        os.makedirs(data_dir, exist_ok=True)
        save_data = {'features': features,
                     'labels': labels,
                     'labeled_id': labeled_id,
                     'unlabeled_id': unlabeled_id,
                     'relation_list': relation_list,
                     'adj_list': adj_list}
        torch.save(save_data, os.path.join(data_dir, "data_" + str(args.knn) + ".pth"))

        for idx in range(num_views):
            file_path = f'{args.data_dir}/view{idx}_graph.txt'
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'w') as f:
                for edge in relation_list[idx].tolist():
                    f.write(f"{edge[0]} {edge[1]}\n")

    view_features = []
    for v in range(num_views):
        view_features.append(torch.from_numpy(features[0, v]))
        view_features[v] = view_features[v].float().to(args.device)
    labels = torch.from_numpy(labels).float().to(args.device)
    return view_features, labels, labeled_id, unlabeled_id, adj_list, relation_list

def collate_fn(batch):
    batch_views = list(zip(*[b[0] for b in batch]))
    batch_labels = torch.stack([b[1] for b in batch])
    return batch_views, batch_labels

def loadMatData(data_name):
    try:
        data = scipy.io.loadmat(data_name)
    except:
        data = hdf5storage.loadmat(data_name)

    try:
        features = data['X']
        label_list = data['Y']
    except:
        features = data['fea']
        label_list = data['gt']

    label_list = label_list.flatten().astype(np.int64)

    label_list = label_list - 1
    return features, label_list


def feature_normalization(features, normalization_type='normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'minmax_scale':
            features[0][idx] = minmax_scale(fea)
        elif normalization_type == 'maxabs_scale':
            features[0][idx] = maxabs_scale(fea)
        elif normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        elif normalization_type == 'robust_scale':
            features[0][idx] = robust_scale(fea)
        elif normalization_type == 'scale':
            features[0][idx] = scale(fea)
        else:
            print("Please enter a correct normalization type!")
    return features


def generate_partition(labels, ratio=0.1, seed=22):

    each_class_num = count_each_class_num(labels)

    labeled_each_class_num = {label: max(round(count * ratio), 1)
                              for label, count in each_class_num.items()}

    index = list(range(len(labels)))

    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)

    shuffled_labels = labels[index]

    p_labeled, p_unlabeled = [], []
    for idx, label in enumerate(shuffled_labels):
        if labeled_each_class_num[label] > 0:
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
        else:
            p_unlabeled.append(index[idx])

    return p_labeled, p_unlabeled


def count_each_class_num(labels):
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

def load_relation_adj(data_name):
    try:
        data_package = scipy.io.loadmat(data_name)
    except:
        data_package = hdf5storage.loadmat(data_name)
    adj_load = data_package['adj']

    relation_list = []
    adj_list = []
    for adj in adj_load[0]:
        np.fill_diagonal(adj, 0)
        adj = torch.from_numpy(np.array(adj)).float()
        relation_list.append(torch.nonzero(adj, as_tuple=False).cuda())
        adj_list.append(adj)
    return relation_list, adj_list