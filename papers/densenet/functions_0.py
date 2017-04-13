from __future__ import print_function

import os, sys
import warnings
import time

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
import itertools
import h5py
import cPickle as pickle
import timeit
import io
import glob
import pickle

from scipy.stats._continuous_distns import semicircular_gen
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from skimage.filters import threshold_otsu
from skimage.transform import resize, rotate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


def use_least_loaded_gpu():
    cmd = 'nvidia-smi --query-gpu="memory.used" --format=csv'
    gpu_mem_util = os.popen(cmd).read().split("\n")[:-1]
    gpu_mem_util.pop(0)
    gpu_mem_util = [util.split(' ')[0] for util in gpu_mem_util]

    cmd = 'nvidia-smi --query-gpu="utilization.gpu" --format=csv'
    gpu_util = os.popen(cmd).read().split("\n")[:-1]
    gpu_util.pop(0)
    gpu_util = [util.split(' ')[0] for util in gpu_util]

    total_util = [int(i) + int(j) for i, j in zip(gpu_mem_util, gpu_util)]
    least_loaded = np.argmin(total_util)
    os.environ["THEANO_FLAGS"] = "device=cuda2"# + str(least_loaded)


use_least_loaded_gpu()
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, regularize_network_params
from lasagne.layers import *


class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(output_path + "log.txt", "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def zca_whiten(train, test, cache=None):
    """
    Use train set statistics to apply the ZCA whitening transform to
    both train and test sets.
    """
    if cache and os.path.isfile(cache):
        with open(cache, 'rb') as f:
            (meanX, W) = pickle.load(f)
    else:
        meanX, W = _compute_zca_transform(train)
        if cache:
            print("Caching ZCA transform matrix")
            with open(cache, 'wb') as f:
                pickle.dump((meanX, W), f, 2)

        print("Applying ZCA whitening transform")
    train_w = np.dot(train - meanX, W)
    test_w = np.dot(test - meanX, W)

    return train_w, test_w


def _compute_zca_transform(imgs, filter_bias=0.1):
    """
    Compute the zca whitening transform matrix.
    """
    print("Computing ZCA transform matrix")
    meanX = np.mean(imgs, 0)

    covX = np.cov(imgs.T)
    D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))

    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0

    D = D ** -.5

    W = np.dot(E, np.dot(np.diag(D), E.T))

    return meanX, W


def get_hex_color(layer_type):
    """
    Determines the hex color for a layer.
    :parameters:
        - layer_type : string
            Class name of the layer
    :returns:
        - color : string containing a hex color for filling block.
    """
    COLORS = ['#4A88B3', '#98C1DE', '#6CA2C8', '#3173A2', '#17649B',
              '#FFBB60', '#FFDAA9', '#FFC981', '#FCAC41', '#F29416',
              '#C54AAA', '#E698D4', '#D56CBE', '#B72F99', '#B0108D',
              '#75DF54', '#B3F1A0', '#91E875', '#5DD637', '#3FCD12',
              '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000',
              '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF']

    hashed = int(hash(layer_type)) % 4

    if "conv" in layer_type.lower():
        return COLORS[:5][hashed]
    if layer_type in lasagne.layers.pool.__all__:
        return COLORS[5:10][hashed]
    if layer_type in lasagne.layers.recurrent.__all__:
        return COLORS[10:15][hashed]
    if layer_type == 'DropoutLayer':
        return COLORS[20:25][hashed]
    if layer_type == 'batch_norm':
        return COLORS[25:30][hashed]
    else:
        return COLORS[15:20][hashed]


def make_pydot_graph(layers, output_shape=True, verbose=False):
    """
    :parameters:
        - layers : list
            List of the layers, as obtained from lasagne.layers.get_all_layers
        - output_shape: (default `True`)
            If `True`, the output shape of each layer will be displayed.
        - verbose: (default `False`)
            If `True`, layer attributes like filter shape, stride, etc.
            will be displayed.
    :returns:
        - pydot_graph : PyDot object containing the graph
    """
    import pydotplus as pydot
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    pydot_nodes = {}
    pydot_edges = []
    for i, layer in enumerate(layers):
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_type
        color = get_hex_color(layer_type)
        if verbose:
            for attr in ['num_filters', 'num_units', 'ds',
                         'filter_shape', 'stride', 'strides', 'p']:
                if hasattr(layer, attr):
                    if attr == 'p':
                        try:
                            label += '\n{}: {:.1f}'.format(attr, np.float32(getattr(layer, attr).eval()))
                        except:
                            pass
                    else:
                        label += '\n{0}: {1}'.format(attr, getattr(layer, attr))
            if hasattr(layer, 'nonlinearity'):
                try:
                    nonlinearity = layer.nonlinearity.__name__
                except AttributeError:
                    nonlinearity = layer.nonlinearity.__class__.__name__
                label += '\nnonlinearity: {0}'.format(nonlinearity)

        if output_shape:
            label += '\nOutput shape: {0}'.format(layer.output_shape)

        pydot_nodes[key] = pydot.Node(
            key, label=label, shape='record', fillcolor=color, style='filled')

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                pydot_edges.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            pydot_edges.append([repr(layer.input_layer), key])

    for node in pydot_nodes.values():
        pydot_graph.add_node(node)

    for edges in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edges[0]], pydot_nodes[edges[1]]))
    return pydot_graph


def draw_to_file(layers, output_path, list_flag=False, **kwargs):
    """
    Draws a network diagram to a file
    :parameters:
        - layers : list or NeuralNet instance
            List of layers or the neural net to draw.
        - filename : string
            The filename to save output to
        - **kwargs: see docstring of make_pydot_graph for other options
    """
    filename = os.path.join(output_path, "architecture.jpg")
    if not list_flag:
        layers = get_all_layers(layers)
    dot = make_pydot_graph(layers, verbose=True, **kwargs)
    ext = filename[filename.rfind('.') + 1:]
    with io.open(filename, 'wb') as fid:
        fid.write(dot.create(format=ext))


def create_result_dirs(output_path, file_name):
    if not os.path.exists(output_path):
        print('creating log folder')
        os.makedirs(output_path)
        func_file_name = os.path.basename(__file__)
        if func_file_name.split('.')[1] == 'pyc':
            func_file_name = func_file_name[:-1]
        functions_full_path = os.path.join(output_path, func_file_name)
        cmd = 'cp ' + func_file_name + ' "' + functions_full_path + '"'
        os.popen(cmd)
        run_file_full_path = os.path.join(output_path, file_name)
        cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)


def visualize_PCA(X, y, path, file_name):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca_model = PCA(n_components=2)
    colors = y
    num_samples = X.shape[0]
    X = X.reshape((num_samples, X.size / num_samples))
    test_X_2dim = pca_model.fit_transform(X)  # input
    plt.scatter(test_X_2dim[:, 0], test_X_2dim[:, 1], c=colors, marker='.', edgecolors='none')
    plt.axis('off')
    file_path = os.path.join(path, file_name + '_PCA.eps')
    plt.savefig(file_path)
    plt.close()


def visualize_TSNE(X, y, path, file_name):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca_model = PCA(n_components=2)
    tsne_model = TSNE(n_components=2)
    colors = y
    num_samples = X.shape[0]
    X = X.reshape((num_samples, X.size / num_samples))
    test_X_2dim = pca_model.fit_transform(X)
    test_X_2dim = tsne_model.fit_transform(test_X_2dim)  # input
    plt.scatter(test_X_2dim[:, 0], test_X_2dim[:, 1], c=colors, marker='.', edgecolors='none')
    plt.axis('off')
    file_path = os.path.join(path, file_name + '_TSNE.eps')
    plt.savefig(file_path)
    plt.close()


def normalize(X_train, X_test, normalization='[-1, 1]'):
    if normalization == '[-1, 1]':
        X_train = (X_train - np.float32(127.5)) / np.float32(127.5)
    elif normalization == 'mean_std':
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = X_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        X_train = (X_train - mean) / std
        mean = X_test.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = X_test.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        X_test = (X_test - mean) / std
        # num_channels = X_train.shape[1]
        # width = X_train.shape[2]
        # height = X_train.shape[3]
        # X_train = X_train.reshape((-1, num_channels * width * height))
        # X_test = X_test.reshape((-1, num_channels * width * height))
        # X_train, X_test = zca_whiten(X_train, X_test, cache='cifar-10-zca-cache.pkl')
        # X_train = X_train.reshape((-1, num_channels, width, height))
        # X_test = X_test.reshape((-1, num_channels, width, height))
    return np.float32(X_train), np.float32(X_test)


def download_dataset(path, source='https://www.cs.toronto.edu/~kriz/'
                                  'cifar-10-python.tar.gz'):
    """
    Downloads and extracts the dataset, if needed.
    """
    files = ['data_batch_%d' % (i + 1) for i in range(5)] + ['test_batch']
    for fn in files:
        if not os.path.exists(os.path.join(path, 'cifar-10-batches-py', fn)):
            break  # at least one file is missing
    else:
        return  # dataset is already complete

    print("Downloading and extracting %s into %s..." % (source, path))
    if sys.version_info[0] == 2:
        from urllib import urlopen
    else:
        from urllib.request import urlopen
    import tarfile
    if not os.path.exists(path):
        os.makedirs(path)
    u = urlopen(source)
    with tarfile.open(fileobj=u, mode='r|gz') as f:
        f.extractall(path=path)
    u.close()


def load_dataset(dataset_name, h5_flag=True):
    if h5_flag:
        hf = h5py.File('/datasets/' + dataset_name + '/data.h5', 'r')
        X_train = np.asarray(hf.get('data'))
        y_train = np.asarray(hf.get('labels'))
        return np.float32(X_train), np.int32(y_train)
    else:
        if dataset_name == "MNIST-full":
            hf = h5py.File('/datasets/' + dataset_name + '/data.h5', 'r')
            X_train = np.asarray(hf.get('data'))
            y_train = np.asarray(hf.get('labels'))
            return np.float32(X_train), np.int32(y_train)
        if dataset_name == 'CIFAR-10':
            dirpath = '/datasets/cifar10_data/cifar-10-batches-py'
            # load training data
            X, y = [], []
            for path in glob.glob('%s/data_batch_*' % dirpath):
                with open(path, 'rb') as f:
                    batch = pickle.load(f)
                X.append(batch['data'])
                y.append(batch['labels'])
            X = np.concatenate(X) \
                .reshape(-1, 3, 32, 32) \
                .astype(np.float32)
            y = np.concatenate(y).astype(np.int32)
            # split into training and validation sets
            # ii = np.random.permutation(len(X))
            X_train = X
            y_train = y
            # load test set
            path = '%s/test_batch' % dirpath
            with open(path, 'rb') as f:
                batch = pickle.load(f)
            X_test = batch['data'] \
                .reshape(-1, 3, 32, 32) \
                .astype(np.float32)
            y_test = np.array(batch['labels'], dtype=np.int32)
            # normalize to zero mean and unity variance
            # offset = np.mean(X_train, 0)
            # scale = np.std(X_train, 0).clip(min=1)
            # X_train = (X_train - offset) / scale
            # X_valid = (X_valid - offset) / scale
            # X_test = (X_test - offset) / scale
            h5 = h5py.File('/datasets/' + dataset_name + '-train/data.h5', 'w')
            h5.create_dataset('data', data=X_train)
            h5.create_dataset('labels', data=y_train)
            h5 = h5py.File('/datasets/' + dataset_name + '-test/data.h5', 'w')
            h5.create_dataset('data', data=X_test)
            h5.create_dataset('labels', data=y_test)
            return X_train, y_train, X_test, y_test

        if dataset_name == 'CIFAR-100':
            data = np.load('cifar-100-python/train')
            X_train = data['data']
            X_train = X_train.reshape(-1, 3, 32, 32)
            y_train = data['fine_labels']
            data = np.load('cifar-100-python/test')
            X_test = data['data']
            X_test = X_test.reshape(-1, 3, 32, 32)
            y_test = data['fine_labels']

            X_train = np.float32(X_train)
            X_test = np.float32(X_test)
            y_train = np.int32(y_train)
            y_test = np.int32(y_test)
            h5 = h5py.File('/datasets/' + dataset_name + '-train/data.h5', 'w')
            h5.create_dataset('data', data=X_train)
            h5.create_dataset('labels', data=y_train)
            h5 = h5py.File('/datasets/' + dataset_name + '-test/data.h5', 'w')
            h5.create_dataset('data', data=X_test)
            h5.create_dataset('labels', data=y_test)

            return X_train, y_train, X_test, y_test


def balanced_subsample(x, y, subsample_size=10000, shuffle_flag=False):
    if shuffle_flag:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

    class_indices = np.unique(y)

    for i in class_indices:
        idx = np.where(y == i)[0]
        if i == 0:
            return_indices = idx[:subsample_size / len(class_indices)]
            y_test = y[idx[:subsample_size / len(class_indices)]]
            X_test = x[idx[:subsample_size / len(class_indices)]]
            y_train = y[idx[subsample_size / len(class_indices):]]
            X_train = x[idx[subsample_size / len(class_indices):]]
        else:
            return_indices = np.append(return_indices, idx[:subsample_size / len(class_indices)])
            y_test = np.append(y_test, y[idx[:subsample_size / len(class_indices)]])
            X_test = np.append(X_test, x[idx[:subsample_size / len(class_indices)]], axis=0)
            y_train = np.append(y_train, y[idx[subsample_size / len(class_indices):]])
            X_train = np.append(X_train, x[idx[subsample_size / len(class_indices):]], axis=0)

    return np.float32(X_train), np.float32(X_test), np.int32(y_train), np.int32(y_test), return_indices


def pad_data(X, X_test, new_size):
    X_test = np.pad(X_test,
                    pad_width=((0, 0), (0, 0), ((new_size - X_test.shape[2]) / 2, (new_size - X_test.shape[2]) / 2),
                               ((new_size - X_test.shape[2]) / 2, (new_size - X_test.shape[2]) / 2)),
                    mode='constant')
    X = np.pad(X, pad_width=(
        (0, 0), (0, 0), ((new_size - X.shape[2]) / 2, (new_size - X.shape[2]) / 2),
        ((new_size - X.shape[2]) / 2, (new_size - X.shape[2]) / 2)),
               mode='constant')
    return X, X_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], excerpt


def augment_minibatch(inputs, flip=0.5, trans=4):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """

    batchsize, c, h, w = inputs.shape
    if flip:
        coins = np.random.rand(batchsize) < flip
        inputs = [inp[:, :, ::-1] if coin else inp
                  for inp, coin in zip(inputs, coins)]
        if not trans:
            inputs = np.asarray(inputs)
    outputs = inputs
    if trans:
        outputs = np.empty((batchsize, c, h, w), inputs[0].dtype)
        shifts = np.random.randint(-trans, trans, (batchsize, 2))
        for outp, inp, (x, y) in zip(outputs, inputs, shifts):
            if x > 0:
                outp[:, :x] = 0
                outp = outp[:, x:]
                inp = inp[:, :-x]
            elif x < 0:
                outp[:, x:] = 0
                outp = outp[:, :x]
                inp = inp[:, -x:]
            if y > 0:
                outp[:, :, :y] = 0
                outp = outp[:, :, y:]
                inp = inp[:, :, :-y]
            elif y < 0:
                outp[:, :, y:] = 0
                outp = outp[:, :, :y]
                inp = inp[:, :, -y:]
            outp[:] = inp

    return outputs


def build_deep_CAE(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[50, 50],
                   kernel_sizes=[5, 5],
                   strides=[2, 2], paddings=[2, 2],
                   dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], num_labeled_data_ratio=None, hyperparameters=[1, 0.1, 0.1],
                   target_flag='soft',
                   batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                   initialization='He'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    encoder = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            encoder[layer_num] = DropoutLayer(
                InputLayer(shape=(None, feature_map_sizes[layer_num], width, height), input_var=input_var),
                p=dropouts[layer_num])
        else:
            encoder[layer_num] = Conv2DLayer(encoder[layer_num - 1],
                                             num_filters=feature_map_sizes[layer_num],
                                             stride=(strides[layer_num - 1], strides[layer_num - 1]),
                                             filter_size=(
                                                 kernel_sizes[layer_num - 1],
                                                 kernel_sizes[layer_num - 1]),
                                             pad=paddings[layer_num - 1],
                                             nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                             W=init)
            if batch_norm_flag:
                encoder[layer_num] = batch_norm(encoder[layer_num])
            encoder[layer_num] = DropoutLayer(encoder[layer_num], p=dropouts[layer_num])

    decoder = {}
    for layer_num in range(num_layers - 2, -1, -1):
        if layer_num == len(feature_map_sizes) - 2:
            last_layer = encoder[num_layers - 1]
        else:
            last_layer = decoder[layer_num + 1]

        decoder[layer_num] = Deconv2DLayer(last_layer, num_filters=feature_map_sizes[layer_num],
                                           stride=(strides[layer_num], strides[layer_num]),
                                           filter_size=(kernel_sizes[layer_num], kernel_sizes[layer_num]),
                                           crop=paddings[layer_num],
                                           nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                           W=init)

        if batch_norm_flag:
            decoder[layer_num] = batch_norm(decoder[layer_num])

    # Loss
    tar = {}
    rec = {}
    rec_clean = {}
    loss_rec = {}
    loss_rec_clean = {}  # Kamran: Double check this for validation
    for layer_num in range(num_layers - 1):
        # target for first layer is just the input value
        if layer_num == 0:
            tar[layer_num] = input_var
        # target for the rest of the layers is the clean encoder output for that layer
        else:
            tar[layer_num] = get_output(encoder[layer_num], deterministic=True)

        # rec is the decoder output from that layer
        rec[layer_num] = get_output(decoder[layer_num])
        loss_rec[layer_num] = lasagne.objectives.squared_error(rec[layer_num], tar[layer_num])

        rec_clean[layer_num] = get_output(decoder[layer_num], deterministic=True)
        loss_rec_clean[layer_num] = lasagne.objectives.squared_error(rec_clean[layer_num], tar[layer_num])

        # Loss summation
        loss_rec[layer_num] *= hyperparameters[layer_num]
        loss_rec_clean[layer_num] *= hyperparameters[layer_num]

        if layer_num == 0:
            loss_reconstruction = loss_rec[layer_num].mean()
            loss_reconstruction_clean = loss_rec_clean[layer_num].mean()
        else:
            loss_reconstruction += loss_rec[layer_num].mean()
            loss_reconstruction_clean += loss_rec_clean[layer_num].mean()

    # Softmax layer for prediction
    classifier = DenseLayer(encoder[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
        loss_reconstruction *= num_labeled_data_ratio
    loss_classification = loss_classification.mean()

    loss = loss_reconstruction + loss_classification
    params_reconstruction = get_all_params(decoder[0], trainable=True)
    updates_reconstruction = lasagne.updates.adam(loss_reconstruction, params_reconstruction,
                                                  learning_rate=learning_rate)
    params_classification = get_all_params([classifier, decoder[0]], trainable=True)
    updates_classification = lasagne.updates.adam(loss, params_classification, learning_rate=learning_rate)

    ratio = 0
    # update_scale / param_scale
    for param in params_classification:
        ratio += ((param - updates_classification[param]).flatten(ndim=1).norm(2) + 1e-45) / (
        param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params_classification)

    # train & test function
    train_fn_reconstruction = theano.function([input_var], loss_reconstruction, updates=updates_reconstruction)
    if semi_supervised_flag:
        train_fn_joint = theano.function([input_var, target_var, semi_supervised_idx],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn_joint = theano.function([input_var, target_var],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    val_fn = theano.function([input_var], loss_reconstruction_clean)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        network = get_all_layers(decoder[0])
        network.append(classifier)
        draw_to_file(network, output_path, list_flag=True)

    return train_fn_reconstruction, train_fn_joint, val_fn, test_fn, classifier, decoder[0], ratio_fn


def build_dense_deep_CAE_old(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[50, 50],
                             kernel_sizes=[5, 5],
                             strides=[2, 2], paddings=[2, 2],
                             dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], num_labeled_data_ratio=None,
                             hyperparameters=[1, 0.1, 0.1], target_flag='soft',
                             batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                             initialization='He'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    encoder = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            encoder[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                            input_var=input_var)
        else:
            intermediate_layers = []
            for input_num in range(layer_num):
                if kernel_sizes[layer_num - input_num - 1][input_num] != 0:
                    encoder[str(layer_num) + '_' + str(input_num)] = Conv2DLayer(encoder[input_num],
                                                                                 num_filters=feature_map_sizes[
                                                                                     layer_num],
                                                                                 stride=(
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 filter_size=(
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 pad=
                                                                                 paddings[layer_num - input_num - 1][
                                                                                     input_num],
                                                                                 nonlinearity=lasagne.nonlinearities.linear,
                                                                                 W=init)
                    intermediate_layers.append(encoder[str(layer_num) + '_' + str(input_num)])
            encoder[layer_num] = ElemwiseSumLayer(intermediate_layers)
            if batch_norm_flag:
                encoder[layer_num] = batch_norm(encoder[layer_num])
            encoder[layer_num] = NonlinearityLayer(encoder[layer_num],
                                                   nonlinearity=lasagne.nonlinearities.leaky_rectify)
        encoder[layer_num] = DropoutLayer(encoder[layer_num], p=dropouts[layer_num])

    decoder = {}
    for layer_num in range(num_layers - 2, -1, -1):
        intermediate_layers = []
        for input_num in range(num_layers - 1, layer_num, -1):
            if input_num == num_layers - 1:
                last_layer = encoder[num_layers - 1]
            else:
                last_layer = decoder[input_num]
            if kernel_sizes[input_num - layer_num - 1][layer_num] != 0:
                decoder[str(layer_num) + '_' + str(input_num)] = Deconv2DLayer(last_layer,
                                                                               num_filters=feature_map_sizes[
                                                                                   layer_num],
                                                                               stride=(
                                                                                   strides[input_num - layer_num - 1][
                                                                                       layer_num],
                                                                                   strides[input_num - layer_num - 1][
                                                                                       layer_num]),
                                                                               filter_size=(
                                                                                   kernel_sizes[
                                                                                       input_num - layer_num - 1][
                                                                                       layer_num],
                                                                                   kernel_sizes[
                                                                                       input_num - layer_num - 1][
                                                                                       layer_num]),
                                                                               crop=paddings[input_num - layer_num - 1][
                                                                                   layer_num],
                                                                               nonlinearity=lasagne.nonlinearities.linear,
                                                                               W=init)
            intermediate_layers.append(decoder[str(layer_num) + '_' + str(input_num)])
        decoder[layer_num] = ElemwiseSumLayer(intermediate_layers)

        if batch_norm_flag:
            decoder[layer_num] = batch_norm(decoder[layer_num])
        decoder[layer_num] = NonlinearityLayer(decoder[layer_num], nonlinearity=lasagne.nonlinearities.leaky_rectify)

    # Loss
    tar = {}
    rec = {}
    rec_clean = {}
    loss_rec = {}
    loss_rec_clean = {}  # Kamran: Double check this for validation
    for layer_num in range(num_layers - 1):
        # target for first layer is just the input value
        if layer_num == 0:
            tar[layer_num] = input_var
        # target for the rest of the layers is the clean encoder output for that layer
        else:
            tar[layer_num] = get_output(encoder[layer_num], deterministic=True)

        # rec is the decoder output from that layer
        rec[layer_num] = get_output(decoder[layer_num])
        loss_rec[layer_num] = lasagne.objectives.squared_error(rec[layer_num], tar[layer_num])

        rec_clean[layer_num] = get_output(decoder[layer_num], deterministic=True)
        loss_rec_clean[layer_num] = lasagne.objectives.squared_error(rec_clean[layer_num], tar[layer_num])

        # Loss summation
        loss_rec[layer_num] *= hyperparameters[layer_num]
        loss_rec_clean[layer_num] *= hyperparameters[layer_num]

        if layer_num == 0:
            loss_reconstruction = loss_rec[layer_num].mean()
            loss_reconstruction_clean = loss_rec_clean[layer_num].mean()
        else:
            loss_reconstruction += loss_rec[layer_num].mean()
            loss_reconstruction_clean += loss_rec_clean[layer_num].mean()

    # Softmax layer for prediction
    classifier = DenseLayer(encoder[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
        loss_reconstruction *= num_labeled_data_ratio
    loss_classification = loss_classification.mean()

    loss = loss_reconstruction + loss_classification
    params_reconstruction = get_all_params(decoder[0], trainable=True)
    updates_reconstruction = lasagne.updates.adam(loss_reconstruction, params_reconstruction,
                                                  learning_rate=learning_rate)
    params_classification = get_all_params([classifier, decoder[0]], trainable=True)
    updates_classification = lasagne.updates.adam(loss, params_classification, learning_rate=learning_rate)

    ratio = 0
    # update_scale / param_scale
    for param in params_classification:
        ratio += ((param - updates_classification[param]).flatten(ndim=1).norm(2) + 1e-45) / (
        param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params_classification)
    # train & test function
    train_fn_reconstruction = theano.function([input_var], loss_reconstruction, updates=updates_reconstruction)
    if semi_supervised_flag:
        train_fn_joint = theano.function([input_var, target_var, semi_supervised_idx],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn_joint = theano.function([input_var, target_var],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    val_fn = theano.function([input_var], loss_reconstruction_clean)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        network = get_all_layers(decoder[0])
        network.append(classifier)
        draw_to_file(network, output_path, list_flag=True)

    return train_fn_reconstruction, train_fn_joint, val_fn, test_fn, classifier, decoder[0], ratio_fn


def build_dense_deep_CAE(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[50, 50],
                         kernel_sizes=[5, 5],
                         strides=[2, 2], paddings=[2, 2],
                         dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], num_labeled_data_ratio=None,
                         hyperparameters=[1, 0.1, 0.1], target_flag='soft',
                         batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                         initialization='He'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    encoder = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            encoder[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                            input_var=input_var)
        else:
            intermediate_layers = []
            for input_num in range(layer_num):
                if kernel_sizes[layer_num - input_num - 1][input_num] != 0:
                    encoder[str(layer_num) + '_' + str(input_num)] = Conv2DLayer(encoder[input_num],
                                                                                 num_filters=feature_map_sizes[
                                                                                     layer_num],
                                                                                 stride=(
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 filter_size=(
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 pad=
                                                                                 paddings[layer_num - input_num - 1][
                                                                                     input_num],
                                                                                 nonlinearity=lasagne.nonlinearities.linear,
                                                                                 W=init)
                    intermediate_layers.append(encoder[str(layer_num) + '_' + str(input_num)])
            encoder[layer_num] = ElemwiseSumLayer(intermediate_layers)
            if batch_norm_flag:
                encoder[layer_num] = batch_norm(encoder[layer_num])
            encoder[layer_num] = NonlinearityLayer(encoder[layer_num],
                                                   nonlinearity=lasagne.nonlinearities.leaky_rectify)
        encoder[layer_num] = DropoutLayer(encoder[layer_num], p=dropouts[layer_num])

    decoder = {}
    for layer_num in range(num_layers - 2, -1, -1):
        if layer_num == len(feature_map_sizes) - 2:
            last_layer = encoder[num_layers - 1]
        else:
            last_layer = decoder[layer_num + 1]

        decoder[layer_num] = Deconv2DLayer(last_layer, num_filters=feature_map_sizes[layer_num],
                                           stride=(strides[layer_num], strides[layer_num]),
                                           filter_size=(kernel_sizes[layer_num], kernel_sizes[layer_num]),
                                           crop=paddings[layer_num],
                                           nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                           W=init)

        if batch_norm_flag:
            decoder[layer_num] = batch_norm(decoder[layer_num])

    # Loss
    tar = {}
    rec = {}
    rec_clean = {}
    loss_rec = {}
    loss_rec_clean = {}  # Kamran: Double check this for validation
    for layer_num in range(num_layers - 1):
        # target for first layer is just the input value
        if layer_num == 0:
            tar[layer_num] = input_var
        # target for the rest of the layers is the clean encoder output for that layer
        else:
            tar[layer_num] = get_output(encoder[layer_num], deterministic=True)

        # rec is the decoder output from that layer
        rec[layer_num] = get_output(decoder[layer_num])
        loss_rec[layer_num] = lasagne.objectives.squared_error(rec[layer_num], tar[layer_num])

        rec_clean[layer_num] = get_output(decoder[layer_num], deterministic=True)
        loss_rec_clean[layer_num] = lasagne.objectives.squared_error(rec_clean[layer_num], tar[layer_num])

        # Loss summation
        loss_rec[layer_num] *= hyperparameters[layer_num]
        loss_rec_clean[layer_num] *= hyperparameters[layer_num]

        if layer_num == 0:
            loss_reconstruction = loss_rec[layer_num].mean()
            loss_reconstruction_clean = loss_rec_clean[layer_num].mean()
        else:
            loss_reconstruction += loss_rec[layer_num].mean()
            loss_reconstruction_clean += loss_rec_clean[layer_num].mean()

    # Softmax layer for prediction
    classifier = DenseLayer(encoder[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
        loss_reconstruction *= num_labeled_data_ratio
    loss_classification = loss_classification.mean()

    loss = loss_reconstruction + loss_classification
    params_reconstruction = get_all_params(decoder[0], trainable=True)
    updates_reconstruction = lasagne.updates.adam(loss_reconstruction, params_reconstruction,
                                                  learning_rate=learning_rate)
    params_classification = get_all_params([classifier, decoder[0]], trainable=True)
    updates_classification = lasagne.updates.adam(loss, params_classification, learning_rate=learning_rate)

    ratio = 0
    # update_scale / param_scale
    for param in params_classification:
        ratio += ((param - updates_classification[param]).flatten(ndim=1).norm(2) + 1e-45) / (
        param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params_classification)
    # train & test function
    train_fn_reconstruction = theano.function([input_var], loss_reconstruction, updates=updates_reconstruction)
    if semi_supervised_flag:
        train_fn_joint = theano.function([input_var, target_var, semi_supervised_idx],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn_joint = theano.function([input_var, target_var],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    val_fn = theano.function([input_var], loss_reconstruction_clean)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        network = get_all_layers(decoder[0])
        network.append(classifier)
        draw_to_file(network, output_path, list_flag=True)

    return train_fn_reconstruction, train_fn_joint, val_fn, test_fn, classifier, decoder[0], ratio_fn


def build_pool_dense_deep_CAE_old(width=None, height=None, num_class=None, learning_rate=1e-3,
                                  feature_map_sizes=[50, 50],
                                  kernel_sizes=[5, 5],
                                  strides=[2, 2], paddings=[2, 2], pool_sizes=None,
                                  dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], num_labeled_data_ratio=None,
                                  hyperparameters=[1, 0.1, 0.1], target_flag='soft',
                                  batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                                  initialization='He'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    encoder = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            encoder[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                            input_var=input_var)
        else:
            intermediate_layers = []
            for input_num in range(layer_num):
                if kernel_sizes[layer_num - input_num - 1][input_num] != 0:
                    if layer_num - input_num - 1 > 0:
                        input_layer = MaxPool2DLayer(encoder[input_num],
                                                     pool_size=pool_sizes[input_num % len(pool_sizes)][
                                                         layer_num - input_num - 1])
                    else:
                        input_layer = encoder[input_num]
                    encoder[str(layer_num) + '_' + str(input_num)] = Conv2DLayer(input_layer,
                                                                                 num_filters=feature_map_sizes[
                                                                                     layer_num],
                                                                                 stride=(
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 filter_size=(
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 pad=
                                                                                 paddings[layer_num - input_num - 1][
                                                                                     input_num],
                                                                                 nonlinearity=lasagne.nonlinearities.linear,
                                                                                 W=init)
                    intermediate_layers.append(encoder[str(layer_num) + '_' + str(input_num)])
            encoder[layer_num] = ElemwiseSumLayer(intermediate_layers)
            if batch_norm_flag:
                encoder[layer_num] = batch_norm(encoder[layer_num])
            encoder[layer_num] = NonlinearityLayer(encoder[layer_num],
                                                   nonlinearity=lasagne.nonlinearities.leaky_rectify)
        encoder[layer_num] = DropoutLayer(encoder[layer_num], p=dropouts[layer_num])

    decoder = {}
    for layer_num in range(num_layers - 2, -1, -1):
        intermediate_layers = []
        for input_num in range(num_layers - 1, layer_num, -1):
            if input_num == num_layers - 1:
                last_layer = encoder[num_layers - 1]
            else:
                last_layer = decoder[input_num]
            if input_num - layer_num - 1 > 0:
                last_layer = Upscale2DLayer(last_layer,
                                            scale_factor=pool_sizes[(num_layers - input_num) % len(pool_sizes)][
                                                input_num - layer_num - 1], mode='repeat')

            if kernel_sizes[input_num - layer_num - 1][layer_num] != 0:
                decoder[str(layer_num) + '_' + str(input_num)] = Deconv2DLayer(last_layer,
                                                                               num_filters=feature_map_sizes[
                                                                                   layer_num],
                                                                               stride=(
                                                                                   strides[input_num - layer_num - 1][
                                                                                       layer_num],
                                                                                   strides[input_num - layer_num - 1][
                                                                                       layer_num]),
                                                                               filter_size=(
                                                                                   kernel_sizes[
                                                                                       input_num - layer_num - 1][
                                                                                       layer_num],
                                                                                   kernel_sizes[
                                                                                       input_num - layer_num - 1][
                                                                                       layer_num]),
                                                                               crop=paddings[input_num - layer_num - 1][
                                                                                   layer_num],
                                                                               nonlinearity=lasagne.nonlinearities.linear,
                                                                               W=init)
            intermediate_layers.append(decoder[str(layer_num) + '_' + str(input_num)])
        decoder[layer_num] = ElemwiseSumLayer(intermediate_layers)

        if batch_norm_flag:
            decoder[layer_num] = batch_norm(decoder[layer_num])
        decoder[layer_num] = NonlinearityLayer(decoder[layer_num], nonlinearity=lasagne.nonlinearities.leaky_rectify)

    # Loss
    tar = {}
    rec = {}
    rec_clean = {}
    loss_rec = {}
    loss_rec_clean = {}  # Kamran: Double check this for validation
    for layer_num in range(num_layers - 1):
        # target for first layer is just the input value
        if layer_num == 0:
            tar[layer_num] = input_var
        # target for the rest of the layers is the clean encoder output for that layer
        else:
            tar[layer_num] = get_output(encoder[layer_num], deterministic=True)

        # rec is the decoder output from that layer
        rec[layer_num] = get_output(decoder[layer_num])
        loss_rec[layer_num] = lasagne.objectives.squared_error(rec[layer_num], tar[layer_num])

        rec_clean[layer_num] = get_output(decoder[layer_num], deterministic=True)
        loss_rec_clean[layer_num] = lasagne.objectives.squared_error(rec_clean[layer_num], tar[layer_num])

        # Loss summation
        loss_rec[layer_num] *= hyperparameters[layer_num]
        loss_rec_clean[layer_num] *= hyperparameters[layer_num]

        if layer_num == 0:
            loss_reconstruction = loss_rec[layer_num].mean()
            loss_reconstruction_clean = loss_rec_clean[layer_num].mean()
        else:
            loss_reconstruction += loss_rec[layer_num].mean()
            loss_reconstruction_clean += loss_rec_clean[layer_num].mean()

    # Softmax layer for prediction
    classifier = DenseLayer(encoder[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
        loss_reconstruction *= num_labeled_data_ratio
    loss_classification = loss_classification.mean()

    loss = loss_reconstruction + loss_classification
    params_reconstruction = get_all_params(decoder[0], trainable=True)
    updates_reconstruction = lasagne.updates.adam(loss_reconstruction, params_reconstruction,
                                                  learning_rate=learning_rate)
    params_classification = get_all_params([classifier, decoder[0]], trainable=True)
    updates_classification = lasagne.updates.adam(loss, params_classification, learning_rate=learning_rate)

    ratio = 0
    # update_scale / param_scale
    for param in params_classification:
        ratio += ((param - updates_classification[param]).flatten(ndim=1).norm(2) + 1e-45) / (
        param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params_classification)
    # train & test function
    train_fn_reconstruction = theano.function([input_var], loss_reconstruction, updates=updates_reconstruction)
    if semi_supervised_flag:
        train_fn_joint = theano.function([input_var, target_var, semi_supervised_idx],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn_joint = theano.function([input_var, target_var],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    val_fn = theano.function([input_var], loss_reconstruction_clean)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        network = get_all_layers(decoder[0])
        network.append(classifier)
        draw_to_file(network, output_path, list_flag=True)

    return train_fn_reconstruction, train_fn_joint, val_fn, test_fn, classifier, decoder[0], ratio_fn


def build_pool_dense_deep_CAE(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[50, 50],
                              kernel_sizes=[5, 5],
                              strides=[2, 2], paddings=[2, 2], pool_sizes=None,
                              dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], num_labeled_data_ratio=None,
                              hyperparameters=[1, 0.1, 0.1], target_flag='soft',
                              batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                              initialization='He'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    encoder = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            encoder[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                            input_var=input_var)
        else:
            intermediate_layers = []
            for input_num in range(layer_num):
                if kernel_sizes[layer_num - input_num - 1][input_num] != 0:
                    if pool_sizes[input_num][layer_num] > 0:
                        input_layer = MaxPool2DLayer(encoder[input_num],
                                                     pool_size=pool_sizes[input_num][layer_num])
                    else:
                        input_layer = encoder[input_num]
                    encoder[str(layer_num) + '_' + str(input_num)] = Conv2DLayer(input_layer,
                                                                                 num_filters=feature_map_sizes[
                                                                                     layer_num],
                                                                                 stride=(
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     strides[layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 filter_size=(
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num],
                                                                                     kernel_sizes[
                                                                                         layer_num - input_num - 1][
                                                                                         input_num]),
                                                                                 pad=
                                                                                 paddings[layer_num - input_num - 1][
                                                                                     input_num],
                                                                                 nonlinearity=lasagne.nonlinearities.linear,
                                                                                 W=init)
                    intermediate_layers.append(encoder[str(layer_num) + '_' + str(input_num)])
            encoder[layer_num] = ElemwiseSumLayer(intermediate_layers)
            if batch_norm_flag:
                encoder[layer_num] = batch_norm(encoder[layer_num])
            encoder[layer_num] = NonlinearityLayer(encoder[layer_num],
                                                   nonlinearity=lasagne.nonlinearities.leaky_rectify)
        encoder[layer_num] = DropoutLayer(encoder[layer_num], p=dropouts[layer_num])

    decoder = {}
    for layer_num in range(num_layers - 2, -1, -1):
        if layer_num == len(feature_map_sizes) - 2:
            last_layer = encoder[num_layers - 1]
        else:
            last_layer = decoder[layer_num + 1]

        decoder[layer_num] = Deconv2DLayer(last_layer, num_filters=feature_map_sizes[layer_num],
                                           stride=(strides[0][layer_num], strides[0][layer_num]),
                                           filter_size=(kernel_sizes[0][layer_num], kernel_sizes[0][layer_num]),
                                           crop=paddings[0][layer_num],
                                           nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                           W=init)

        if batch_norm_flag:
            decoder[layer_num] = batch_norm(decoder[layer_num])

    # Loss
    tar = {}
    rec = {}
    rec_clean = {}
    loss_rec = {}
    loss_rec_clean = {}  # Kamran: Double check this for validation
    for layer_num in range(num_layers - 1):
        # target for first layer is just the input value
        if layer_num == 0:
            tar[layer_num] = input_var
        # target for the rest of the layers is the clean encoder output for that layer
        else:
            tar[layer_num] = get_output(encoder[layer_num], deterministic=True)

        # rec is the decoder output from that layer
        rec[layer_num] = get_output(decoder[layer_num])
        loss_rec[layer_num] = lasagne.objectives.squared_error(rec[layer_num], tar[layer_num])

        rec_clean[layer_num] = get_output(decoder[layer_num], deterministic=True)
        loss_rec_clean[layer_num] = lasagne.objectives.squared_error(rec_clean[layer_num], tar[layer_num])

        # Loss summation
        loss_rec[layer_num] *= hyperparameters[layer_num]
        loss_rec_clean[layer_num] *= hyperparameters[layer_num]

        if layer_num == 0:
            loss_reconstruction = loss_rec[layer_num].mean()
            loss_reconstruction_clean = loss_rec_clean[layer_num].mean()
        else:
            loss_reconstruction += loss_rec[layer_num].mean()
            loss_reconstruction_clean += loss_rec_clean[layer_num].mean()

    # Softmax layer for prediction
    classifier = DenseLayer(encoder[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
        loss_reconstruction *= num_labeled_data_ratio
    loss_classification = loss_classification.mean()

    loss = loss_reconstruction + loss_classification
    params_reconstruction = get_all_params(decoder[0], trainable=True)
    updates_reconstruction = lasagne.updates.adam(loss_reconstruction, params_reconstruction,
                                                  learning_rate=learning_rate)
    params_classification = get_all_params([classifier, decoder[0]], trainable=True)
    updates_classification = lasagne.updates.adam(loss, params_classification, learning_rate=learning_rate)

    ratio = 0
    # update_scale / param_scale
    for param in params_classification:
        ratio += ((param - updates_classification[param]).flatten(ndim=1).norm(2) + 1e-45) / (
        param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params_classification)
    # train & test function
    train_fn_reconstruction = theano.function([input_var], loss_reconstruction, updates=updates_reconstruction)
    if semi_supervised_flag:
        train_fn_joint = theano.function([input_var, target_var, semi_supervised_idx],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn_joint = theano.function([input_var, target_var],
                                         [loss, loss_classification, loss_reconstruction, prediction_noisy],
                                         updates=updates_classification)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    val_fn = theano.function([input_var], loss_reconstruction_clean)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        network = get_all_layers(decoder[0])
        network.append(classifier)
        draw_to_file(network, output_path, list_flag=True)

    return train_fn_reconstruction, train_fn_joint, val_fn, test_fn, classifier, decoder[0], ratio_fn


def build_deep_CNN(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[1, 50, 50],
                   kernel_sizes=[5, 5],
                   strides=[2, 2], paddings=[2, 2],
                   dropouts=[0.1, 0.1, 0.1], hyperparameters=[1, 0.1, 0.1], target_flag='soft',
                   batch_norm_flag=True, draw_flag=True, output_path=None, semi_supervised_flag=False,
                   global_pool_flag=False, initialization='He', update='adam', weight_decay=0.001):
    input_var = T.tensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    num_layers = len(feature_map_sizes)

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()
    elif initialization == 'Gaussian':
        init = lasagne.init.Normal(std=0.05)

    # Create hidden layers
    layers = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            layers[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                           input_var=input_var)
        else:
            layers[layer_num] = Conv2DLayer(layers[layer_num - 1],
                                            num_filters=feature_map_sizes[layer_num],
                                            stride=(strides[layer_num - 1], strides[layer_num - 1]),
                                            filter_size=(
                                                kernel_sizes[layer_num - 1],
                                                kernel_sizes[layer_num - 1]),
                                            pad=paddings[layer_num - 1],
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=init, b=lasagne.init.Constant(0.))
            if batch_norm_flag:
                layers[layer_num] = batch_norm(layers[layer_num])
        if (layer_num % 3 == 0) & (layer_num < num_layers - 1):
            layers[layer_num] = DropoutLayer(layers[layer_num], p=dropouts[layer_num / 3])

    last_layer = layers[num_layers - 1]
    # Global pool layer
    if global_pool_flag:
        last_layer = GlobalPoolLayer(last_layer)

    # Softmax layer for prediction
    classifier = DenseLayer(last_layer, num_units=num_class, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy, target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
    loss_classification = loss_classification.mean()
    # regu_loss = lasagne.regularization.regularize_network_params(
    #     classifier, lasagne.regularization.l2)
    # loss_classification += weight_decay * regu_loss
    l2_loss = weight_decay * lasagne.regularization.regularize_network_params(
        classifier, lasagne.regularization.l2, {'trainable': True})
    loss_classification += l2_loss

    params = get_all_params(classifier, trainable=True)
    if update == 'adam':
        updates = lasagne.updates.adam(loss_classification, params, learning_rate=learning_rate)
    elif update == 'Nesterov':
        updates = lasagne.updates.nesterov_momentum(loss_classification, params, learning_rate=learning_rate,
                                                    momentum=0.9)

    ratio = 0
    # update_scale / param_scale
    for param in params:
        ratio += ((param - updates[param]).flatten(ndim=1).norm(2) + 1e-45) / (param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params)
    # train & test function
    if semi_supervised_flag:
        train_fn = theano.function([input_var, target_var, semi_supervised_idx],
                                   [loss_classification, prediction_noisy], updates=updates)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn = theano.function([input_var, target_var], [loss_classification, prediction_noisy], updates=updates)
        ratio_fn = theano.function([input_var, target_var], ratio)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    l2_fn = theano.function([], l2_loss)
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        draw_to_file(classifier, output_path, list_flag=False)

    return train_fn, test_fn, classifier, ratio_fn, l2_fn


def build_dense_deep_CNN(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[50, 50],
                         kernel_sizes=[5, 5],
                         strides=[2, 2], paddings=[2, 2],
                         dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], hyperparameters=[1, 0.1, 0.1], target_flag='soft',
                         batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                         initialization='He', update='adam'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    layers = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            layers[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                           input_var=input_var)
        else:
            intermediate_layers = []
            for input_num in range(layer_num):
                if kernel_sizes[layer_num - input_num - 1][input_num] != 0:
                    layers[str(layer_num) + '_' + str(input_num)] = Conv2DLayer(layers[input_num],
                                                                                num_filters=feature_map_sizes[
                                                                                    layer_num],
                                                                                stride=(
                                                                                    strides[layer_num - input_num - 1][
                                                                                        input_num],
                                                                                    strides[layer_num - input_num - 1][
                                                                                        input_num]),
                                                                                filter_size=(
                                                                                    kernel_sizes[
                                                                                        layer_num - input_num - 1][
                                                                                        input_num],
                                                                                    kernel_sizes[
                                                                                        layer_num - input_num - 1][
                                                                                        input_num]),
                                                                                pad=paddings[layer_num - input_num - 1][
                                                                                    input_num],
                                                                                nonlinearity=lasagne.nonlinearities.linear,
                                                                                W=init)
                    intermediate_layers.append(layers[str(layer_num) + '_' + str(input_num)])
            layers[layer_num] = ElemwiseSumLayer(intermediate_layers)
            if batch_norm_flag:
                layers[layer_num] = batch_norm(layers[layer_num])
            layers[layer_num] = NonlinearityLayer(layers[layer_num], nonlinearity=lasagne.nonlinearities.leaky_rectify)
        layers[layer_num] = DropoutLayer(layers[layer_num], p=dropouts[layer_num])

    # Softmax layer for prediction
    classifier = DenseLayer(layers[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0.),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
    loss_classification = loss_classification.mean()

    params = get_all_params(classifier, trainable=True)
    if update == 'adam':
        updates = lasagne.updates.adam(loss_classification, params, learning_rate=learning_rate)
    elif update == 'Nesterov':
        updates = lasagne.updates.nesterov_momentum(loss_classification, params, learning_rate=learning_rate,
                                                    momentum=0.9)

    ratio = 0
    # update_scale / param_scale
    for param in params:
        ratio += ((param - updates[param]).flatten(ndim=1).norm(2) + 1e-45) / (param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params)
    # train & test function
    if semi_supervised_flag:
        train_fn = theano.function([input_var, target_var, semi_supervised_idx],
                                   [loss_classification, prediction_noisy], updates=updates)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn = theano.function([input_var, target_var], [loss_classification, prediction_noisy], updates=updates)
        ratio_fn = theano.function([input_var, target_var], ratio)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        draw_to_file(classifier, output_path, list_flag=False)

    return train_fn, test_fn, classifier, ratio_fn


def build_pool_dense_deep_CNN(width=None, height=None, num_class=None, learning_rate=1e-3, feature_map_sizes=[50, 50],
                              kernel_sizes=[5, 5],
                              strides=[2, 2], paddings=[2, 2],
                              dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], pool_sizes=None, hyperparameters=[1, 0.1, 0.1],
                              target_flag='soft',
                              batch_norm_flag=True, draw_flag=False, output_path=None, semi_supervised_flag=False,
                              initialization='He', update='adam'):
    input_var = T.ftensor4('inputs')
    if target_flag == 'hard':
        target_var = T.ivector('target_var')
    else:
        target_var = T.matrix('target_var')
    semi_supervised_idx = T.ivector('semi_idx_var')

    if initialization == 'Glorot':
        init = lasagne.init.GlorotUniform()
    elif initialization == 'He':
        init = lasagne.init.HeUniform()

    num_layers = len(feature_map_sizes)

    layers = {}
    for layer_num in range(num_layers):
        if layer_num == 0:
            layers[layer_num] = InputLayer(shape=(None, feature_map_sizes[layer_num], width, height),
                                           input_var=input_var)
        else:
            # iterate over al previous layers to compute inputs for current layer
            intermediate_layers = []
            for input_num in range(layer_num):
                if kernel_sizes[layer_num - input_num - 1][input_num] != 0:
                    if pool_sizes[input_num][layer_num] > 0:
                        # input_layer = MaxPool2DLayer(layers[input_num], pool_size=pool_sizes[input_num % len(pool_sizes)][layer_num - input_num - 1])
                        input_layer = MaxPool2DLayer(layers[input_num], pool_size=pool_sizes[input_num][layer_num])
                    else:
                        input_layer = layers[input_num]
                    layers[str(layer_num) + '_' + str(input_num)] = Conv2DLayer(input_layer,
                                                                                num_filters=feature_map_sizes[
                                                                                    layer_num],
                                                                                stride=(
                                                                                    strides[layer_num - input_num - 1][
                                                                                        input_num],
                                                                                    strides[layer_num - input_num - 1][
                                                                                        input_num]),
                                                                                filter_size=(
                                                                                    kernel_sizes[
                                                                                        layer_num - input_num - 1][
                                                                                        input_num],
                                                                                    kernel_sizes[
                                                                                        layer_num - input_num - 1][
                                                                                        input_num]),
                                                                                pad=paddings[layer_num - input_num - 1][
                                                                                    input_num],
                                                                                nonlinearity=lasagne.nonlinearities.linear,
                                                                                W=init)
                    intermediate_layers.append(layers[str(layer_num) + '_' + str(input_num)])
            layers[layer_num] = ElemwiseSumLayer(intermediate_layers)
            if batch_norm_flag:
                layers[layer_num] = batch_norm(layers[layer_num])
            layers[layer_num] = NonlinearityLayer(layers[layer_num], nonlinearity=lasagne.nonlinearities.leaky_rectify)
        layers[layer_num] = DropoutLayer(layers[layer_num], p=dropouts[layer_num])

    # Softmax layer for prediction
    classifier = DenseLayer(layers[num_layers - 1], num_units=num_class, W=init,
                            b=lasagne.init.Constant(0.),
                            nonlinearity=lasagne.nonlinearities.softmax)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy,
                                                                      target_var)  # loss function
    if semi_supervised_flag:
        loss_classification = semi_supervised_idx * loss_classification
    loss_classification = loss_classification.mean()

    params = get_all_params(classifier, trainable=True)
    if update == 'adam':
        updates = lasagne.updates.adam(loss_classification, params, learning_rate=learning_rate)
    elif update == 'Nesterov':
        updates = lasagne.updates.nesterov_momentum(loss_classification, params, learning_rate=learning_rate,
                                                    momentum=0.9)
    ratio = 0
    # update_scale / param_scale
    for param in params:
        ratio += ((param - updates[param]).flatten(ndim=1).norm(2) + 1e-45) / (param.flatten(ndim=1).norm(2) + 1e-45)

    ratio /= len(params)

    # train & test function
    if semi_supervised_flag:
        train_fn = theano.function([input_var, target_var, semi_supervised_idx],
                                   [loss_classification, prediction_noisy], updates=updates)
        ratio_fn = theano.function([input_var, target_var, semi_supervised_idx], ratio)
    else:
        train_fn = theano.function([input_var, target_var], [loss_classification, prediction_noisy], updates=updates)
        ratio_fn = theano.function([input_var, target_var], ratio)
    # test_fn = theano.function([input_var], [prediction_clean, tar[num_layers - 1]])
    test_fn = theano.function([input_var], prediction_clean)
    if draw_flag:
        draw_to_file(classifier, output_path, list_flag=False)

    return train_fn, test_fn, classifier, ratio_fn


def train_deep(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, num_classes=None,
               parameters=None,
               num_epochs=None, val_accuracies=None,
               test_accuracies=None, best_epoch=None, classifier=None, batch_size=None, test_batch_size=None,
               train_fn=None, test_fn=None, lr=None,
               dropouts=None, loss_acc_plt_flag=None, output_path=None, best_loss=None, augment_flag=False,
               semi_supervised_flag=False, semi_supervised_idx=None, ratio_fn=None):
    y_prob_val = np.zeros((X_val.shape[0], num_classes))
    y_prob_train = np.zeros((X_train.shape[0], num_classes))
    y_prob_test = np.zeros((X_test.shape[0], num_classes))
    best_acc = 0
    parameters.append(get_all_param_values(classifier))
    loss_hist = []
    val_acc_hist = []
    test_acc_hist = []
    train_acc_hist = []
    best_epoch.append(0)
    best_loss.append(np.inf)
    update_counter = 0
    for epoch in range(num_epochs):

        train_loss = 0
        ratio = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets, idx = batch

            if augment_flag:
                inputs = augment_minibatch(inputs)

            if semi_supervised_flag:
                semi_supervised_batch_idx = semi_supervised_idx[idx]
                loss, prob = train_fn(inputs, targets, semi_supervised_batch_idx)
                ratio += ratio_fn(inputs, targets, semi_supervised_batch_idx)
            else:
                loss, prob = train_fn(inputs, targets)
                ratio += ratio_fn(inputs, targets)
            train_loss += loss
            y_prob_train[idx] = prob
            train_batches += 1

        pred_y_train = np.argmax(y_prob_train, axis=1)
        acc_train = accuracy_score(pred_y_train, y_train)
        train_acc_hist.append(acc_train)
        loss_hist.append(train_loss / train_batches)

        # And a full pass over the validation data:
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, test_batch_size, shuffle=False):
            inputs, targets, idx = batch

            prob = test_fn(inputs)
            y_prob_val[idx] = prob
            val_batches += 1

        pred_y_val = np.argmax(y_prob_val, axis=1)
        acc_val = accuracy_score(pred_y_val, y_val)
        val_acc_hist.append(acc_val)

        update_counter += 1
        if acc_val > best_acc:
            print('New best here!')
            best_acc = acc_val
            parameters.pop()
            parameters.append(get_all_param_values(classifier))
            best_epoch.pop()
            best_epoch.append(epoch)
            best_loss.pop()
            best_loss.append(train_loss / train_batches)
            update_counter = 0

        if update_counter > 300:
            break

        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, test_batch_size, shuffle=False):
            inputs, targets, idx = batch

            prob = test_fn(inputs)
            y_prob_test[idx] = prob
            test_batches += 1

        pred_y_test = np.argmax(y_prob_test, axis=1)
        acc_test = accuracy_score(pred_y_test, y_test)
        test_acc_hist.append(acc_test)

        # Then we print the results for this epoch:
        print(
            "Epoch {} of {}\ttraining loss:\t{:.10f}\tvalidation accuracy:\t{:.2f}\ttest accuracy:\t{:.2f}\tratio:\t{:.1e}\tLearning rate: {:.1e} took {:.3f}s".format(
                epoch + 1, num_epochs, train_loss / train_batches, acc_val * 100, acc_test * 100, ratio / train_batches,
                lr, time.time() - start_time))

    # Reproduce best model results
    test_batches = 0
    set_all_param_values(classifier, parameters[-1])
    for batch in iterate_minibatches(X_test, y_test, test_batch_size, shuffle=False):
        inputs, targets, idx = batch

        prob = test_fn(inputs)
        y_prob_test[idx] = prob
        test_batches += 1

    pred_y = np.argmax(y_prob_test, axis=1)
    acc_test = accuracy_score(pred_y, y_test)

    # Then we print the results for this epoch:
    print("\nBest model test accuracy:\t\t{:.2f}".format(acc_test * 100))

    if loss_acc_plt_flag:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        loss_line, = ax1.plot(loss_hist, 'r', label='training loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss')
        ax2 = plt.twinx(ax1)
        val_line, = ax2.plot(val_acc_hist, 'g', label='validation accuracy')
        test_line, = ax2.plot(test_acc_hist, 'b', label='test accuracy')
        train_line, = ax2.plot(train_acc_hist, 'y', label='train accuracy')
        ax2.set_ylabel('accuracies')
        fig.legend((loss_line, val_line, test_line, train_line),
                   ('training loss', 'validation accuracy', 'test accuracy', 'train accuracy'),
                   'upper left')
        fig.savefig(os.path.join(output_path, str(lr) + '_' + str(dropouts) + '.eps'))

    test_accuracies.append(acc_test)
    val_accuracies.append(best_acc)


def train_deep_final(X=None, y=None, X_test=None, y_test=None, num_classes=None,
                     num_epochs=None, batch_size=None, test_batch_size=None, train_fn=None, test_fn=None, lr=None,
                     dropouts=None, loss_acc_plt_flag=None, output_path=None, best_loss=None, augment_flag=False,
                     semi_supervised_flag=False, semi_supervised_idx=None, ratio_fn=None, l2_fn=None, lr_shared=None):
    y_prob_test = np.zeros((X_test.shape[0], num_classes))
    y_prob_train = np.zeros((X.shape[0], num_classes))
    best_acc = 0
    loss_hist = []
    val_acc_hist = []
    test_acc_hist = []
    train_acc_hist = []
    for epoch in range(num_epochs):

        train_loss = 0
        ratio = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
            inputs, targets, idx = batch

            if augment_flag:
                inputs = augment_minibatch(inputs)

            if semi_supervised_flag:
                semi_supervised_batch_idx = semi_supervised_idx[idx]
                loss, prob = train_fn(inputs, targets, semi_supervised_batch_idx)
                ratio += ratio_fn(inputs, targets, semi_supervised_batch_idx)
            else:
                loss, prob = train_fn(inputs, targets)
                ratio += ratio_fn(inputs, targets)
            train_loss += loss
            y_prob_train[idx] = prob
            train_batches += 1

        pred_y_train = np.argmax(y_prob_train, axis=1)
        acc_train = accuracy_score(pred_y_train, y)
        train_acc_hist.append(acc_train)
        loss_hist.append(train_loss / train_batches)
        l2_loss = l2_fn()

        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, test_batch_size, shuffle=False):
            inputs, targets, idx = batch

            prob = test_fn(inputs)
            y_prob_test[idx] = prob
            test_batches += 1

        pred_y_test = np.argmax(y_prob_test, axis=1)
        acc_test = accuracy_score(pred_y_test, y_test)
        test_acc_hist.append(acc_test)

        if acc_test > best_acc:
            print('New best here!')
            best_acc = acc_test
            best_epoch = epoch
            best_l = train_loss / train_batches

        # Then we print the results for this epoch:
        print(
            "Epoch {} of {}\ttraining loss:\t{:.5f}\tl2 loss:\t{:.5f}\ttrain accuracy:\t{:.2f}\ttest accuracy:\t{:.2f}\tratio:\t{:.1e}\tLearning rate: {:.1e} took {:.3f}s".format(
                epoch + 1, num_epochs, train_loss / train_batches, float(l2_loss), acc_train * 100, acc_test * 100,
                ratio / train_batches,
                lr, time.time() - start_time))

        if (epoch == 200) | (epoch == 250) | (epoch == 300):
            lr /= 10
            print('new learning rate: ', lr)
            lr_shared.set_value(lr)

        # if (ratio / train_batches) > 1e-2:
        #     lr /= 2
        #     print('new learning rate: ', lr)
        #     lr_shared.set_value(lr)
        # elif (ratio / train_batches) < 1e-4:
        #     lr *= 2
        #     print('new learning rate: ', lr)
        #     lr_shared.set_value(lr)


        if train_loss / train_batches < best_loss:
            break

    print('Best test accuracy:\t{:.2f}\t at epoch {} with loss:\t{:.10f}'.format(best_acc * 100, best_epoch, best_l))

    if loss_acc_plt_flag:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        loss_line, = ax1.plot(loss_hist, 'r', label='training loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss')
        ax2 = plt.twinx(ax1)
        test_line, = ax2.plot(test_acc_hist, 'b', label='test accuracy')
        train_line, = ax2.plot(train_acc_hist, 'y', label='train accuracy')
        ax2.set_ylabel('accuracies')
        fig.legend((loss_line, test_line, train_line), ('training loss', 'test accuracy', 'train accuracy'),
                   'upper left')
        fig.savefig(os.path.join(output_path, str(lr) + '_' + str(dropouts) + '_best.eps'))


def train_deep_AE_final(X=None, y=None, X_test=None, y_test=None, num_classes=None,
                        num_epochs=None, batch_size=None, test_batch_size=None, train_fn=None, test_fn=None, lr=None,
                        dropouts=None, loss_acc_plt_flag=None, output_path=None, best_loss=None, augment_flag=False,
                        semi_supervised_flag=False, semi_supervised_idx=None, ratio_fn=None):
    y_prob_test = np.zeros((X_test.shape[0], num_classes))
    y_prob_train = np.zeros((X.shape[0], num_classes))
    best_acc = 0
    loss_hist = []
    val_acc_hist = []
    test_acc_hist = []
    train_acc_hist = []
    for epoch in range(num_epochs):

        train_loss = 0
        class_loss = 0
        rec_loss = 0
        ratio = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
            inputs, targets, idx = batch

            if augment_flag:
                inputs = augment_minibatch(inputs)

            if semi_supervised_flag:
                semi_supervised_batch_idx = semi_supervised_idx[idx]
                loss, c_loss, re_loss, prob = train_fn(inputs, targets, semi_supervised_batch_idx)
                ratio += ratio_fn(inputs, targets, semi_supervised_batch_idx)
            else:
                loss, c_loss, re_loss, prob = train_fn(inputs, targets)
                ratio += ratio_fn(inputs, targets)

            train_loss += loss
            class_loss += c_loss
            rec_loss += re_loss
            y_prob_train[idx] = prob

            train_batches += 1

        pred_y_train = np.argmax(y_prob_train, axis=1)
        acc_train = accuracy_score(pred_y_train, y)
        train_acc_hist.append(acc_train)
        loss_hist.append(train_loss / train_batches)

        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, test_batch_size, shuffle=False):
            inputs, targets, idx = batch

            prob = test_fn(inputs)
            y_prob_test[idx] = prob
            test_batches += 1

        pred_y_test = np.argmax(y_prob_test, axis=1)
        acc_test = accuracy_score(pred_y_test, y_test)
        test_acc_hist.append(acc_test)

        if acc_test > best_acc:
            print('New best here!')
            best_acc = acc_test
            best_epoch = epoch
            best_l = train_loss / train_batches

        # Then we print the results for this epoch:
        print(
            "Epoch {} of {}\ttraining loss:\t{:.10f}\tclassification loss:\t{:.10f}\treconstruction loss:\t{:.10f}\ttest accuracy:\t{:.2f}\tratio:\t{:.1e}\tLearning rate: {:.1e} took {:.3f}s".format(
                epoch + 1, num_epochs, train_loss / train_batches, class_loss / train_batches, rec_loss / train_batches,
                acc_test * 100, ratio / train_batches,
                lr, time.time() - start_time))

        if train_loss / train_batches < best_loss:
            break

    print('Best test accuracy:\t{:.2f}\t at epoch {} with loss:\t{:.10f}'.format(best_acc * 100, best_epoch, best_l))

    if loss_acc_plt_flag:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        loss_line, = ax1.plot(loss_hist, 'r', label='training loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss')
        ax2 = plt.twinx(ax1)
        test_line, = ax2.plot(test_acc_hist, 'b', label='test accuracy')
        train_line, = ax2.plot(train_acc_hist, 'y', label='train accuracy')
        ax2.set_ylabel('accuracies')
        fig.legend((loss_line, test_line, train_line), ('training loss', 'test accuracy', 'train_accuracy'),
                   'upper left')
        fig.savefig(os.path.join(output_path, str(lr) + '_' + str(dropouts) + '_best.eps'))


def hyperparameter_tuning_lr_drop(X=None, y=None, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None,
                                  y_test=None, num_class=None,
                                  num_epochs=None, classifier=None, batch_size=None,
                                  test_batch_size=None, train_fn=None, test_fn=None,
                                  output_path=None, learning_rate_set=None, dropout_set=None, lr_shared=None,
                                  dropouts_shared=None, augment_flag=False,
                                  semi_supervised_flag=None,
                                  semi_supervised_idx=None):
    parameters = []
    val_accuracies = []
    test_accuracies = []
    best_epoch = []
    best_loss = []
    learning_rate = []
    dropout = []
    init_params = lasagne.layers.get_all_param_values(classifier)
    num_params = 0
    for param in init_params:
        num_params += param.size
    print('Number of parameters: ', num_params)
    loss_acc_plt_flag = True

    for l_rate in learning_rate_set:
        lr_shared.set_value(np.array(l_rate, dtype='float32'))
        for d_out in dropout_set:
            dropouts_shared.set_value(np.array(d_out, dtype='float32'))

            set_all_param_values(classifier, init_params)
            learning_rate.append(l_rate)
            dropout.append(d_out)
            train_deep(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                       num_classes=num_class, parameters=parameters,
                       num_epochs=num_epochs, val_accuracies=val_accuracies, test_accuracies=test_accuracies,
                       best_epoch=best_epoch, classifier=classifier, batch_size=batch_size,
                       test_batch_size=test_batch_size, train_fn=train_fn, test_fn=test_fn,
                       loss_acc_plt_flag=loss_acc_plt_flag,
                       output_path=output_path, lr=l_rate, dropouts=d_out,
                       best_loss=best_loss, augment_flag=augment_flag, semi_supervised_flag=semi_supervised_flag,
                       semi_supervised_idx=semi_supervised_idx)  # Amir, Kamran: Plot val test accuracy, train loss

            print('\nlearning rate: ', l_rate)
            print('dropouts: ', d_out)
            print('best epoch: ', best_epoch[-1])
            print('validation accuracies: ', val_accuracies[-1])
            print('test accuracies: ', test_accuracies[-1])
            print('\n------------------------------------------------------------------------------\n')

    print('\nlearning rate: ', learning_rate_set)
    print('dropouts: ', dropout_set)
    print('best epoch: ', best_epoch)
    print('validation accuracies: ', val_accuracies)
    print('test accuracies: ', test_accuracies)
    print('\n------------------------------------------------------------------------------\n')

    best_index = np.argmax(val_accuracies)
    print('learning rate: ', learning_rate[best_index])
    print('dropouts: ', dropout[best_index])
    print('best epoch: ', best_epoch[best_index])
    print('validation accuracies: ', val_accuracies[best_index])
    print('test accuracies: ', test_accuracies[best_index])

    # Final training with validation set as well
    set_all_param_values(classifier, init_params)
    lr_shared.set_value(np.array(learning_rate[best_index], dtype='float32'))
    dropouts_shared.set_value(np.array(dropout[best_index], dtype='float32'))

    train_deep_final(X, y, X_test, y_test, num_class,
                     10000, batch_size, test_batch_size, train_fn, test_fn, learning_rate[best_index],
                     dropout[best_index], loss_acc_plt_flag, output_path, best_loss[best_index],
                     augment_flag=augment_flag, semi_supervised_flag=semi_supervised_flag,
                     semi_supervised_idx=semi_supervised_idx)


def hyperparameter_tuning_lr_drop_augmented(X=None, y=None, X_train=None, y_train=None, X_val=None, y_val=None,
                                            X_test=None,
                                            y_test=None, num_class=None,
                                            num_epochs=None, classifier=None, batch_size=None,
                                            test_batch_size=None, train_fn=None, test_fn=None,
                                            output_path=None, learning_rate_set=None, dropout_set=None, lr_shared=None,
                                            dropouts_shared=None):
    parameters = []
    val_accuracies = []
    test_accuracies = []
    best_epoch = []
    best_loss = []
    learning_rate = []
    dropout = []
    init_params = lasagne.layers.get_all_param_values(classifier)
    num_params = 0
    for param in init_params:
        num_params += param.size
    print('Number of parameters: ', num_params)
    loss_acc_plt_flag = True

    print("Starting training...")
    for l_rate in learning_rate_set:
        lr_shared.set_value(np.array(l_rate, dtype='float32'))
        for d_out in dropout_set:
            dropouts_shared.set_value(np.array(d_out, dtype='float32'))

            set_all_param_values(classifier, init_params)
            train_deep_with_aug(X_train, y_train, X_val, y_val, X_test, y_test, num_class, parameters,
                                num_epochs, val_accuracies, test_accuracies, best_epoch, classifier, batch_size,
                                test_batch_size, train_fn, test_fn, l_rate, d_out, loss_acc_plt_flag,
                                output_path, learning_rate, dropout,
                                best_loss)  # Amir, Kamran: Plot val test accuracy, train loss

            print('\nlearning rate: ', l_rate)
            print('dropouts: ', d_out)
            print('best epoch: ', best_epoch[-1])
            print('validation accuracies: ', val_accuracies[-1])
            print('test accuracies: ', test_accuracies[-1])
            print('\n------------------------------------------------------------------------------\n')

    print('\nlearning rate: ', learning_rate_set)
    print('dropouts: ', dropout_set)
    print('best epoch: ', best_epoch)
    print('validation accuracies: ', val_accuracies)
    print('test accuracies: ', test_accuracies)
    print('\n------------------------------------------------------------------------------\n')

    best_index = np.argmax(val_accuracies)
    print('learning rate: ', learning_rate[best_index])
    print('dropouts: ', dropout[best_index])
    print('best epoch: ', best_epoch[best_index])
    print('validation accuracies: ', val_accuracies[best_index])
    print('test accuracies: ', test_accuracies[best_index])

    # Final training with validation set as well
    set_all_param_values(classifier, parameters[best_index])
    lr_shared.set_value(np.array(learning_rate[best_index], dtype='float32'))
    dropouts_shared.set_value(np.array(dropout[best_index], dtype='float32'))

    train_deep_final_with_aug(X, y, X_test, y_test, num_class,
                              10000, batch_size, test_batch_size, train_fn, test_fn, learning_rate[best_index],
                              dropout[best_index], loss_acc_plt_flag, output_path, best_loss[best_index])


# output_path = '/results/dense_deep_AE/' + os.path.basename(__file__).split('.')[0] + '/' + time.strftime(
#     "%d-%m-%Y_") + time.strftime("%H:%M:%S") + '/'
# this_file_name = os.path.basename(__file__)
# create_result_dirs(output_path, this_file_name)
# sys.stdout = Logger(output_path)
#
# dataset = 'MNIST-full'
# X, y = load_dataset(dataset)
# X = (X - np.float32(127.5)) / np.float32(127.5)
# num_clusters = len(np.unique(y))
# feature_map_sizes = [X.shape[1], 64, 64, 64]
# dropouts = [0.1, 0.1, 0.1, 0]
# learning_rate = 1e-4
# num_epochs = 4000
# batch_size = 100
# # kernel_sizes = [[4, 4, 4],
# #                 [6, 6, 0],
# #                 [10, 0, 0]]
# # strides = [[2, 2, 2],
# #            [4, 4, 0],
# #            [8, 0, 0]]
# # paddings = [[1, 2, 1],
# #             [3, 2, 0],
# #             [3, 0, 0]]
# # dropouts = [0.1, 0.1, 0.1, 0.1]
# # learning_rate = 1e-4
# # num_epochs = 4000
# # batch_size = 100
# kernel_sizes = [4, 4, 4]
# strides = [2, 2, 2]
# paddings = [1, 2, 1]
# test_batch_size = 100
# # train_fn_reconstruction, train_fn_joint, val_fn, test_fn, classifier, loss, params_reconstruction, params_classification, prediction_clean, prediction_noisy, loss_reconstruction, loss_reconstruction_clean, loss_classification, classifier, encoder, decoder, input_var = build_dense_deep_CAE(
# #     width=X.shape[2], height=X.shape[3], num_class=num_clusters, learning_rate=1e-4,
# #     feature_map_sizes=feature_map_sizes,
# #     kernel_sizes=kernel_sizes,
# #     strides=strides, paddings=paddings,
# #     dropouts=dropouts, hyperparameters=[1, 0.1, 0.1], target_flag='soft',
# #     batch_norm_flag=True)
#
# train_fn, test_fn, classifier, loss, params, prediction_clean, prediction_noisy, loss_classification, classifier, last_layer, input_var = build_deep_CNN(
#     width=X.shape[2], height=X.shape[3], num_class=num_clusters, learning_rate=1e-4,
#     feature_map_sizes=feature_map_sizes,
#     kernel_sizes=kernel_sizes,
#     strides=strides, paddings=paddings,
#     dropouts=dropouts, hyperparameters=[1, 0.1, 0.1], target_flag='soft',
#     batch_norm_flag=True)
#
# # network = get_all_layers(decoder)
# # network.append(classifier)
#
# draw_to_file(classifier, output_path, list_flag=False)
# #
# train_MdA_val(dataset, input_var, X, y, train_fn_reconstruction, val_fn, num_clusters, encoder, decoder,
#               batch_size=batch_size, test_batch_size=test_batch_size,
#               num_epochs=num_epochs, learning_rate=learning_rate, verbose=2)

# ##############################
# # Clustering on MdA Features #
# ##############################
# y_pred, centroids = Clustering(dataset, X, y, input_var, encoder, num_clusters,
#                                test_batch_size=test_batch_size)
#
# classifier.W = theano.shared(centroids)
#
# train_RLC(dataset, X, y, input_var, train_fn_joint, decoder, prediction_noisy, encoder, num_clusters, y_pred,
#           loss_reconstruction, batch_size=batch_size,
#           test_batch_size=test_batch_size, num_epochs=num_epochs,
#           learning_rate=learning_rate,
#           centroids=centroids)


def train_deepDis(dataset_name, X, y, true_label, gold_label_idx, train_fn, test_fn,
                  batch_size=100, num_epochs=1000, num_feature=500, verbose=1):
    X = np.float32(X)
    if y.ndim == 1:
        y = np.int32(y)
        num_class = np.unique(y).shape[0]
    else:
        num_class = y.shape[0]
        y = np.float32(y.T)

    num_samples = y.shape[0]
    y_pred = np.zeros((num_samples, num_class))
    feat = np.zeros((num_samples, num_feature))

    for epoch in range(num_epochs):
        train_err = 0
        loss = 0
        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
            minibatch_input, minibatch_target, idx = batch
            minibatch_error = train_fn(minibatch_input, minibatch_target)
            train_err += minibatch_error[0]

            y_pred[idx, :], _ = test_fn(minibatch_input)

        if type(true_label) is np.ndarray:
            # acc = find_accuracy(y_pred[gold_label_idx, :].T, true_label)
            print("iteration: %d \t loss: %f \t acc: %f" % (epoch, train_err, acc))
        else:
            print("iteration: %d \t loss: %f " % (epoch, train_err))

    for batch in iterate_minibatches(X, y, batch_size, shuffle=False):
        minibatch_input, minibatch_target, idx = batch
        _, feat[idx, :] = test_fn(minibatch_input)

    return feat


def load_dicoms(pic_size=224):
    if os.path.isfile('datap.npy'):
        X = np.load('datap.npy')
        y = np.load('labelp.npy')
    else:
        data_dir = '../../Downloads/INBreast/AllDICOMs'
        patients = [file for file in os.listdir(data_dir) if file.endswith('.dcm')]
        labels_df = pd.read_excel('../../Downloads/INBreast/INbreast.xls', index_col=5)
        y = []
        X = np.zeros((len(patients), 1, pic_size, pic_size))
        for i in range(len(patients)):
            print(i)
            patient = patients[i]
            label = labels_df.get_value(float(patient.split('_')[0]), 'Bi-Rads')
            if label == '4a' or label == '4b' or label == '4c':
                label = 4
            if 0 < label < 4:
                y.append(0)
            elif 3 < label < 7:
                y.append(1)
            path = os.path.join(data_dir, patient)
            patient = dicom.read_file(path).pixel_array
            thresh = threshold_otsu(patient)
            binary = patient > thresh
            counter = 0
            for j in range(binary.shape[0]):
                if binary[j, :].sum() == 0:
                    patient = np.delete(patient, j - counter, 0)
                    counter += 1

            counter = 0
            for j in range(binary.shape[1]):
                if binary[:, j].sum() == 0:
                    patient = np.delete(patient, j - counter, 1)
                    counter += 1

            patient = np.asarray(patient, dtype='float32') / float(patient.max())

            patient = resize(patient, (pic_size, pic_size))
            patient = patient.reshape(-1, 1, pic_size, pic_size)
            X[i, 0] = patient

    np.save('datap.npy', X)
    np.save('labelp.npy', y)
    return X, y


def load_dataset_256():
    if os.path.isfile('data256.npy'):
        X = np.load('data256.npy')
        X_mass = np.load('masses256.npy')
        X_mass_mask = np.load('mass_masks256.npy')
        y = np.load('labels256.npy')
    else:
        path = 'Mammograms/'
        mass_path = 'Masses/'

        img = os.listdir(path)

        px = 256

        X = np.zeros((len(img), 1, px, px), dtype='float32')
        X_mass = np.zeros((len(img), 1, px, px), dtype='float32')
        X_mass_mask = np.zeros((len(img), 1, px, px), dtype='float32')

        y = np.zeros(len(img), dtype='int32')

        index = 0

        workbook = xlrd.open_workbook('INbreast.xls')

        worksheet = workbook.sheet_by_name('Sheet1')

        patient_ids = np.array(worksheet.col_values(5)[1:-2], dtype='int32')

        Birads = np.array(worksheet.col_values(7)[1:-2])

        Birads[Birads == '4a'] = 4.0
        Birads[Birads == '4b'] = 4.0
        Birads[Birads == '4c'] = 4.0

        Birads = np.genfromtxt(Birads, dtype='int32')

        for i in range(0, len(img)):
            print(i)

            if img[i].split('.')[-1] == 'jpeg':

                patient_id = int(img[i].split('.')[0])

                if Birads[patient_ids == patient_id] > 3:
                    label = 1
                else:
                    label = 0

                img_ = np.array(Image.open(path + img[i]))

                thresh = threshold_otsu(img_)
                binary = img_ > thresh
                mass_flag = os.path.isfile(mass_path + img[i].split('.')[0] + 'extract.' + img[i].split('.')[-1])
                if mass_flag:
                    mass_ = np.array(Image.open(mass_path + img[i].split('.')[0] + 'extract.' + img[i].split('.')[-1]))
                    mass_mask = np.array(
                        Image.open(mass_path + img[i].split('.')[0] + 'extracted_masks.' + img[i].split('.')[-1]))

                counter = 0
                for j in range(binary.shape[0]):
                    if binary[j, :].sum() == 0:
                        img_ = np.delete(img_, j - counter, 0)
                        if mass_flag:
                            mass_ = np.delete(mass_, j - counter, 0)
                            mass_mask = np.delete(mass_mask, j - counter, 0)
                        counter += 1

                counter = 0
                for j in range(binary.shape[1]):
                    if binary[:, j].sum() == 0:
                        img_ = np.delete(img_, j - counter, 1)
                        if mass_flag:
                            mass_ = np.delete(mass_, j - counter, 1)
                            mass_mask = np.delete(mass_mask, j - counter, 1)
                        counter += 1

                img_ = np.asarray(img_, dtype='float32') / 256.

                if mass_flag:
                    mass_ = np.asarray(mass_, dtype='float32') / 256.
                    mass_mask = np.asarray(mass_mask, dtype='float32') / 256.

                img_ = resize(img_, (px, px))
                img_ = img_.reshape(-1, 1, px, px)

                if mass_flag:
                    mass_ = resize(mass_, (px, px))
                    mass_ = mass_.reshape(-1, 1, px, px)
                    mass_mask = resize(mass_mask, (px, px))
                    mass_mask = mass_mask.reshape(-1, 1, px, px)

                if mass_flag:
                    X_mass[index] = mass_
                    X_mass_mask[index] = mass_mask

                X[index] = img_
                y[index] = label
                index += 1
        np.save('data256.npy', X)
        np.save('masses256.npy', X_mass)
        np.save('mass_masks256.npy', X_mass_mask)
        np.save('labels256.npy', y)
    return X, y, X_mass, X_mass_mask


def load_dataset_227():
    if os.path.isfile('data227.npy'):
        X = np.load('data227.npy')
        X_mass = np.load('masses227.npy')
        X_mass_mask = np.load('mass_masks227.npy')
        y = np.load('labels227.npy')
    else:
        path = 'Mammograms/'
        mass_path = 'Masses/'

        img = os.listdir(path)

        px = 227

        X = np.zeros((len(img), 1, px, px), dtype='float32')
        X_mass = np.zeros((len(img), 1, px, px), dtype='float32')
        X_mass_mask = np.zeros((len(img), 1, px, px), dtype='float32')

        y = np.zeros(len(img), dtype='int32')

        index = 0

        workbook = xlrd.open_workbook('INbreast.xls')

        worksheet = workbook.sheet_by_name('Sheet1')

        patient_ids = np.array(worksheet.col_values(5)[1:-2], dtype='int32')

        Birads = np.array(worksheet.col_values(7)[1:-2])

        Birads[Birads == '4a'] = 4.0
        Birads[Birads == '4b'] = 4.0
        Birads[Birads == '4c'] = 4.0

        Birads = np.genfromtxt(Birads, dtype='int32')

        for i in range(0, len(img)):
            print(i)

            if img[i].split('.')[-1] == 'jpeg':

                patient_id = int(img[i].split('.')[0])

                if Birads[patient_ids == patient_id] > 3:
                    label = 1
                else:
                    label = 0

                img_ = np.array(Image.open(path + img[i]))

                thresh = threshold_otsu(img_)
                binary = img_ > thresh
                mass_flag = os.path.isfile(mass_path + img[i].split('.')[0] + 'extract.' + img[i].split('.')[-1])
                if mass_flag:
                    mass_ = np.array(Image.open(mass_path + img[i].split('.')[0] + 'extract.' + img[i].split('.')[-1]))
                    mass_mask = np.array(
                        Image.open(mass_path + img[i].split('.')[0] + 'extracted_masks.' + img[i].split('.')[-1]))

                counter = 0
                for j in range(binary.shape[0]):
                    if binary[j, :].sum() == 0:
                        img_ = np.delete(img_, j - counter, 0)
                        if mass_flag:
                            mass_ = np.delete(mass_, j - counter, 0)
                            mass_mask = np.delete(mass_mask, j - counter, 0)
                        counter += 1

                counter = 0
                for j in range(binary.shape[1]):
                    if binary[:, j].sum() == 0:
                        img_ = np.delete(img_, j - counter, 1)
                        if mass_flag:
                            mass_ = np.delete(mass_, j - counter, 1)
                            mass_mask = np.delete(mass_mask, j - counter, 1)
                        counter += 1

                img_ = np.asarray(img_, dtype='float32') / 256.

                if mass_flag:
                    mass_ = np.asarray(mass_, dtype='float32') / 256.
                    mass_mask = np.asarray(mass_mask, dtype='float32') / 256.

                img_ = resize(img_, (px, px))
                img_ = img_.reshape(-1, 1, px, px)

                if mass_flag:
                    mass_ = resize(mass_, (px, px))
                    mass_ = mass_.reshape(-1, 1, px, px)
                    mass_mask = resize(mass_mask, (px, px))
                    mass_mask = mass_mask.reshape(-1, 1, px, px)

                if mass_flag:
                    X_mass[index] = mass_
                    X_mass_mask[index] = mass_mask

                X[index] = img_
                y[index] = label
                index += 1
        np.save('data227.npy', X)
        np.save('masses227.npy', X_mass)
        np.save('mass_masks227.npy', X_mass_mask)
        np.save('labels227.npy', y)
    return X, y, X_mass, X_mass_mask


def load_dataset_bak():
    if os.path.isfile('data.npy'):
        X = np.load('data.npy')
        X_mass = np.load('masses.npy')
        X_mass_mask = np.load('mass_masks.npy')
        y = np.load('labels.npy')
    else:
        path = 'Mammograms/'
        mass_path = 'Masses/'

        img = os.listdir(path)

        px = 224

        X = np.zeros((len(img), 1, px, px), dtype='float32')
        X_mass = np.zeros((len(img), 1, px, px), dtype='float32')
        X_mass_mask = np.zeros((len(img), 1, px, px), dtype='float32')

        y = np.zeros(len(img), dtype='int32')

        index = 0

        workbook = xlrd.open_workbook('INbreast.xls')

        worksheet = workbook.sheet_by_name('Sheet1')

        patient_ids = np.array(worksheet.col_values(5)[1:-2], dtype='int32')

        Birads = np.array(worksheet.col_values(7)[1:-2])

        Birads[Birads == '4a'] = 4.0
        Birads[Birads == '4b'] = 4.0
        Birads[Birads == '4c'] = 4.0

        Birads = np.genfromtxt(Birads, dtype='int32')

        for i in range(0, len(img)):
            print(i)

            if img[i].split('.')[-1] == 'jpeg':

                patient_id = int(img[i].split('.')[0])

                if Birads[patient_ids == patient_id] > 3:
                    label = 1
                else:
                    label = 0

                img_ = np.array(Image.open(path + img[i]))

                thresh = threshold_otsu(img_)
                binary = img_ > thresh
                mass_flag = os.path.isfile(mass_path + img[i].split('.')[0] + 'extract.' + img[i].split('.')[-1])
                if mass_flag:
                    mass_ = np.array(Image.open(mass_path + img[i].split('.')[0] + 'extract.' + img[i].split('.')[-1]))
                    mass_mask = np.array(
                        Image.open(mass_path + img[i].split('.')[0] + 'extracted_masks.' + img[i].split('.')[-1]))

                counter = 0
                for j in range(binary.shape[0]):
                    if binary[j, :].sum() == 0:
                        img_ = np.delete(img_, j - counter, 0)
                        if mass_flag:
                            mass_ = np.delete(mass_, j - counter, 0)
                            mass_mask = np.delete(mass_mask, j - counter, 0)
                        counter += 1

                counter = 0
                for j in range(binary.shape[1]):
                    if binary[:, j].sum() == 0:
                        img_ = np.delete(img_, j - counter, 1)
                        if mass_flag:
                            mass_ = np.delete(mass_, j - counter, 1)
                            mass_mask = np.delete(mass_mask, j - counter, 1)
                        counter += 1

                img_ = np.asarray(img_, dtype='float32') / 256.

                if mass_flag:
                    mass_ = np.asarray(mass_, dtype='float32') / 256.
                    mass_mask = np.asarray(mass_mask, dtype='float32') / 256.

                img_ = resize(img_, (px, px))
                img_ = img_.reshape(-1, 1, px, px)

                if mass_flag:
                    mass_ = resize(mass_, (px, px))
                    mass_ = mass_.reshape(-1, 1, px, px)
                    mass_mask = resize(mass_mask, (px, px))
                    mass_mask = mass_mask.reshape(-1, 1, px, px)

                if mass_flag:
                    X_mass[index] = mass_
                    X_mass_mask[index] = mass_mask

                X[index] = img_
                y[index] = label
                index += 1
        np.save('data.npy', X)
        np.save('masses.npy', X_mass)
        np.save('mass_masks.npy', X_mass_mask)
        np.save('labels.npy', y)
    return X, y, X_mass, X_mass_mask


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], excerpt


def getClassifierParam(fileName, bias):
    if os.path.exists(fileName):
        param = np.cast['float32'](np.load(fileName))
        return lasagne.utils.create_param(param, param.shape, name=fileName.split('/')[-1].split('.')[0])
    else:
        return (init, lasagne.init.Constant(0.))[bias]


def initNetwork(X, Y, config, pretrained_flag):
    alexNetModel = alexNet(config, X, pretrained_flag)
    # network = FlattenLayer(alexNetModel.outLayer)
    # network = DropoutLayer(alexNetModel.outLayer, p=config['prob_drop'], rescale=False)  # dropout
    wtFileName = config['weightsDir'] + 'W_5b.npy';
    bFileName = config['weightsDir'] + 'b_5b.npy'
    network = DenseLayer(alexNetModel.outLayer, num_units=2, W=getClassifierParam(wtFileName, False),
                         b=getClassifierParam(bFileName, True),
                         nonlinearity=lasagne.nonlinearities.softmax)  # if classifier weights are not present, init with random weights
    print(get_output_shape(network), 'ddd')

    regMult = [float(i) for i in config['regularize'].split(
        ',')]  # read off a line like :regularize: 0.1,0.1,0.1,0.1,0.1,0.1 from the config.yaml file
    layersRegMultiplier = {alexNetModel.layers[layerId]: regMult[layerId] for layerId in
                           range(len(alexNetModel.layers))}
    layersRegMultiplier[network] = regMult[-1]
    l2_penalty = regularize_layer_params_weighted(layersRegMultiplier, l2)

    prediction = get_output(network, deterministic=True)
    lossAll = lasagne.objectives.categorical_crossentropy(prediction, Y)  # loss function
    loss = lossAll.mean()
    # loss += l2_penalty

    # accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), Y), dtype=theano.config.floatX)
    # match = T.eq(T.argmax(prediction, axis=1), Y)
    prediction = get_output(network, X, deterministic=True)
    params = get_all_params(network, trainable=True)

    return [loss, params, prediction, network]


def build_CNN_bn(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                 dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], kernel_sizes=[5, 5],
                 strides=[2, 2],
                 paddings=[2, 2], hlayer_loss_param=0.1, targets=None, num_classes=2):
    # ENCODER
    l_e0 = DropoutLayer(
        InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                     filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                     nonlinearity=lasagne.nonlinearities.rectify, W=init)),
        p=dropouts[1]))

    l_e2 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                     filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                     nonlinearity=lasagne.nonlinearities.rectify, W=init)),
        p=dropouts[2]))

    l_e3 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e2, num_filters=feature_map_sizes[2], stride=(strides[2], strides[2]),
                     filter_size=(kernel_sizes[2], kernel_sizes[2]), pad=paddings[2],
                     nonlinearity=lasagne.nonlinearities.rectify, W=init)),
        p=dropouts[3]))

    l_e4 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e3, num_filters=feature_map_sizes[3], stride=(strides[3], strides[3]),
                     filter_size=(kernel_sizes[3], kernel_sizes[3]), pad=paddings[3],
                     nonlinearity=lasagne.nonlinearities.rectify, W=init)),
        p=dropouts[4]))

    l_e5 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e4, num_filters=feature_map_sizes[4], stride=(strides[4], strides[4]),
                     filter_size=(kernel_sizes[4], kernel_sizes[4]), pad=paddings[4],
                     nonlinearity=lasagne.nonlinearities.rectify, W=init)),
        p=dropouts[5]))

    classifier = DenseLayer(l_e5, num_units=num_classes, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    l2_penalty = regularize_network_params(classifier, l2)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    lossAll = lasagne.objectives.categorical_crossentropy(prediction_noisy, targets)  # loss function
    loss = lossAll.mean()

    params = get_all_params(classifier, trainable=True)

    return classifier, loss, params, prediction_clean, prediction_noisy


def build_DCGAN(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], kernel_sizes=[5, 5],
                strides=[2, 2],
                paddings=[2, 2], hlayer_loss_param=0.1, targets=None, num_classes=2):
    # ENCODER
    l_e0 = DropoutLayer(
        InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = DropoutLayer(
        (Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                     filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[1])

    l_e2 = DropoutLayer(
        (Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                     filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[2])

    l_e3 = DropoutLayer(
        (Conv2DLayer(l_e2, num_filters=feature_map_sizes[2], stride=(strides[2], strides[2]),
                     filter_size=(kernel_sizes[2], kernel_sizes[2]), pad=paddings[2],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[3])

    l_e4 = DropoutLayer(
        (Conv2DLayer(l_e3, num_filters=feature_map_sizes[3], stride=(strides[3], strides[3]),
                     filter_size=(kernel_sizes[3], kernel_sizes[3]), pad=paddings[3],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[4])

    l_e5 = DropoutLayer(
        (Conv2DLayer(l_e4, num_filters=feature_map_sizes[4], stride=(strides[4], strides[4]),
                     filter_size=(kernel_sizes[4], kernel_sizes[4]), pad=paddings[4],
                     nonlinearity=lasagne.nonlinearities.tanh,
                     W=init)),
        p=dropouts[5])

    classifier = DenseLayer(l_e5, num_units=num_classes, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    l2_penalty = regularize_network_params(classifier, l2)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy, targets)  # loss function
    loss = loss_classification.mean()

    params = get_all_params(classifier, trainable=True)

    return classifier, loss, params, prediction_clean, prediction_noisy


def build_DCGAN_bn(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                   dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], kernel_sizes=[5, 5],
                   strides=[2, 2],
                   paddings=[2, 2], hlayer_loss_param=0.1, targets=None, num_classes=2):
    # ENCODER
    l_e0 = DropoutLayer(
        InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                     filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[1]))

    l_e2 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                     filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[2]))

    l_e3 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e2, num_filters=feature_map_sizes[2], stride=(strides[2], strides[2]),
                     filter_size=(kernel_sizes[2], kernel_sizes[2]), pad=paddings[2],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[3]))

    l_e4 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e3, num_filters=feature_map_sizes[3], stride=(strides[3], strides[3]),
                     filter_size=(kernel_sizes[3], kernel_sizes[3]), pad=paddings[3],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[4]))

    l_e5 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e4, num_filters=feature_map_sizes[4], stride=(strides[4], strides[4]),
                     filter_size=(kernel_sizes[4], kernel_sizes[4]), pad=paddings[4],
                     nonlinearity=lasagne.nonlinearities.tanh,
                     W=init)),
        p=dropouts[5]))

    classifier = DenseLayer(l_e5, num_units=num_classes, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    l2_penalty = regularize_network_params(classifier, l2)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy, targets)  # loss function
    loss = loss_classification.mean()

    params = get_all_params(classifier, trainable=True)

    return classifier, loss, params, prediction_clean, prediction_noisy


def build_DCGAN_MdA(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                    dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], kernel_sizes=[5, 5],
                    strides=[2, 2],
                    paddings=[2, 2], hlayer_loss_param=0.1, recons_loss_param=1, targets=None, num_classes=2):
    # ENCODER
    l_e0 = DropoutLayer(
        InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = DropoutLayer(
        (Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                     filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[1])

    l_e2 = DropoutLayer(
        (Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                     filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[2])

    l_e3 = DropoutLayer(
        (Conv2DLayer(l_e2, num_filters=feature_map_sizes[2], stride=(strides[2], strides[2]),
                     filter_size=(kernel_sizes[2], kernel_sizes[2]), pad=paddings[2],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[3])

    l_e4 = DropoutLayer(
        (Conv2DLayer(l_e3, num_filters=feature_map_sizes[3], stride=(strides[3], strides[3]),
                     filter_size=(kernel_sizes[3], kernel_sizes[3]), pad=paddings[3],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify,
                     W=init)),
        p=dropouts[4])

    l_e5 = DropoutLayer(
        (Conv2DLayer(l_e4, num_filters=feature_map_sizes[4], stride=(strides[4], strides[4]),
                     filter_size=(kernel_sizes[4], kernel_sizes[4]), pad=paddings[4],
                     nonlinearity=lasagne.nonlinearities.tanh,
                     W=init)),
        p=dropouts[5])

    l_d4 = Deconv2DLayer(l_e5, num_filters=feature_map_sizes[3], stride=(strides[4], strides[4]),
                         filter_size=(kernel_sizes[4], kernel_sizes[4]), crop=paddings[4],
                         nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         W=init)

    l_d3 = Deconv2DLayer(l_d4, num_filters=feature_map_sizes[2], stride=(strides[3], strides[3]),
                         filter_size=(kernel_sizes[3], kernel_sizes[3]), crop=paddings[3],
                         nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         W=init)

    l_d2 = Deconv2DLayer(l_d3, num_filters=feature_map_sizes[1], stride=(strides[2], strides[2]),
                         filter_size=(kernel_sizes[2], kernel_sizes[2]), crop=paddings[2],
                         nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         W=init)

    l_d1 = Deconv2DLayer(l_d2, num_filters=feature_map_sizes[0], stride=(strides[1], strides[1]),
                         filter_size=(kernel_sizes[1], kernel_sizes[1]), crop=paddings[1],
                         nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         W=init)

    l_d0 = Deconv2DLayer(l_d1, num_filters=n_in[0], stride=(strides[0], strides[0]),
                         filter_size=(kernel_sizes[0], kernel_sizes[0]), crop=paddings[0],
                         nonlinearity=lasagne.nonlinearities.tanh,
                         W=init)

    # Loss
    tar0 = input_var
    tar1 = get_output(l_e1, deterministic=True)
    tar2 = get_output(l_e2, deterministic=True)
    tar3 = get_output(l_e3, deterministic=True)
    tar4 = get_output(l_e4, deterministic=True)
    rec4 = get_output(l_d4)
    rec3 = get_output(l_d3)
    rec2 = get_output(l_d2)
    rec1 = get_output(l_d1)
    rec0 = get_output(l_d0)
    rec4_clean = get_output(l_d4, deterministic=True)
    rec3_clean = get_output(l_d3, deterministic=True)
    rec2_clean = get_output(l_d2, deterministic=True)
    rec1_clean = get_output(l_d1, deterministic=True)
    rec0_clean = get_output(l_d0, deterministic=True)

    loss0 = lasagne.objectives.squared_error(rec0, tar0)
    loss1 = lasagne.objectives.squared_error(rec1, tar1) * hlayer_loss_param
    loss2 = lasagne.objectives.squared_error(rec2, tar2) * hlayer_loss_param
    loss3 = lasagne.objectives.squared_error(rec3, tar3) * hlayer_loss_param
    loss4 = lasagne.objectives.squared_error(rec4, tar4) * hlayer_loss_param

    loss0_clean = lasagne.objectives.squared_error(rec0_clean, tar0)
    loss1_clean = lasagne.objectives.squared_error(rec1_clean, tar1) * hlayer_loss_param
    loss2_clean = lasagne.objectives.squared_error(rec2_clean, tar2) * hlayer_loss_param
    loss3_clean = lasagne.objectives.squared_error(rec3_clean, tar3) * hlayer_loss_param
    loss4_clean = lasagne.objectives.squared_error(rec4_clean, tar4) * hlayer_loss_param

    loss_recons = loss0.mean() + loss1.mean() + loss2.mean() + loss3.mean() + loss4.mean()
    loss_recons_clean = loss0_clean.mean() + loss1_clean.mean() + loss2_clean.mean() + loss3_clean.mean() + loss4_clean.mean()

    classifier = DenseLayer(l_e5, num_units=num_classes, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    l2_penalty = regularize_network_params(classifier, l2)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy, targets).mean()  # loss function
    loss = loss_classification + recons_loss_param * loss_recons

    params = get_all_params(classifier, trainable=True)

    return classifier, loss, params, prediction_clean, prediction_noisy, loss_classification, loss_recons, loss_recons_clean


def build_CNN(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
              dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], kernel_sizes=[5, 5],
              strides=[2, 2],
              paddings=[2, 2], hlayer_loss_param=0.1, targets=None, num_classes=2):
    # ENCODER
    l_e0 = DropoutLayer(
        InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = DropoutLayer(
        (Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                     filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                     nonlinearity=lasagne.nonlinearities.elu, W=init)),
        p=dropouts[1])

    l_e2 = DropoutLayer(
        (Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                     filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                     nonlinearity=lasagne.nonlinearities.elu, W=init)),
        p=dropouts[2])

    l_e3 = DropoutLayer(
        (Conv2DLayer(l_e2, num_filters=feature_map_sizes[2], stride=(strides[2], strides[2]),
                     filter_size=(kernel_sizes[2], kernel_sizes[2]), pad=paddings[2],
                     nonlinearity=lasagne.nonlinearities.elu, W=init)),
        p=dropouts[3])

    l_e4 = DropoutLayer(
        (Conv2DLayer(l_e3, num_filters=feature_map_sizes[3], stride=(strides[3], strides[3]),
                     filter_size=(kernel_sizes[3], kernel_sizes[3]), pad=paddings[3],
                     nonlinearity=lasagne.nonlinearities.elu, W=init)),
        p=dropouts[4])

    l_e5 = DropoutLayer(
        (Conv2DLayer(l_e4, num_filters=feature_map_sizes[4], stride=(strides[4], strides[4]),
                     filter_size=(kernel_sizes[4], kernel_sizes[4]), pad=paddings[4],
                     nonlinearity=lasagne.nonlinearities.elu, W=init)),
        p=dropouts[5])

    classifier = DenseLayer(l_e5, num_units=num_classes, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    l2_penalty = regularize_network_params(classifier, l2)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    lossAll = lasagne.objectives.categorical_crossentropy(prediction_noisy, targets)  # loss function
    loss = lossAll.mean()

    params = get_all_params(classifier, trainable=True)

    return classifier, loss, params, prediction_clean, prediction_noisy


def build_MdA_bn(input_var=None, input_mass_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                 dropouts=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], kernel_sizes=[5, 5],
                 strides=[2, 2],
                 paddings=[2, 2], hlayer_loss_param=0.1, recons_loss_param=1, targets=None, num_classes=2):
    # ENCODER
    l_e0 = DropoutLayer(
        InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                     filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify, W=init)),
        p=dropouts[1]))

    l_e2 = batch_norm(DropoutLayer(
        (Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                     filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                     nonlinearity=lasagne.nonlinearities.leaky_rectify, W=init)),
        p=dropouts[2]))

    l_e2_flat = flatten(l_e2)

    l_e3 = batch_norm(DenseLayer(l_e2_flat, num_units=feature_map_sizes[2],
                                 nonlinearity=lasagne.nonlinearities.sigmoid))

    # DECODER
    l_d2_flat = batch_norm(DenseLayer(l_e3, num_units=l_e2_flat.output_shape[1],
                                      nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_d2 = batch_norm(reshape(l_d2_flat,
                              shape=[-1, l_e2.output_shape[1], l_e2.output_shape[2],
                                     l_e2.output_shape[3]]))

    l_d1 = batch_norm(
        Deconv2DLayer(l_d2, num_filters=feature_map_sizes[0], stride=(strides[1], strides[1]),
                      filter_size=(kernel_sizes[1], kernel_sizes[1]), crop=paddings[1],
                      nonlinearity=lasagne.nonlinearities.leaky_rectify))

    l_d0 = batch_norm(
        Deconv2DLayer(l_d1, num_filters=n_in[0], stride=(strides[0], strides[0]),
                      filter_size=(kernel_sizes[0], kernel_sizes[0]), crop=paddings[0],
                      nonlinearity=lasagne.nonlinearities.sigmoid))

    # Loss
    # Reconstruct mass only
    tar0 = input_mass_var
    tar1 = get_output(l_e1, deterministic=True)
    tar2 = get_output(l_e2, deterministic=True)
    rec2 = get_output(l_d2)
    rec1 = get_output(l_d1)
    rec0 = get_output(l_d0)
    rec2_clean = get_output(l_d2, deterministic=True)
    rec1_clean = get_output(l_d1, deterministic=True)
    rec0_clean = get_output(l_d0, deterministic=True)

    loss0 = lasagne.objectives.squared_error(rec0, tar0)
    loss1 = lasagne.objectives.squared_error(rec1, tar1) * hlayer_loss_param
    loss2 = lasagne.objectives.squared_error(rec2, tar2) * hlayer_loss_param

    loss0_clean = lasagne.objectives.squared_error(rec0_clean, tar0)
    loss1_clean = lasagne.objectives.squared_error(rec1_clean, tar1) * hlayer_loss_param
    loss2_clean = lasagne.objectives.squared_error(rec2_clean, tar2) * hlayer_loss_param

    loss_recons = loss0.mean() + loss1.mean() + loss2.mean()
    loss_recons_clean = loss0_clean.mean() + loss1_clean.mean() + loss2_clean.mean()

    classifier = DenseLayer(l_e3, num_units=num_classes, W=init,
                            b=lasagne.init.Constant(0),
                            nonlinearity=lasagne.nonlinearities.softmax)

    l2_penalty = regularize_network_params(classifier, l2)

    prediction_clean = get_output(classifier, deterministic=True)
    prediction_noisy = get_output(classifier, deterministic=False)
    loss_classification = lasagne.objectives.categorical_crossentropy(prediction_noisy, targets).mean()  # loss function
    loss = loss_classification + recons_loss_param * loss_recons

    params = get_all_params(classifier, trainable=True)

    return classifier, loss, params, prediction_clean, prediction_noisy, loss_recons, loss_recons_clean, loss_classification


def combinations(data, y, transformation_count, file_name):
    orig_idx = np.arange(len(y))
    trans_idx = list(itertools.product([0, 1], repeat=transformation_count))
    for i in range(trans_idx.__len__()):
        if i == 0:
            indices = orig_idx
            labels = y.copy()
        else:
            indices = np.concatenate((indices, orig_idx), axis=0)
            labels = np.concatenate((labels, y), axis=0)
    if os.path.isfile(file_name):
        X = np.load(file_name)
    else:
        # trans_idx = list(itertools.product([0, 1], repeat=transformation_count))
        for i in range(trans_idx.__len__()):
            print(i)
            for j in range(len(y)):
                temp = data[j][0]
                # Flip horizantally
                if trans_idx[i][0]:
                    temp = np.fliplr(temp)
                # Shift by 1/10th of the image size horizantally
                if trans_idx[i][1]:
                    temp = np.roll(temp, data.shape[2] / 10, axis=0)
                # Shift by 1/10th of the image size vertically
                if trans_idx[i][2]:
                    temp = np.roll(temp, data.shape[3] / 10, axis=1)
                # Rotate by 45 degrees
                if trans_idx[i][3]:
                    temp = rotate(temp, angle=45)
                if i == 0 and j == 0:
                    X = temp.reshape(1, 1, temp.shape[0], temp.shape[1])
                else:
                    temp = temp.reshape(1, 1, temp.shape[0], temp.shape[1])
                    X = np.concatenate((X, temp), axis=0)

            if i == 0:
                labels = y.copy()
            else:
                labels = np.concatenate((labels, y), axis=0)
        np.save(file_name, X)
    return X, labels, indices


def cross_val_idx(num_data=410, k_fold=5, shuffle_flag=0):
    orig_idx = np.arange(num_data, dtype='int32')
    train_indices = np.zeros((k_fold, num_data - (2 * (num_data / k_fold))))
    val_indices = np.zeros((k_fold, num_data / k_fold))
    test_indices = np.zeros((k_fold, num_data / k_fold))
    if shuffle_flag:
        np.random.shuffle(orig_idx)

    for k in range(k_fold):
        temp_idx = np.copy(orig_idx)
        test_idx = temp_idx[k * (num_data / k_fold):(k + 1) * (num_data / k_fold)]
        temp_idx = np.delete(temp_idx, np.arange(k * (num_data / k_fold), (k + 1) * (num_data / k_fold)))
        if k == k_fold - 1:
            val_idx = temp_idx[0:(num_data / k_fold)]
            temp_idx = np.delete(temp_idx, np.arange(0, (num_data / k_fold)))
        else:
            val_idx = temp_idx[k * (num_data / k_fold):(k + 1) * (num_data / k_fold)]
            temp_idx = np.delete(temp_idx, np.arange(k * (num_data / k_fold), (k + 1) * (num_data / k_fold)))
        train_idx = temp_idx

        train_indices[k] = train_idx
        val_indices[k] = val_idx
        test_indices[k] = test_idx

    return np.array(train_indices, dtype='int32'), np.array(val_indices, dtype='int32'), np.array(test_indices,
                                                                                                  dtype='int32')


def train_forward_builtin_aug(X_train, X_train_masses, X_train_mass_masks, y_train, X_val, y_val, X_test, y_test,
                              parameters,
                              num_epochs, val_accuracies, val_roc_aucs,
                              test_accuracies, test_roc_aucs, network, minibatch_size, train_fn, test_fn, lr,
                              num_classes, balance_flag,
                              occlusion_flag):
    y_prob_val = np.zeros((X_val.shape[0], 2))
    y_prob_test = np.zeros((X_test.shape[0], 2))
    best_roc_auc = 0
    best_acc = 0
    parameters.append(get_all_param_values(network))
    for epoch in range(num_epochs):

        # lr.set_value(lr.get_value() * (np.float32(num_epochs - epoch) / np.float32(num_epochs)))

        # In each epoch, we do a full pass over the training data:

        train_err = 0

        train_batches = 0

        start_time = time.time()

        occlusion_size = 50

        for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):
            inputs, targets, idx = batch
            input_masses = X_train_masses[idx]
            input_mass_masks = X_train_mass_masks[idx]

            # Random occlusion
            if occlusion_flag:
                for i in range(len(targets)):
                    # flip horizantally
                    if np.random.randint(2):
                        inputs[i, 0] = np.fliplr(inputs[i, 0])
                        input_mass_masks[i, 0] = np.fliplr(input_mass_masks[i, 0])
                    # Shift within 1/10th of the image size horizantally
                    h_shift = np.random.randint(inputs.shape[2] / 10)
                    inputs[i, 0] = np.roll(inputs[i, 0], h_shift, axis=0)
                    input_mass_masks[i, 0] = np.roll(input_mass_masks[i, 0], h_shift, axis=0)
                    # Shift within 1/10th of the image size vertically
                    v_shift = np.random.randint(inputs.shape[3] / 10)
                    inputs[i, 0] = np.roll(inputs[i, 0], v_shift, axis=1)
                    input_mass_masks[i, 0] = np.roll(input_mass_masks[i, 0], v_shift, axis=1)
                    # Rotate within 45 degrees
                    rotation = np.random.randint(45)
                    inputs[i, 0] = rotate(inputs[i, 0], angle=rotation)
                    input_mass_masks[i, 0] = rotate(input_mass_masks[i, 0], angle=rotation)
                    occlusion_mask = np.zeros(inputs[i, 0].shape)
                    redo_occ_flag = 1
                    while redo_occ_flag:
                        height = np.random.randint(inputs.shape[2] - occlusion_size)
                        width = np.random.randint(inputs.shape[3] - occlusion_size)
                        occlusion_mask[height:height + occlusion_size, width:width + occlusion_size] = 1
                        combination = occlusion_mask + input_mass_masks[i]
                        combination[np.where(combination > 1)] = 1
                        if (float(np.sum(combination) - np.sum(input_mass_masks[i])) /
                                np.sum(occlusion_mask)) > 0.5:
                            redo_occ_flag = 0
                    inputs[i, 0][height:height + occlusion_size, width:width + occlusion_size] = 0

            err = train_fn(inputs, targets)

            train_err += err

            train_batches += 1

        # And a full pass over the validation data:

        val_err = 0

        val_pred = 0

        val_batches = 0

        for batch in iterate_minibatches(X_val, y_val, minibatch_size, shuffle=False):
            inputs, targets, idx = batch

            err, pred = test_fn(inputs, targets)

            val_err += err

            y_prob_val[idx] = pred

            val_batches += 1

        pred_y_val = np.argmax(y_prob_val, axis=1)

        # Then we print the results for this epoch:
        learning_rate = float(lr.get_value())

        accuracy = accuracy_score(pred_y_val, y_val)
        roc_auc_val = ROC_AUC(y_val, pred_y_val)

        if roc_auc_val > best_roc_auc:
            print('New best here!')
            best_roc_auc = roc_auc_val
            best_acc = accuracy
            parameters.pop()
            parameters.append(get_all_param_values(network))

        # print(
        #     "Epoch {} of {} took {:.3f}s\tLearning rate: {:.6f}\ttraining loss:\t\t{:.6f}\tvalidation loss:\t\t{:.6f}\tvalidation accuracy:\t\t{:.2f}\tvalidation AUC:\t\t{:.2f}\tnumber of ones:\t\t{:d}".format(
        #
        #         epoch + 1, num_epochs, time.time() - start_time, learning_rate, train_err / train_batches,
        #         val_err / val_batches, accuracy * 100, roc_auc * 100, sum(pred_y_val)))
        #

        test_err = 0

        test_batches = 0

        for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
            inputs, targets, idx = batch

            err, pred = test_fn(inputs, targets)

            test_err += err

            y_prob_test[idx] = pred

            test_batches += 1

        pred_y_test = np.argmax(y_prob_test, axis=1)

        acc_test = accuracy_score(pred_y_test, y_test)

        roc_auc_test = ROC_AUC(y_test, pred_y_test)

        # Then we print the results for this epoch:

        print(
            "Epoch {} of {}\ttraining loss:\t{:.6f}\tvalidation accuracy:\t{:.2f}\tvalidation AUC: \t{:.2f}\ttest accuracy:\t{:.2f}\ttest AUC:\t{:.2f} \tnumber of ones in val:\t{:d}\tLearning rate: {:.1e} took {:.3f}s".format(
                epoch + 1, num_epochs, train_err / train_batches, accuracy * 100, roc_auc_val * 100, acc_test * 100,
                roc_auc_test * 100, sum(pred_y_val), learning_rate, time.time() - start_time))

        # print("  \ttest accuracy:\t\t{:.2f}\ttest AUC:\t\t{:.2f}\tnumber of ones:\t\t{:d}".format(test_err / test_batches,
        #                                                                                      acc_test * 100,
        #                                                                                      roc_auc * 100, sum(pred_y_test)))
    test_err = 0

    test_batches = 0

    set_all_param_values(network, parameters[-1])

    for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
        inputs, targets, idx = batch

        err, pred = test_fn(inputs, targets)

        test_err += err

        y_prob_test[idx] = pred

        test_batches += 1

    pred_y = np.argmax(y_prob_test, axis=1)

    acc_test = accuracy_score(pred_y, y_test)

    roc_auc = ROC_AUC(y_test, pred_y)

    # Then we print the results for this epoch:

    print("  test loss:\t\t{:.6f}\ttest accuracy:\t\t{:.2f}\ttest AUC:\t\t{:.2f}".format(test_err / test_batches,
                                                                                         acc_test * 100, roc_auc * 100))

    test_accuracies.append(acc_test)
    test_roc_aucs.append(roc_auc)
    val_accuracies.append(best_acc)
    val_roc_aucs.append(best_roc_auc)


def train_forward(X_train, X_train_masses, X_train_mass_masks, y_train, X_val, y_val, X_test, y_test, parameters,
                  num_epochs, val_accuracies, val_roc_aucs,
                  test_accuracies, test_roc_aucs, network, minibatch_size, train_fn, test_fn, lr, num_classes,
                  balance_flag,
                  occlusion_flag):
    y_prob_val = np.zeros((X_val.shape[0], 2))
    y_prob_test = np.zeros((X_test.shape[0], 2))
    best_roc_auc = 0
    best_acc = 0
    parameters.append(get_all_param_values(network))
    for epoch in range(num_epochs):

        # lr.set_value(lr.get_value() * (np.float32(num_epochs - epoch) / np.float32(num_epochs)))

        # In each epoch, we do a full pass over the training data:

        train_err = 0

        train_batches = 0

        start_time = time.time()

        occlusion_size = 50

        for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):
            inputs, targets, idx = batch
            input_masses = X_train_masses[idx]
            input_mass_masks = X_train_mass_masks[idx]

            # Random occlusion
            if occlusion_flag:
                for i in range(len(targets)):
                    occlusion_mask = np.zeros(inputs[i, 0].shape)
                    if np.random.randint(2):
                        redo_occ_flag = 1
                        while redo_occ_flag:
                            height = np.random.randint(inputs.shape[2] - occlusion_size)
                            width = np.random.randint(inputs.shape[3] - occlusion_size)
                            occlusion_mask[height:height + occlusion_size, width:width + occlusion_size] = 1
                            combination = occlusion_mask + input_mass_masks[i]
                            combination[np.where(combination > 1)] = 1
                            if (float(np.sum(combination) - np.sum(input_mass_masks[i])) /
                                    np.sum(occlusion_mask)) > 0.5:
                                redo_occ_flag = 0
                        inputs[i, 0][height:height + occlusion_size, width:width + occlusion_size] = 0

            err = train_fn(inputs, targets)

            train_err += err

            train_batches += 1

        # And a full pass over the validation data:

        val_err = 0

        val_pred = 0

        val_batches = 0

        for batch in iterate_minibatches(X_val, y_val, minibatch_size, shuffle=False):
            inputs, targets, idx = batch

            err, pred = test_fn(inputs, targets)

            val_err += err

            y_prob_val[idx] = pred

            val_batches += 1

        pred_y_val = np.argmax(y_prob_val, axis=1)

        # Then we print the results for this epoch:
        learning_rate = float(lr.get_value())

        accuracy = accuracy_score(pred_y_val, y_val)
        roc_auc_val = ROC_AUC(y_val, pred_y_val)

        if roc_auc_val > best_roc_auc:
            print('New best here!')
            best_roc_auc = roc_auc_val
            best_acc = accuracy
            parameters.pop()
            parameters.append(get_all_param_values(network))

        # print(
        #     "Epoch {} of {} took {:.3f}s\tLearning rate: {:.6f}\ttraining loss:\t\t{:.6f}\tvalidation loss:\t\t{:.6f}\tvalidation accuracy:\t\t{:.2f}\tvalidation AUC:\t\t{:.2f}\tnumber of ones:\t\t{:d}".format(
        #
        #         epoch + 1, num_epochs, time.time() - start_time, learning_rate, train_err / train_batches,
        #         val_err / val_batches, accuracy * 100, roc_auc * 100, sum(pred_y_val)))
        #

        test_err = 0

        test_batches = 0

        for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
            inputs, targets, idx = batch

            err, pred = test_fn(inputs, targets)

            test_err += err

            y_prob_test[idx] = pred

            test_batches += 1

        pred_y_test = np.argmax(y_prob_test, axis=1)

        acc_test = accuracy_score(pred_y_test, y_test)

        roc_auc_test = ROC_AUC(y_test, pred_y_test)

        # Then we print the results for this epoch:

        print(
            "Epoch {} of {}\ttraining loss:\t{:.6f}\tvalidation accuracy:\t{:.2f}\tvalidation AUC: \t{:.2f}\ttest accuracy:\t{:.2f}\ttest AUC:\t{:.2f} \tnumber of ones in val:\t{:d}\tLearning rate: {:.8f} took {:.3f}s".format(
                epoch + 1, num_epochs, train_err / train_batches, accuracy * 100, roc_auc_val * 100, acc_test * 100,
                roc_auc_test * 100, sum(pred_y_val), learning_rate, time.time() - start_time))

        # print("  \ttest accuracy:\t\t{:.2f}\ttest AUC:\t\t{:.2f}\tnumber of ones:\t\t{:d}".format(test_err / test_batches,
        #                                                                                      acc_test * 100,
        #                                                                                      roc_auc * 100, sum(pred_y_test)))
    test_err = 0

    test_batches = 0

    set_all_param_values(network, parameters[-1])

    for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
        inputs, targets, idx = batch

        err, pred = test_fn(inputs, targets)

        test_err += err

        y_prob_test[idx] = pred

        test_batches += 1

    pred_y = np.argmax(y_prob_test, axis=1)

    acc_test = accuracy_score(pred_y, y_test)

    roc_auc = ROC_AUC(y_test, pred_y)

    # Then we print the results for this epoch:

    print("  test loss:\t\t{:.6f}\ttest accuracy:\t\t{:.2f}\ttest AUC:\t\t{:.2f}".format(test_err / test_batches,
                                                                                         acc_test * 100, roc_auc * 100))

    test_accuracies.append(acc_test)
    test_roc_aucs.append(roc_auc)
    val_accuracies.append(best_acc)
    val_roc_aucs.append(best_roc_auc)


def train(X_train, X_train_masses, X_train_mass_masks, y_train, X_val, y_val, X_test, y_test, parameters, num_epochs,
          val_accuracies, val_roc_aucs,
          test_accuracies, test_roc_aucs, network, minibatch_size, train_fn, test_fn, lr, num_classes, balance_flag,
          occlusion_flag):
    y_prob_val = np.zeros((X_val.shape[0], 2))
    y_prob_test = np.zeros((X_test.shape[0], 2))
    best_roc_auc = 0
    best_acc = 0
    parameters.append(get_all_param_values(network))
    for epoch in range(num_epochs):

        # lr.set_value(lr.get_value() * (np.float32(num_epochs - epoch) / np.float32(num_epochs)))

        # In each epoch, we do a full pass over the training data:

        train_err = 0
        train_err_recons = 0
        train_err_classification = 0

        train_batches = 0

        start_time = time.time()

        occlusion_size = 50

        for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):
            inputs, targets, idx = batch
            input_masses = X_train_masses[idx]
            input_mass_masks = X_train_mass_masks[idx]

            # Random occlusion
            if occlusion_flag:
                for i in range(len(targets)):
                    occlusion_mask = np.zeros(inputs[i, 0].shape)
                    if np.random.randint(2):
                        redo_occ_flag = 1
                        while redo_occ_flag:
                            height = np.random.randint(inputs.shape[2] - occlusion_size)
                            width = np.random.randint(inputs.shape[3] - occlusion_size)
                            occlusion_mask[height:height + occlusion_size, width:width + occlusion_size] = 1
                            combination = occlusion_mask + input_mass_masks
                            combination[np.where(combination > 1)] = 1
                            if (float(sum(sum(combination)) - sum(sum(input_mass_masks))) / sum(
                                    sum(occlusion_mask))) > 0.5:
                                redo_occ_flag = 0
                        inputs[i, 0][height:height + occlusion_size, width:width + occlusion_size] = 0

            err, err_recons, err_classification = train_fn(inputs, input_masses, targets)

            train_err += err
            train_err_recons += err_recons
            train_err_classification += err_classification

            train_batches += 1

        # And a full pass over the validation data:

        val_err = 0

        val_pred = 0

        val_batches = 0

        for batch in iterate_minibatches(X_val, y_val, minibatch_size, shuffle=False):
            inputs, targets, idx = batch

            err, pred = test_fn(inputs, targets)

            val_err += err

            y_prob_val[idx] = pred

            val_batches += 1

        pred_y_val = np.argmax(y_prob_val, axis=1)

        # Then we print the results for this epoch:
        learning_rate = float(lr.get_value())

        accuracy = accuracy_score(pred_y_val, y_val)
        roc_auc_val = ROC_AUC(y_val, pred_y_val, num_classes)

        if roc_auc_val > best_roc_auc:
            print('New best here!')
            best_roc_auc = roc_auc_val
            best_acc = accuracy
            parameters.pop()
            parameters.append(get_all_param_values(network))

        # print(
        #     "Epoch {} of {} took {:.3f}s\tLearning rate: {:.6f}\ttraining loss:\t\t{:.6f}\tvalidation loss:\t\t{:.6f}\tvalidation accuracy:\t\t{:.2f}\tvalidation AUC:\t\t{:.2f}\tnumber of ones:\t\t{:d}".format(
        #
        #         epoch + 1, num_epochs, time.time() - start_time, learning_rate, train_err / train_batches,
        #         val_err / val_batches, accuracy * 100, roc_auc * 100, sum(pred_y_val)))
        #

        test_err = 0

        test_batches = 0

        for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
            inputs, targets, idx = batch

            err, pred = test_fn(inputs, targets)

            test_err += err

            y_prob_test[idx] = pred

            test_batches += 1

        pred_y_test = np.argmax(y_prob_test, axis=1)

        acc_test = accuracy_score(pred_y_test, y_test)

        roc_auc_test = ROC_AUC(y_test, pred_y_test, num_classes)

        # Then we print the results for this epoch:

        print(
            "Epoch {} of {}\ttraining loss:\t{:.6f}\tvalidation accuracy:\t{:.2f}\tvalidation AUC: \t{:.2f}\ttest accuracy:\t{:.2f}\ttest AUC:\t{:.2f} \tnumber of ones in val:\t{:d}\tLearning rate: {:.6f} took {:.3f}s".format(
                epoch + 1, num_epochs, train_err / train_batches, accuracy * 100, roc_auc_val * 100, acc_test * 100,
                roc_auc_test * 100, sum(pred_y_val), learning_rate, time.time() - start_time))

        # print("  \ttest accuracy:\t\t{:.2f}\ttest AUC:\t\t{:.2f}\tnumber of ones:\t\t{:d}".format(test_err / test_batches,
        #                                                                                      acc_test * 100,
        #                                                                                      roc_auc * 100, sum(pred_y_test)))
    test_err = 0

    test_batches = 0

    set_all_param_values(network, parameters[-1])

    for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
        inputs, targets, idx = batch

        err, pred = test_fn(inputs, targets)

        test_err += err

        y_prob_test[idx] = pred

        test_batches += 1

    pred_y = np.argmax(y_prob_test, axis=1)

    acc_test = accuracy_score(pred_y, y_test)

    roc_auc = ROC_AUC(y_test, pred_y, num_classes)

    # Then we print the results for this epoch:

    print("  test loss:\t\t{:.6f}\ttest accuracy:\t\t{:.2f}\ttest AUC:\t\t{:.2f}".format(test_err / test_batches,
                                                                                         acc_test * 100, roc_auc * 100))

    test_accuracies.append(acc_test)
    test_roc_aucs.append(roc_auc)
    val_accuracies.append(best_acc)
    val_roc_aucs.append(best_roc_auc)


def ROC_AUC(y, pred_y):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y, pred_y)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def balancing(X, X_masses, X_mass_masks, y, balance_flag):
    if balance_flag:
        num_positives = sum(y)
        num_negatives = len(y) - num_positives
        num_positives_temp = num_positives
        num_negatives_temp = num_negatives
        inputs = X.copy()
        input_masses = X_masses.copy()
        input_mass_masks = X_mass_masks.copy()
        targets = y.copy()
        while inputs.shape[0] != num_negatives * 2:
            if num_negatives_temp - num_positives_temp > num_positives_temp:
                inputs = np.concatenate((inputs, X[np.where(y == 1)]), axis=0)
                input_masses = np.concatenate((input_masses, X_masses[np.where(y == 1)]), axis=0)
                input_mass_masks = np.concatenate((input_mass_masks, X_mass_masks[np.where(y == 1)]), axis=0)
                targets = np.concatenate((targets, np.ones(np.where(y == 1)[0].shape, dtype='int32')), axis=0)
            else:
                inputs = np.concatenate((inputs, X[
                    np.where(targets == 1)[0][
                        np.random.randint(num_positives, size=num_negatives_temp - num_positives_temp)]]), axis=0)
                input_masses = np.concatenate((input_masses, X_masses[
                    np.where(targets == 1)[0][
                        np.random.randint(num_positives, size=num_negatives_temp - num_positives_temp)]]), axis=0)
                input_mass_masks = np.concatenate((input_mass_masks, X_mass_masks[
                    np.where(targets == 1)[0][
                        np.random.randint(num_positives, size=num_negatives_temp - num_positives_temp)]]), axis=0)
                targets = np.concatenate((targets, np.ones((num_negatives_temp - num_positives_temp), dtype='int32')),
                                         axis=0)
            num_positives_temp = sum(targets)
            num_negatives_temp = len(inputs) - num_positives_temp

        return inputs, input_masses, input_mass_masks, targets
    return X, X_masses, X_mass_masks, y
