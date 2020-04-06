"""
	DBSCAN clustering
	Also, implementation of a greedy seach of optimal hyperparameters of DBSCAN for face clustering.

	author: Ricardo Kleinlein
	date: 02/2020

	Usage:
		python grid_search.py <program-csv>

	Options:
		--output-dir	Directory to save results in
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""

import os
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from arguments import DbscanArgs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def load_db(db_path):
    """Load a pd.Dataframe of detections."""
    return pd.read_csv(db_path)


def load_embeddings(db):
    """Load a list of vector embeddings from a
    pd.Dataframe

    Args:
        db (pd.DataFrame): DataFrame object

    Return:
        a list of vector embeddings in np.ndarray form
        a list of bouding boxes sizes (np.ndarray)
    """
    size = db['size'].values
    emb = db['embedding'].values
    emb = [np.load(i).flatten() for i in emb]
    return emb, size


def size_filter(db, threshold):
    """Filter out samples under threshold.

    Args:
        db (pd.DataFrame): DataFrame object
        threshold (int): Minimal size to accept

    Return:
        a pd.DataFrame of samples that meet the condition
    """
    idx2rm = []
    for i, x in db.iterrows():
        if x['size'] < threshold:
            idx2rm.append(i)

    print("Numero de imagenes: ",len(db)-len(idx2rm)," de ", len(db))
    return db.drop(idx2rm)


def hist_face_sizes(X, measure, output_dir):
    """Save a histogram depicting the face sizes.

    Args:
        X (float): List of face sizes
        measure (str): Perimeter or area
        output_dir (str): Directory to save in
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.clf()
    plt.hist(X, bins=100)
    plt.xlabel('Face bounding box')
    plt.ylabel('Frequency')
    plt.savefig(join(output_dir, 'face_' + measure + '.png'))


def dbscan_(X, eps, min_samples, metric='euclidean'):
    """DBSCAN clustering for a set of parameters over the
    sampels X.

    Args:
        X (float): Feature sample vectors
        eps (float): Epsilon hparam
        min_samples (int): Min-samples hparam
        metric (str, optional): distance metric [default: euclidean]

    Return:
        np.ndarray of labels for each samples, with
        noisy samples given a `-1`
    """
    f = DBSCAN(eps=eps,
               min_samples=min_samples,
               metric=metric)
    f.fit(X)
    return f.labels_, f.core_sample_indices_


def export(path, data):
    """Export data to external csv file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, np.ndarray):
        np.save(path, data)
    elif isinstance(data, pd.DataFrame):
        data.to_csv(path + '.csv')
    elif hasattr(data, '__dict__'):
        with open(path, 'w+') as f:
            for k, v in zip(data.__dict__.keys(), data.__dict__.values()):
                f.write(k + ': ' + str(v) + '\n')
    else:
        raise IOError('Invalid data format')


def nearestneighbors(X, n, metric='euclidean'):
    """Compute the distance to the n-th neighbour in an array
    of feature samples X.

    Args:
        X (float): Array of feature samples.
        n (int): N-th neighbor to consider.
        metric (str): DIstance measure [default: euclidean]

    Return:
        An np.ndarray of distances up to the n-th neighbors
    """
    nn = NearestNeighbors(n_neighbors=n,
                          metric=metric,
                          n_jobs=-1)
    nbrs = nn.fit(X)
    dist, _ = nbrs.kneighbors(X)
    sort_dist = np.sort(dist, axis=0)[:, 1:]
    return sort_dist


def get_n_noise_samples(labels):
    return list(labels).count(-1)


def get_number_clusters(labels):
    return len(set(labels)) - (1 if -1 in labels else 0)


def measure_silhouette(X, labels, metric, with_noise=True):
    """Clean up noise samples and compute Silhouette score.
    Since DBSCAN assigns a -1 label to those samples that are not attached to any cluster, we should dismiss those ones from the point of view of validation.

    Args:
        X (float): Feature samples
        labels (int): Prediction labels. -1 denotes noise
        metric (str): Distance measure
        with_noise (bool, optional): Account for -1 tags

    Return:
        a float, the average Silhouette score of the clean samples.
    """
    if get_number_clusters(labels) < 2:
        return -1
    if -1 in labels:
        if with_noise:
            return silhouette_score(X, labels, metric=metric)
        else:
            idx2keep = []
            for i, x in enumerate(labels):
                if x != -1:
                    idx2keep.append(i)
            X = np.array([X[i] for i in idx2keep])
            labels = labels[idx2keep]
    return silhouette_score(X, labels, metric=metric)


def _eps_search(X, eps, min_samples, metric, quiet=False):
    """Epsilon-based DBSCAN protocol.

    Args:
        X (float): Vector embeddings
        eps (float): Eps value
        min_samples (int): Minimum min_samples
        metric (str): Pairwise distance to use
        quiet (bool, optional): Whethet to display info

    Return:
        a np.ndarray of labels according to the final cluster
        a dict of measures representing the experiment
    """
    score = -1
    num_cluster = 0
    num_noise = 0

    pred, core_ = dbscan_(X, eps, min_samples, metric=metric)
    try:
        score_sil = measure_silhouette(X, pred, metric, True)
        score = score_sil
        num_cluster = get_number_clusters(pred)
        num_noise = get_n_noise_samples(pred) / len(pred)
    except KeyboardInterrupt:
        raise KeyboardInterrupt

    if not quiet:
        print('> Eval configuration - eps = {:.2f}'.format(eps))

    vals = {'score': score,
            'num_clusters': num_cluster,
            'noise': num_noise}

    return pred, vals


def _status(pred):
    """Given a list of predictions, find the cluster sparse representation.
    """
    n = len(pred)
    pred_copy = pred.copy()
    cluster_groups = []
    for i in range(n):
        within_group = []
        for j in range(n):
            if pred[i] != -1 and pred[i] == pred[j]:
                within_group.append(j)
        cluster_groups.append(within_group)
    return [x for x in np.unique(cluster_groups) if x != []]


def compare(status, old_pred):
    """How many new and merged groups."""
    num_cl = len(status)
    new_clusters = 0
    merged_clusters = 0

    for i in range(num_cl):
        group = status[i]
        # print('Group: ', group)

        were_noise = True
        has_merged = False

        for k in range(len(group)):
            sample = group[k]
            old_tag = old_pred[sample]
            # print('Sample {:d}, old tag: {:d}'.format(sample, int(old_tag)))
            if old_tag != -1:
                were_noise = False
                if len(np.unique(old_pred[group])) > 1:
                    has_merged = True

        if were_noise:
            new_clusters += 1
        if has_merged:
            merged_clusters += len(np.unique(old_pred[group])) - 1

    return new_clusters, merged_clusters


def save_clusters(db, savepath, time_req):
    """Prepares image mosaics of the clusters and computes their centroids and stddev.

    Args:
        db (pd.DataFrame): Data with full info
        dirname (str): Root output directory
        time_req (int): Minimum time to be considered participant
    """
    # try:
    #     all_id = np.unique(db['pred_labels'])  # Includes noise tag
    # except:
    print('fallo de unique')
    import pdb
    pdb.set_trace()
    all_id = []
    num_id = [0 for _ in range(50)]
    for line in db['pred_labels']:
        if all_id.count(line)==0:
            all_id.append(line)
        num_id[line+1]+=1
    pdb.set_trace()


    for iddty in all_id:
        data = db.loc[db['pred_labels'] == iddty]
        if len(data) >= time_req and iddty != -1:
            id_path = join(savepath, 'id_' + str(iddty))
            os.makedirs(id_path, exist_ok=True)

            data_vector, size = load_embeddings(data)
            centroid = np.mean(data_vector, axis=0)
            std = np.std(data_vector, axis=0)
            cov = np.cov(np.array(data_vector).T)

            print(all_id)
            print(cov)

            inv_cov = np.linalg.inv(cov)
            export(path=join(id_path, 'centroid'),
                   data=centroid)
            export(path=join(id_path, 'std'),
                   data=std)
            export(path=join(id_path, 'covmat'),
                   data=cov)
            export(path=join(id_path, 'inv_covmat'),
                   data=inv_cov)

            imgs = data['img'].values
            for img_path in imgs:
                img = PIL.Image.open(img_path)
                img_name = img_path.split('/')[-1]
                img.save(join(id_path, img_name))


def eps_search(db, X, time_req, eps_low, eps_high, trials, min_samples, metric, output_dir, quiet=False):
    """Export a scaled search of parameters based on the value of epsilon and the minimal intervention time.

    Args:
        db (pd.DataFrame): Data information
        X (float): Vector embeddings
        time_req (int): Minimum time to be considered participant
        eps_low (float): Lower limit of search
        eps_high (float): Upper limit of search
        min_samples (int): Fixed DBSCAN hparam. Low to assure convergence
        metric (str): Pairwise distance to use
        output_dir (str): Output directory
        quiet (bool, optional): Whether to display info

    Return:
        a float, the best epsilon found
    """
    maindir = join(dirname, 'min_samples_' + str(min_samples))
    overall = {
        'score': [],  # Silhouette average score
        'num_clusters': [],  # Total number of clusters
        'noise': [],  # Proportion of noise samples
        'newly_created': [],  # Generated from scratch clusters
        'merging_processes': []  # Cases of merging clusters
    }
    eps = np.linspace(eps_low, eps_high, trials)
    old_pred = np.ones(len(X)) * (-1)
    best_score = -1
    best_eps = eps_low
    best_pred = None

    import pdb

    for e in range(len(eps)):
        pred_labels, info = _eps_search(
            X=X,
            eps=eps[e],
            min_samples=min_samples,
            metric=metric,
            quiet=quiet)
        status = _status(pred_labels)
        info['newly_created'], info['merging_processes'] = compare(
            status, old_pred)
        old_pred = pred_labels.copy()
        print(info)
        if info['score'] > best_score:
            best_score = info['score']
            best_eps = eps[e]
            best_pred = pred_labels

        for k in info.keys():
            overall[k].append(info[k])


    epsdir = join(maindir, 'eps_' + str(best_eps))
    db['pred_labels'] = best_pred
    export(join(epsdir, 'labels'), db)
    if not quiet:
        print('> Configuration saved')
        print('> Computing centroids & cluster mosaics')

    save_clusters(db, epsdir, time_req)

    overall = pd.DataFrame(
        data=overall,
        index=eps,
        columns=overall.keys())
    export(join(maindir, 'results'), overall)

    return best_eps


if __name__ == "__main__":
    args = DbscanArgs().parse()
    db = load_db(args.program_csv)
    _, size = load_embeddings(db)
    hist_face_sizes(size, 'size', args.output_dir)
    if not args.quiet:
        print('> Perimeter and area computed for all faces')
    db = size_filter(db, args.min_area)
    X, size = load_embeddings(db)
    dirname = join(args.output_dir, 'dbscan_' + args.metric)

    export(join(dirname, 'min_samples_' + str(args.min_samples),
                'hparams.txt'), args)

    if not args.quiet:
        print('> Proceed to hyperparameters search...')
    filename = 'dist_' + str(args.nthneigh - 1) + 'th_neighbor.csv'
    dists = nearestneighbors(X, args.nthneigh, metric=args.metric)
    export(join(dirname, filename), data=dists)

    if not args.quiet:
        print('> Distance to neighbors exported')
        print('> Searching a nice DBSCAN configuration')

    eps_search(
        db=db,
        X=X,
        time_req=args.min_part,
        eps_low=args.eps_low,
        eps_high=args.eps_upper,
        trials=args.trials,
        min_samples=args.min_samples,
        metric=args.metric,
        output_dir=dirname,
        quiet=args.quiet)
