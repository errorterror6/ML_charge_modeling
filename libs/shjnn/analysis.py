
''' imports '''

#from torch.utils.data import Dataset


# density clustering using HDBSCAN* algorithm
import hdbscan

# dimensionality reduction using UMAP
import umap



''' dimensionality reduction '''


## UMAP

# initialise umap mapper instance
# mapper = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1)

# perform embedding using dataset, optionally supervised with labels
# mapper.fit(data, y = labels)

# get embedding for training data
# embedding = mapper.embedding_

# use trained mapper and transform data into embedding
# embedding = mapper.transform(data)


## HDBSCAN

# initialise hdbscan clusterer instance
# clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=15, prediction_data=True)

# perform density clustering using dataset
# clusterer.fit(data)

# get existing clusterer fit labels
# clusterer.labels_

# get existing clusterer fit probability score within each cluster
# clusterer.probabilities_

# use trained clusterer for classification
# labels, strengths = hdbscan.approximate_predict(clusterer, data)


def umap_embedding(dimensions, n_neighbours = 15, min_dist = 0.8, n_components = 2):

    ''' Get UMAP Embedding

        Perform dimensionality reduction on collection of audio data by extracted features using UMAP library; by
        default only uses left channel feature data; feature data embedded on 2d manifold for exploration

    Args:
        dimensions (dict): feature dimensions with headers and labels

    Returns:
        dict: audio feature dimensions and headers
    '''

    ## UMAP for dimansionality reduction, prepare for clustering

    # initialise umap mapper instance
    mapper = umap.UMAP(n_neighbors = n_neighbours, n_components = n_components, min_dist = min_dist)


    # perform embedding using dataset, optionally supervised with labels
    mapper.fit( dimensions )

    #mapper.fit(data['dimensions'], y = data['labels'])


    # get embedding for training data
    embedding = mapper.embedding_

    # use trained mapper and transform data into embedding
    #embedding = mapper.transform(data)


    # return 2d embedding
    return mapper, embedding



def dimension_reduction(dimensions, params):

    ''' Dimensionality reduction

    Args:
        dimensions (np.array): dimension data
        params (dict): parameters for dimensionality reduction

    Returns:
        np.array: embedding of dimensions with reduced dimensionality
    '''

    # UMAP for dimensionality reduction
    mapper, embedding = umap_embedding(dimensions = dimensions, n_neighbours = params['n_neighbours'],
        min_dist = params['min_dist'], n_components = params['n_components'])


    # return dimensions embedding
    return mapper, embedding



def get_2d_embedding(dimensions, n_neighbors = 15, min_dist = 0.8):

    ''' Get 2-Dimensional Embedding


    Args:
        dimensions (dict): feature dimensions with headers and labels

    Returns:
        dict: audio feature dimensions and headers
    '''

    ## UMAP for dimansionality reduction to 2 dimensions, prepare for display

    # set UMAP parameters
    params = {'n_neighbours': n_neighbors, 'min_dist': min_dist, 'n_components': 2}

    # get embedding
    mapper, embedding = dimension_reduction(dimensions = dimensions, params = params)


    # return 2d embedding
    return mapper, embedding
