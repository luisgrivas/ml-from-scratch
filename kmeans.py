import numpy as np


def euclidean_distance(X1, X2) -> float:
	""" Returns the (square) euclidean distance between an array of dimension
	    m x n and an array of dimension 1 x n.
	    
	    input variables: 
	    > X1: an array of dimension m x n.
	    > X2: an array of dimension 1 x n.

	    output:
	    > (float) the euclidian square distance the arrays X1 and X2.
	"""
	return np.sum((X1 - X2)**2, axis=1)


def random_integers(n: int, k: int, replace=True) -> list:
	"""Returns a list k random integers 
	   in the range [0, n].
	"""
	rng = np.random.default_rng()
	random_integers = list(rng.choice(n, size=k, replace=replace))
	return random_integers


def random_points(k: int, dimension: int, range: list) -> list:
	""" Returns k random points (drawn from a uniform distribution)
	in a given range.

		input variables: 
		> k (integer): the number of points.
		> dimension (integer): the points' dimension.
		> range (list): the range of each of points' coordinates.

		function output:
		> a list with k elements. Each entry is an array with shape 1 x dimension.
	"""

	rng = np.random.default_rng()
	random_points = rng.uniform(range[0], range[1], [k, dimension])
	random_points = [np.array(a) for a in random_points.tolist()]
	return random_points


def elkan(X, number_of_clusters, centroids, distance):
	distances = [distance(X, centroid) for centroid in centroids]
	distances = np.array(distances)
	clusters = np.argmin(distances, axis=0)
	new_centroids = [np.mean(X[clusters == ind], axis=0) for ind in range(number_of_clusters)]
	inertia = [distance(X[clusters == ind], centroid) for centroid in centroids]
	return clusters, new_centroids


class KMeans:
	""" KMeans object."""
	
	def __init__(self, k=3, distance=euclidean_distance):
		self.number_of_clusters = k
		self.distance = distance
		self.clusters = None


	def fit(self, X):
		number_of_points, dimension = X.shape
		clusters = random_integers(self.number_of_clusters, number_of_points)
		centroids = random_points(self.number_of_clusters, X.shape[1],
			                      [np.min(X), np.max(X)])
		i = 0

		while i < 100:
			distances = [self.distance(X, centroid) for centroid in centroids]
			distances = np.array(distances)
			clusters = np.argmin(distances, axis=0)
			centroids = [np.mean(X[clusters == ind], axis=0) for ind in range(self.number_of_clusters)]
			i+=1

		self.clusters = clusters


	def score():
		return self.clusters
