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

		output:
		> a list with k elements. Each entry is an array with shape 1 x dimension.
	"""

	rng = np.random.default_rng()
	random_points = rng.uniform(range[0], range[1], [k, dimension])
	random_points = [np.array(a) for a in random_points.tolist()]
	return random_points


def frobenius_norm(x1, x2):
	"""Returns the Frobenius norm of the difference between x1 and x2.
		input variables:
		> x1,x2: numpy arrays with same shape.

		output:
		the frobenius norm of the diference between x1 and x2.
	"""
	x1 = np.array(x1)
	x2 = np.array(x2)
	abs_difference = np.abs(x1-x2)
	frob = np.sqrt(np.sum(abs_difference**2))
	return frob


def random_dataset_points(X, k):
	indices = random_integers(X.shape[0], k=k, replace=False)
	return X[indices]


def lloyd(X, centroids, distance, max_runs, epsilon):
	"""Classical KMeans algorithm.

		input variables
		> X: the dataset (numpy array) with shape m x n, where m is the size
		     of the dataset and n the number of featuras.
	    > centroids: (numpy array) the initial centroids.
	    > distance: the distance function to use in the algorithm. the standard is
	                to use the euclidean distance.
		> max_runs: (int) the number of iterations to run the algorithm
		> epsilon:  (float) ...
	"""
	old_centroids = centroids
	old_clusters = None
	number_of_clusters = len(centroids)
	
	for __ in range(max_runs):
		distances = np.array([distance(X, centroid) for centroid in centroids])
		clusters = np.argmin(distances, axis=0)
		new_centroids = [np.mean(X[clusters == ind], axis=0) for ind in range(number_of_clusters)]

		if old_clusters is not None and np.array_equal(old_clusters, clusters):
			break  # the assignmant hasn't change.

		frob_norm = frobenius_norm(old_centroids, new_centroids)
		if frob_norm < epsilon:
			break

		old_clusters = clusters
	
	partial_sum = [sum(distance(X[clusters == ind], centroid)) for ind, centroid in enumerate(centroids)]
	inertia = sum(partial_sum)

	return clusters, new_centroids, inertia


class KMeans:
	""" KMeans object."""
	
	def __init__(self, k=3, n_iter=10, max_runs=300, epsilon=0.001, distance=euclidean_distance):
		self.number_of_clusters = k
		self.distance = distance
		self.n_iter = n_iter
		self.epsilon = epsilon
		self.max_runs = max_runs
		self.clusters = None
		self.inertia = None
		self.centroids = None


	def fit(self, X):
		best_inertia = None
		for __ in range(self.n_iter):
			centroids = random_dataset_points(X, self.number_of_clusters)
			centroids = [centroids[i] for i in range(self.number_of_clusters)]
		
			new_clusters, new_centroids, new_inertia = lloyd(X=X, centroids=centroids, distance=self.distance,
						                                    max_runs=self.max_runs, epsilon=self.epsilon)
				
			if not best_inertia or (new_inertia < best_inertia):
				best_inertia = new_inertia
				best_clusters = new_clusters
				best_centroids = new_centroids
			
		self.clusters = best_clusters
		self.inertia = best_inertia
		self.centroids = best_centroids

	
	def score(self):
		print("Clusters: ", self.clusters)
		print("Inertia: ", self.inertia)