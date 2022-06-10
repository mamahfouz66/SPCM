"""Implementations of a number of C-means algorithms.

References
----------
.. [1] J. C. Bezdek, J. Keller, R. Krisnapuram, and N. R. Pal, Fuzzy models
   and algorithms for pattern recognition and image processing. Kluwer Academic
   Publishers, 2005.

"""
import time
from collections import Counter
from timeit import timeit
import bisect
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import sklearn.datasets as dt

from initialization import initialize_random, initialize_probabilistic



class CMeans:
    """Base class for C-means algorithms.

    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters to find.
    n_init : int, optional
        The number of times to attempt convergence with new initial centroids.
    max_iter : int, optional
        The number of cycles of the alternating optimization routine to run for
        *each* convergence.
    tol : float, optional
        The stopping condition. Convergence is considered to have been reached
        when the objective function changes less than `tol`.
    verbosity : int, optional
        The verbosity of the instance. May be 0, 1, or 2.

        .. note:: Very much not yet implemented.

    random_state : :obj:`int` or :obj:`np.random.RandomState`, optional
        The generator used for initialization. Using an integer fixes the seed.
    eps : float, optional
        To avoid numerical errors, zeros are sometimes replaced with a very
        small number, specified here.

    Attributes
    ----------
    metric : :obj:`string` or :obj:`function`
        The distance metric used. May be any of the strings specified for
        :obj:`cdist`, or a user-specified function.
    initialization : function
        The method used to initialize the cluster centers.
    centers : :obj:`np.ndarray`
        (n_clusters, n_features)
        The derived or supplied cluster centers.
    memberships : :obj:`np.ndarray`
        (n_samples, n_clusters)
        The derived or supplied cluster memberships.

    """

    metric = 'euclidean'
    initialization = staticmethod(initialize_random)

    def __init__(self, n_clusters=2,m=2.3, n_init=20, max_iter=1000, tol=1e-4,max_fail = 20,
                 verbosity=0, random_state=None, eps=1e-18, **kwargs):

        self.max_fail=max_fail
        self.m=m
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbosity = verbosity
        self.random_state = random_state
        self.eps = eps
        self.params = kwargs
        self.centers = None
        self.memberships = None
        self.tot_iterations = 0
    def distances(self, x):
        """Calculates the distance between data x and the centers.

        The distance, by default, is calculated according to `metric`, but this
        method should be overridden by subclasses if required.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        Returns
        -------
        :obj:`np.ndarray`
            (n_samples, n_clusters)
            Each entry (i, j) is the distance between sample i and cluster
            center j.

        """
        try:
            return cdist(x, self.centers, metric=self.metric)
        except:
            return cdist(x, self.centers, metric=self.metric)


    def calculate_memberships(self, x):
        raise NotImplementedError(
            "`calculate_memberships` should be implemented by subclasses.")


    def calculate_centers(self, x):
        raise NotImplementedError(
            "`calculate_centers` should be implemented by subclasses.")


    def objective(self, x):
        raise NotImplementedError(
            "`objective` should be implemented by subclasses.")


    def fit(self, x):
        """Optimizes cluster centers by restarting convergence several times.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        objective_best = np.infty
        memberships_best = None
        centers_best = None
        j_list = []
        for i in range(self.n_init):
            self.centers = None
            self.memberships = None
            self.tot_iterations += self.converge(x)[0]
            start = time.time()
            objective = self.objective(x)
            end = time.time()
            print('objective time = {0:8.8f}'.format(end-start))
            j_list.append(objective)
            if objective < objective_best:
                memberships_best = self.memberships.copy()
                centers_best = self.centers.copy()
                objective_best = objective
        self.memberships = memberships_best
        self.centers = centers_best
        return memberships_best,centers_best,j_list,self.tot_iterations


    # def converge(self, x):
    #     """Finds cluster centers through an alternating optimization routine.
    #
    #     Terminates when either the number of cycles reaches `max_iter` or the
    #     objective function changes by less than `tol`.
    #
    #     Parameters
    #     ----------
    #     x : :obj:`np.ndarray`
    #         (n_samples, n_features)
    #         The original data.
    #
    #     """
    #     centers = []
    #     j_new = np.infty
    #     objective_best = np.infty
    #     for i in range(self.max_iter):
    #         j_old = j_new
    #         self.update(x)
    #         centers.append(self.centers)
    #         j_new = self.objective(x)
    #         if j_new < objective_best:
    #             memberships_best = self.memberships.copy()
    #             centers_best = self.centers.copy()
    #             objective_best = j_new
    #         if np.abs(j_old - j_new) < self.tol:
    #            break
    #     self.memberships = memberships_best
    #     self.centers = centers_best
    #     #return
    #     return i,np.array(centers)

    def converge(self, x):
        """Finds cluster centers through an alternating optimization routine.

        Terminates when either the number of cycles reaches `max_iter` or the
        objective function changes by less than `tol`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        centers = []
        j_best = np.infty
        fail_count=0
        for i in range(self.max_iter):
            self.update(x)
         #   centers.append(self.centers)
            j_new = self.objective(x)
            if j_best < j_new:
                fail_count += 1
            else:
                j_best = j_new
                fail_count = 0
                memberships_best = self.memberships.copy()
                centers_best = self.centers.copy()

            if fail_count> self.max_fail:
                self.memberships = memberships_best
                self.centers = centers_best
                break

        return  i-fail_count+1,np.array(centers_best)


    def update(self, x):
        """Updates cluster memberships and centers in a single cycle.

        If the cluster centers have not already been initialized, they are
        chosen according to `initialization`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        start = time.time()
        self.initialize(x)
        end = time.time()
        print ('initialization time = {0:8.8f}'.format(end-start))
        start = time.time()
        self.memberships = self.calculate_memberships(x)
        end = time.time()
        print('membership time = {0:8.8f}'.format(end-start))
        start = time.time()
        self.centers = self.calculate_centers(x)
        end = time.time()
        print('center time = {0:8.8f}'.format(end-start))



    def initialize(self, x):
        if self.centers is None and self.memberships is None:
            self.memberships, self.centers = \
                self.initialization(x, self.n_clusters, self.random_state)
        elif self.memberships is None:
            self.memberships = \
                self.initialization(x, self.n_clusters, self.random_state)[0]
        elif self.centers is None:
            self.centers = \
                self.initialization(x, self.n_clusters, self.random_state)[1]



class Hard(CMeans):
    """Hard C-means, equivalent to K-means clustering.

    Methods
    -------
    calculate_memberships(x)
        The membership of a sample is 1 to the closest cluster and 0 otherwise.
    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them.
    objective(x)
        Interpretable as the data's rotational inertia about the cluster
        centers. To be minimised.

    """

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return (np.arange(distances.shape[1])[:, np.newaxis] == np.argmin(
            distances, axis=1)).T


    def calculate_centers(self, x):
        return np.dot(self.memberships.T, x) / \
               np.sum(self.memberships, axis=0)[..., np.newaxis]


    def objective(self, x):
        if self.memberships is None or self.centers is None:
            return np.infty
        distances = self.distances(x)
        return np.sum(self.memberships * distances)



class Fuzzy(CMeans):
    """Base class for fuzzy C-means clusters.

    Attributes
    ----------
    m : float
        Fuzziness parameter. Higher values reduce the rate of drop-off from
        full membership to zero membership.

    Methods
    -------
    fuzzifier(memberships)
        Fuzzification operator. By default, for memberships $u$ this is $u^m$.
    objective(x)
        Interpretable as the data's weighted rotational inertia about the
        cluster centers. To be minimised.

    """

    m = 2

    def fuzzifier(self, memberships):
        return np.power(memberships, self.m)


    def objective(self, x):
        if self.memberships is None or self.centers is None:
            return np.infty
        distances = self.distances(x)
        return np.sum(self.fuzzifier(self.memberships) * distances)


class Probabilistic(Fuzzy):
    """Probabilistic C-means.

    In the probabilistic algorithm, sample points have total membership of
    unity, distributed equally among each of the centers. This tends to push
    cluster centers away from each other.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance :math:`d_{ij}` between the
        sample :math:`j` and the cluster center :math:`i`.

        .. math::

            u_{ik} = \left(\sum_j \left( \\frac{d_{ik}}{d_{jk}} \\right)^{\\frac{2}{m - 1}} \\right)^{-1}

    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships.

        .. math:: c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \\right.

    """
    def calculate_memberships(self, x):
        distances = self.distances(x)
        distances[distances == 0.] = 1e-18
        return np.sum(np.power(
            np.divide(distances[:, :, np.newaxis], distances[:, np.newaxis, :]),
            2 / (self.m - 1)), axis=2) ** -1

    def calculate_centers(self, x):
        return np.dot(self.fuzzifier(self.memberships).T, x) / \
               np.sum(self.fuzzifier(self.memberships).T, axis=1)[..., np.newaxis]



class Possibilistic(Fuzzy):
    """Possibilistic C-means.

    In the possibilistic algorithm, sample points are assigned memberships
    according to their relative proximity to the centers. This is controlled
    through a weighting to the cluster centers, approximately the variance of
    each cluster.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance :math:`d_{ij}` between the
        sample :math:`j` and the cluster center :math:`i`, and the weighting
        :math:`w_i` of each center.

        .. math::

            u_{ik} = \left(1 + \left(\\frac{d_{ik}}{w_i}\\right)^\\frac{1}{m
            -1} \\right)^{-1}

    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships.

        .. math::

            c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \\right.

    """

    initialization = staticmethod(initialize_probabilistic)
    _weights = None

    def weights(self, x):
        if self._weights is None:
            distances = self.distances(x)
            memberships = self.memberships
            self._weights = np.sum(self.fuzzifier(memberships) * distances,
                                   axis=0) / np.sum(self.fuzzifier(memberships),
                                                    axis=0)
        return self._weights


    def calculate_memberships(self, x):
        distances = self.distances(x)
        return (1. + (distances / self.weights(x)) ** (
            1. / (self.m - 1))) ** -1.


    def calculate_centers(self, x):

        return np.divide(np.dot(self.fuzzifier(self.memberships).T, x),
                         np.sum(self.fuzzifier(self.memberships), axis=0)[
                             ..., np.newaxis])

class WPCM(Fuzzy):
    """Possibilistic C-means.

    In the possibilistic algorithm, sample points are assigned memberships
    according to their relative proximity to the centers. This is controlled
    through a weighting to the cluster centers, approximately the variance of
    each cluster.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance :math:`d_{ij}` between the
        sample :math:`j` and the cluster center :math:`i`, and the weighting
        :math:`w_i` of each center.

        .. math::

            u_{ik} = \left(1 + \left(\\frac{d_{ik}}{w_i}\\right)^\\frac{1}{m
            -1} \\right)^{-1}

    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships.

        .. math::

            c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \\right.

    """

    initialization = staticmethod(initialize_probabilistic)
    _weights = None

    def weights(self, x):
        if self._weights is None:
            distances = self.distances(x)
            memberships = self.memberships
            self._weights = np.sum(self.fuzzifier(memberships) * distances,
                                   axis=0) / np.sum(self.fuzzifier(memberships),
                                                    axis=0)
        return self._weights

    # def initialization(self,x, k, random_state=None):
    #
    #     clusterer = Probabilistic(n_clusters=k, random_state=random_state)
    #     clusterer.converge(x)
    #     return clusterer.memberships, clusterer.centers

    def calculate_memberships(self, x):
        distances = self.distances(x)
        # for outliers use small weights
        wout= np.sum(np.exp(-0.5*distances),axis=1)
        return wout[:,np.newaxis] * ((1. + (distances / self.weights(x)) ** (
            1. / (self.m - 1))) ** -1.)


    def calculate_centers(self, x):

        return np.divide(np.dot(self.fuzzifier(self.memberships).T, x),
                         np.sum(self.fuzzifier(self.memberships), axis=0)[
                             ..., np.newaxis])

class SPCM(Fuzzy):
    initialization = staticmethod(initialize_probabilistic)
    _weights = None
    # posbins=None
    # xzero2one=None
    # x2pm=None
    # xb=None
    # zeta=None
    # xd=None
    # xdm=None
    def __init__(self, bits=8,m=2.0,crisp=True, alpha=0.02,stp=3,n_clusters=2 ):

        super(SPCM,self).__init__(n_clusters=n_clusters,m=m)

        self.bits=bits
        self.m=m
        self.alpha=alpha
        self.stp=stp
        self.crisp=crisp;

        self.full = (1 << self.bits) - 1
        self.xzero2one = np.linspace(0.0000001, 1, self.full + 1)
        self.xb = np.flip(self.xzero2one)
        self.xb = np.delete(self.xb, 0)
        self.x2pm = self.xzero2one ** self.m   #self.xzero2one
        # posbins computing corresponding bins for squared distance corresponding to possibilistic memberships
        self.posbins = np.array((1 / self.xb - 1) ** (self.m - 1))
        self.xd = np.zeros((self.n_clusters, self.full))
        self.xdm = np.zeros((self.n_clusters, len(self.xb)))

    def fuzzifier(self, memberships):
        if self._weights is None:
             return np.power(memberships, self.m)
        else:
             return self.x2pm[memberships.astype(int)]

    def weights(self, x):
        if self._weights is None:
            distances = self.distances(x)
            memberships = self.memberships
            self._weights = np.sum(self.fuzzifier(memberships) * distances,
                                   axis=0) / np.sum(self.fuzzifier(memberships),
                                                    axis=0)
        return self._weights

    def binary_search(self,arr):
        outliers_prcnt=0.0625* self.full
        core_threshold= 0.8* self.full
        #arr = tpl[1] if we use enumerate we receive on parameter tuple
        #list1 = self.xd[tpl[0]]
        for  i in range( len(list(arr))):
            list1 = self.xd[i]
            low = 0
            high = len(list1) - 1
            mid = 0
            found=False
            while low <= high:
                # for get integer result
                mid = (high + low) // 2

                # Check if n is present at mid
                if list1[mid] < arr[i]:
                    if (list1[mid+1] > arr[i]):
                          arr[i] =  self.full-mid-1 if mid < len(list(arr)) -1  else self.full-mid
                          found=True
                          break
                    else:
                          low = mid

                    # If n is greater, compare to the right of mid
                elif list1[mid] > arr[i]:
                    if (list1[mid - 1] < arr[i]):
                        arr[i] =  self.full-mid+1  if (mid<=0) else self.full-mid
                        found=True
                        break
                    else:
                        high = mid - 1

                    # If n is smaller, compared to the left of mid
                else:
                    arr[i]=  self.full- mid
                    found=True
                    break
            if (not found):
                 arr[i]=-1
        # sumarr= sum(arr)
        # maxarr= max(arr)
        # if (sumarr<outliers_prcnt):
        #     arr=np.zeros(len(arr))
        # elif (maxarr>  core_threshold ):
        #     for i in range(len(arr)):
        #         if (arr[i]< maxarr ):
        #             arr[i]=max(arr[i]-self.stp,0)
        #         else:
        #             arr[i]=min(arr[i]+self.stp,self.full)
                # element was not present in the list, return -1
        return arr
    def calculate_memberships(self, x):
        self.xd = [self.weights(x)[i] * self.posbins for i in range(self.n_clusters)]
        #if we want to reduce the number calls
        # distances = np.transpose(self.distances(x))
        # u=list(map(self.binary_search,  enumerate(distances)  ))
        # u=np.transpose(u)
        distances =  self.distances(x)
        # u=list(map(self.binary_search,   distances ))
        u = np.zeros((x.shape[0], self.n_clusters), dtype=int)
        if (self.crisp):
            for i in range(self.n_clusters):
                 u[:,i] = [self.full-bisect.bisect(self.xd[i],distances[j,i]) for j in range(distances.shape[0])]
        # triangular membership
        else:
            for i in range(self.n_clusters):
                 u[:,i] = [ self.full-bisect.bisect(self.xd[i],distances[j,i])  for j in range(distances.shape[0])]

        #u = np.zeros((x.shape[0], self.n_clusters), dtype=int)
        #for i in range(self.n_clusters):
        #    u[:,i] = self.full- np.digitize(distances[:, i], xd[i],right=False)
        #outliers_actions = self.compute_actions_for_large_distances( u)
        #core_actions = self.compute_actions_for_core_objects(u)
        #u = u + self.stp * (core_actions - outliers_actions)
        #u[u > self.full] = self.full
        #u[u < 0] = 0
        return np.array(u)

    def calculate_centers(self, x):
        #dom=np.full(self.n_clusters,0.0)
        #nom=np.zeros((self.n_clusters,x.shape[1]))
        ####cnt=np.apply_along_axis(lambda x: np.bincount(x, minlength=self.full), axis=0, arr=self.memberships)
        # cnt = np.array( [[np.count_nonzero(self.memberships[:, i] == key) for i in range(self.n_clusters)] for key in range(self.full+1)])
        # dom= np.dot(self.x2pm[range(self.full+1)][:,np.newaxis].T , cnt).flatten()
        # nom= np.array([[sum(x[self.memberships[:, i] == key])*self.x2pm[key]/dom[i] for i in range(self.n_clusters) ] for key in range(self.full+1)])
        # return np.array(np.sum(nom,axis=0))
        ####nom= self.x2pm[range(self.full+1)][:,np.newaxis] *  suma
        ## another solution
        # for i in range(self.n_clusters):
        #     res = dict(Counter(self.memberships[:, i]))
        #     for key, cnt in res.items():
        #          nom[i] += self.x2pm[int(key)] * sum(x[self.memberships[:, i] == key])
        #          dom[i] += self.x2pm[int(key)] * cnt
        #     nom[i] = nom[i]/dom[i]

        ## traditional solution
        #return nom
        fm = self.fuzzifier(self.memberships)
        return np.divide(np.dot(fm.T, x),
                         np.sum(fm, axis=0)[
                             ..., np.newaxis])



        #return nom / dom


    def compute_actions_for_large_distances(self, u):
        w = np.sum(u, axis=1)
        epson = (1 - self.alpha) * np.min(w) + self.alpha * np.max(w)
        w = (w < epson).astype(int)
        w = np.broadcast_to(w[:, np.newaxis], shape=np.shape(u))
        return w

    def compute_actions_for_core_objects(self, u):
        threshold= np.floor(2**self.bits*0.75)
        w=np.zeros(np.shape(u))
        au =np.argmax(u,axis=1)
        for i in range(np.shape(u)[0]):
            if (u[i,au[i]]>threshold):
                    for j in range(np.shape(u)[1]):
                        w[i, j]=-1
                        w[i,au[i]]=1
        return w

    # def objective(self, x):
    #     if self.memberships is None or self.centers is None:
    #         return np.infty
    #     distances = self.distances(x)
    #     _obj = 0  # np.zeros(self.n_clusters)
    #     for i in range(self.n_clusters):
    #         res = dict(Counter(self.memberships[:, i]))
    #         for key, cnt in res.items():
    #             _obj += self.x2pm[int(key)] * sum(distances[:,i][self.memberships[:, i] == key])
    #
    #     return _obj


class GustafsonKesselMixin(Fuzzy):
    """Gives clusters ellipsoidal character.

    The Gustafson-Kessel algorithm redefines the distance measurement such that
    clusters may adopt ellipsoidal shapes. This is achieved through updates to
    a covariance matrix assigned to each cluster center.

    Examples
    --------
    Create a algorithm for probabilistic clustering with ellipsoidal clusters:

    >>> class ProbabilisticGustafsonKessel(GustafsonKesselMixin, Probabilistic):
    >>>     pass %to implement on the future
    >>> pgk = ProbabilisticGustafsonKessel()
    >>> pgk.fit(x)

    """
    covariance = None

    def fit(self, x):
        """Optimizes cluster centers by restarting convergence several times.

        Extends the default behaviour by recalculating the covariance matrix
        with resultant memberships and centers.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        j_list = super(GustafsonKesselMixin, self).fit(x)
        self.covariance = self.calculate_covariance(x)
        return j_list


    def update(self, x):
        """Single update of the cluster algorithm.

        Extends the default behaviour by including a covariance calculation
        after updating the centers

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        self.initialize(x)
        self.centers = self.calculate_centers(x)
        self.covariance = self.calculate_covariance(x)
        self.memberships = self.calculate_memberships(x)


    def distances(self, x):
        covariance = self.covariance if self.covariance is not None \
            else self.calculate_covariance(x)
        d = x - self.centers[:, np.newaxis]
        left_multiplier = \
            np.einsum('...ij,...jk', d, np.linalg.inv(covariance))
        return np.sum(left_multiplier * d, axis=2).T


    def calculate_covariance(self, x):
        """Calculates the covariance of the data `u` with cluster centers `v`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        Returns
        -------
        :obj:`np.ndarray`
            (n_clusters, n_features, n_features)
            The covariance matrix of each cluster.

        """
        v = self.centers
        if v is None:
            return None
        q, p = v.shape
        if self.memberships is None:
            # If no memberships have been calculated assume n-spherical clusters
            return (np.eye(p)[..., np.newaxis] * np.ones((p, q))).T
        q, p = v.shape
        vector_difference = x - v[:, np.newaxis]
        fuzzy_memberships = self.fuzzifier(self.memberships)
        right_multiplier = \
            np.einsum('...i,...j->...ij', vector_difference, vector_difference)
        einstein_sum = \
            np.einsum('i...,...ijk', fuzzy_memberships, right_multiplier) / \
            np.sum(fuzzy_memberships, axis=0)[..., np.newaxis, np.newaxis]
        return np.nan_to_num(
            einstein_sum / (np.linalg.det(einstein_sum) ** (1 / q))[
                ..., np.newaxis, np.newaxis])

def main():
    data, labels = dt.make_blobs(n_features=2,
                                 centers=50,
                                 n_samples=100000,
                                 cluster_std=2,
                                 random_state=11)
    sc = StandardScaler()
    # Compute the mean and standard deviation based on the training data
    sc.fit(data)
    # Scale the training data to be of mean 0 and of unit variance
    x = sc.transform(data)
    start = time.time()
    p = Possibilistic(n_clusters=50)
    p.memberships, p.centers = \
        p.initialization(x, p.n_clusters, p.random_state)
    d = p.distances(x)
    w =p.weights(x)
    for i in range(100):
        u= (1. + (d/ w) ** (1. / (1.3 - 1))) ** -1.
    end = time.time()
    print('pcm = {0:8.8f}'.format(end - start))
    start = time.time()
    p = SPCM(n_clusters=50)
    xd=np.linspace(np.min(d), np.max(d))
    for i in range(100):
        u  =np.digitize(d,xd )
        # u = np.zeros((x.shape[0], p.n_clusters), dtype=int)
        # xd = [w[i] * p.posbins for i in range(p.n_clusters)]
        # for i in range(p.n_clusters):
        #     u[:, i] = p.full - np.digitize(d[:, i], xd[i])
        # outliers_actions = p.compute_actions_for_large_distances(u)
        # core_actions = p.compute_actions_for_core_objects(u)
        # u = u + p.stp * (core_actions - outliers_actions)
        # u[u > p.full] = p.full
        # u[u < 0] = 0
    #print(timeit(stmt="p.calculate_memberships",globals=globals(), number=10))
    #p.fit(data)
    end = time.time()
    print('spcm = {0:8.8f}'.format(end - start))
class Testme:
    if __name__ == '__main__':
      main()
