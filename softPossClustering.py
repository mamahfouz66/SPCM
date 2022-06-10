import decimal
from copy import deepcopy
from scipy.spatial.distance import cdist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as dt
from pandas import read_csv
from sklearn import metrics
from sklearn.datasets import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, load_digits, load_wine
from collections import  Counter
import skmeans
import pandas as pd
import time

from pfcm import pfcm


class SoftPossClustering:

    def __init__(self, num_clusters=3, stp=1, m=2.3, alpha=0.1, bits=4, tolerance=0.0001, max_iterations=300):

        self.num_clusters = num_clusters
        self.msr_threshold = alpha
        self.bits = bits
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.m = m
        self.stp = self.bits
        #self.stp = np.floor(2 ** self.bits * 0.20)
        self.alpha = alpha

    def _validate_parameters(self):
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be > 0, got {}".format(self.num_clusters))
        if self.bits > 32:
            raise ValueError(
                " half the number of possible membership should be less than <= 128  got {}".format(self.bits))

    def run(self, data, labels):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """

        self._validate_parameters()

        # full = 2^self.bits
        full = (1 << self.bits) - 1
        data = check_array(data, dtype=np.double, copy=True)
        # memberships 0 to full replacing 0 with very small membership
        xzero2one = np.linspace(0.0000001, 1, full + 1)
        xb=np.flip(xzero2one)
        xb=np.delete(xb,0)
        #xb = np.delete(xzero2one,-1)
        x2pm = xzero2one**self.m
        # posbins computing corresponding bins for squared distance corresponding to possibilistic memberships
        posbins =(1 / xb - 1) ** (self.m - 1)
        # xum computing corresponding bins for squared distance corresponding to probabilistic memberships
        #xum = (1 / xb ** self.m - 1) ** (self.m - 1)
        num_rows, num_cols = data.shape
        # scalling parameter that will be changed dynamically
        zeta = np.ones(self.num_clusters)
        # u hold the soft memberships
        u = np.zeros((data.shape[0], self.num_clusters), dtype=int)
        # um hold the memberships to the power m
        #um = np.zeros((data.shape[0], self.num_clusters))
        # xd hold the memberships to the power m
        xd = np.zeros((self.num_clusters, full ))
        # xdm hold the array of bins for digitizing the distances to get the memberships
        xdm = np.zeros((self.num_clusters, len(xb)))
        clusters = []
        centers = data[np.random.randint(num_rows, size=self.num_clusters), :]
        # another approach for generating ceneters
        # mean = np.mean(data, axis = 0)
        # std = np.std(data, axis = 0)
        # centers = np.random.randn(self.num_clusters,c)*std + mean
        #centers_old = np.zeros(centers.shape)  # to store old centers
        #centers_new = deepcopy(centers)  # Store new centers
        distances = np.zeros((num_rows, self.num_clusters))
        for i in range(self.num_clusters):
            # distances[:, i] = np.linalg.norm(data - centers_new[i], axis=1)
             d_c = data - centers[i]
             distances[:, i] = np.sum(d_c * d_c, axis=1)  # squared ecludian distance
        clusters = np.argmin(distances, axis=1)
        u = np.zeros((num_rows, self.num_clusters), dtype=int)
        for i in range(num_rows):
            u[i, clusters[i]] = full
        #pru = np.zeros((num_rows, self.num_clusters))
        # you may stop if no more actions i.e. all zeros
        iterno = 0

        final_mem = xzero2one[u]
        while True:
            # compute distances to centers
            for i in range(self.num_clusters):
                zeta[i] = np.sum(distances[:, i][clusters == i] * final_mem[:, i][clusters == i], axis=0)
                zeta_dom = np.sum(final_mem[:, i][clusters == i], axis=0)
                if (zeta_dom>0.0):
                    zeta[i] = zeta[i] / zeta_dom
                else:
                    zeta[i]=1
            for i in range(self.num_clusters):
                distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
                #d_c = data - centers[i]
                # computing squared ecludian distance
                #distances[:, i] = np.sum(d_c * d_c, axis=1)
                # posbins passed as encrypted parameter and zeta updated dynamically
                xd[i] = posbins * zeta[i]  # np.sqrt(
                # probabilistic memberships bins
                ##xdm[i] = xum * zeta[i]  # np.sqrt(
                u[:, i] = full - np.digitize(distances[:, i], xd[i])
                # probabilistic memberships
                ##um[:, i] = np.digitize(distances[:, i], xdm[i])
                # old formula zeta[i] = np.average(distances[:, i], axis=0, weights=um[:, i])

            outliers_actions = compute_actions_for_large_distances(self,u)
            core_actions = compute_actions_for_core_objects(self,u)
            # pru, pru_actions = actions_probabilistic(self,pru, xb, u)
            # # possibilistic actions moved by step
            # u_actions = (u - oldu)/self.stp
            u = u+self.stp*(core_actions-outliers_actions)
            u[u > full] = full
            u[u < 0] = 0
            # store new soft memberships for next iteration
            #oldu = deepcopy(u)
            # store old centers for termination condition
            centers_old = deepcopy(centers)
            # compute new centers from final memberships without division
            for i in range(self.num_clusters):
                res=dict(Counter(u[:,i]))
                nom=dom=0
                for key,cnt in res.items():
                    nom+= x2pm[int (key)] *sum(data[u[:,i]==key])
                    dom+=x2pm[int(key)]*cnt
                centers[i] =  nom/dom
                #centers_new[i] = np.sum(data * xum[u][:, i][:, np.newaxis], axis=0)
                # centers_new_dom = np.sum(final_mem[:, i], axis=0)
                # compute_one_over_n(centers_new_dom, one_over_n)
                # one_over_n an array holds encrypted 1/n
                # centers[i] = centers[i] /np.sum(final_mem[:, i], axis=0)

            # defuzzify the output
            clusters = np.argmax(u, axis=1)
            #print("adjusted rand score: %0.3f" % metrics.adjusted_rand_score(labels, clusters))
            # zeta can be computed for next iteration after the clusters are known from average d in clusters
            iterno = iterno + 1
            if (iterno > self.max_iterations): #or (np.linalg.norm(centers - centers_old) < self.tolerance):
                break

        #print("number of iterations = ",iterno)
        # masking matrix values
        return u, centers,iterno


def actions_probabilistic(self, oldpru, xb, u):
    # calculate sum of probabilistic memberships over a row
    sumu = np.sum(u, axis=1)[:, np.newaxis]  # per rows
    sumu[sumu == 0.] = 1
    # probabilistic memberships calculated first
    pru = u / sumu  # divide per row //
    pru = np.digitize(pru, xb)
    maxu = np.max(u, axis=1)  # per rows no need pru is enough
    # compute new u from oldu and new actions
    # probabilistic memberships
    pru_actions = pru - oldpru
    pru_actions[pru_actions > self.stp] = self.stp
    pru_actions[pru_actions < -self.stp] = -self.stp
    pru = oldpru + pru_actions
    return pru, pru_actions


def compute_actions_for_large_distances(self, u):
    # objects having very large distances to all centers are candidate outliers
    # but no need to worry about them because poss. mem will be near zeros
    w = np.sum(u, axis=1)
    # find threshold for sum dist to be considered noise alpha represent noise ratio
    epson = (1 - self.alpha) * np.min(w) + self.alpha * np.max(w)

    # true and false converted to 0,1
    w = (w < epson).astype(int)
    w = np.broadcast_to(w[:, np.newaxis], shape=np.shape(u))
    # w = self.stp*(w > epson).astype(int)
    # w = self.stp * w  # [:, np.newaxis]
    return w
def compute_actions_for_core_objects(self, u):
    # objects having very large distances to all centers are candidate outliers
    # but no need to worry about them because poss. mem will be near zeros
    threshold= np.floor(2**self.bits*0.75)
    w=np.zeros(np.shape(u))
    au =np.argmax(u,axis=1)
    for i in range(np.shape(u)[0]):
        if (u[i,au[i]]>threshold):
                for j in range(np.shape(u)[1]):
                    w[i, j]=-1
                    w[i,au[i]]=1

    return w


def compute_one_over_n(dom, one_over_n):
    """ approximate 1/dom using one_over_n holds 1/x for x=1..n , dom is the dominator which is a real number """
    domi = dom.astype(int)
    f_dom = one_over_n[domi] - one_over_n[domi] * one_over_n[domi] * (dom - domi) + one_over_n[domi] * one_over_n[
        domi] * one_over_n[domi] * (dom - domi) * (dom - domi)
    return f_dom


def verify_clusters(labels, clm, data, v, c):
    ssd_actual = 0

    for i in range(c):
        # All points in class
        x1 = data[labels == i]
        # Mean of class
        m = np.mean(x1, axis=0)

        for pt in x1:
            ssd_actual += np.linalg.norm(pt - m)

    ssd_clusters = 0

    for i in range(c):
        # Points clustered in a class
        x2 = data[clm == i]

        for pt in x2:
            ssd_clusters += np.linalg.norm(pt - v[i])

    return (ssd_clusters / ssd_actual)

def data_example():
    #filename = '.\\datasets\\data\\x12.csv'
    #raw_data = open(filename, 'rt')
    #reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    #x = list(reader)
    #data = numpy.array(x).astype('float')
    #df = DataFrame(data, columns=['x', 'y'])
    #np.array(df[df.columns[:-1]].astype(float))
    data = np.array([[1,1],[1.5,2],[3,4],[5, 7],[3.5,5],[4.5,5], [3.5,4.5],[4,15]]).astype('float')
    label = np.array([0,0,1,1,1,1, 1,2]).astype('int')
    initial_centroids = np.array([[1, 1], [5, 7]])
    computed_centroids = np.array([[1.18, 1.36], [4.66, 6.09]]).astype('float')
    #print(data.shape)
    return data , label,initial_centroids,computed_centroids
def data_X12():
    data = np.array([[-5, -3.34, -3.34, -3.34, -1.67, 1.67, 3.34, 3.34, 3.34, 5, 0, 0],
                     [0, 1.67, 0, -1.67, 0, 0, 1.67, 0, -1.67, 0, 0, 20]]).transpose().astype('float')
    label=np.array([0,0,0,0,0,0,1,1,1,1, 1,2]).astype('int')
    initial_centroids = np.array([[-5, 0], [5, 0]])
    computed_centroids = np.array([[-3.10,0], [3.20, 0]]).astype('float')
    return data, label, initial_centroids,computed_centroids
def load_plot(fn):
    data, label,initial_centroids,computed_centroids  = fn()
    # Getting unique labels
    u_labels = np.unique(label)

    actual_centroids = np.zeros((3,2))
    # plotting the results:

    for i in u_labels:
        actual_centroids[i] = np.average(data[label == i],axis=0)
    for i in u_labels:
        plt.scatter(data[label == i, 0], data[label == i, 1],c=['r','g','b'][i] ,marker=['o','s','x'][i])   #label=i,
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.scatter(actual_centroids[:, 0], actual_centroids[:, 1], s=80, c='k',marker='*')
    plt.scatter(computed_centroids[:, 0], computed_centroids[:, 1], s=80, c='m',marker='+')
    plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], s=80, c='m',marker='1')
    #plt.legend()
    plt.show()
def generate_clusters(f,c,std):
    seed = 11
    rand_state = 11

    # Define the color maps for plots
    color_map = plt.cm.get_cmap('RdYlBu')
    color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "cyan", "magenta", "blue"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    plt_ind_list = np.arange(3) + 131

    for std, plt_ind in zip([0.5, 1, 1.5,2], plt_ind_list):
        x, label = dt.make_blobs(n_features=f,
                                 centers=c,
                                 cluster_std=std,
                                 random_state=rand_state )

        plt.subplot(plt_ind)
        my_scatter_plot = plt.scatter(x[:, 0],
                                      x[:, 1],
                                      c=label,
                                      vmin=min(label),
                                      vmax=max(label),
                                      cmap=color_map_discrete)
        plt.title('cluster_std: ' + str(std))

    fig.subplots_adjust(hspace=0.3, wspace=.3)
    plt.suptitle('make_blobs() With Different cluster_std Values', fontsize=20)
    plt.show()
    return x, label


def read_csvfile( dir_name, file_name, category_first=False):
    # import os
    # os.chdir('/Users/stevenhurwitt/Documents/Blog/Classification')

    df = read_csv(dir_name + '/{}.csv'.format(file_name), sep=',', header=0)  # first line of data is header

    # df.info()
    if (category_first):
        # df_majority = df[df.columns[0].values == 0]
        # df_minority = df[df.balance == 1]
        # df_minority_upsampled = resample(df_minority,
        #                                  replace=True,  # sample with replacement
        #                                  n_samples=576,  # to match majority class
        #                                  random_state=123)  # reproducible results
        # Combine majority class with upsampled minority class
        # df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        # Display new class counts
        # df_upsampled.balance.value_counts()
        X = np.array(df[df.columns[1:]].astype(float))  # X = df[1:,:-1]
        y = np.array(df[df.columns[0]].astype('category').cat.codes)  # y= df[1:,-1:]
    else:
        X = np.array(df[df.columns[:-1]].astype(float))  # X = df[1:,:-1]
        y = np.array(df[df.columns[-1]].astype('category').cat.codes)  # y= df[1:,-1:]

    return X, y

def compute_cstar(x,centers, labels, labels_pred):
    memberships = np.zeros((x.shape[0],centers.shape[0]))
    for i in range(x.shape[0]):
        memberships[i][labels_pred[i]] = 1
    actual_centers= np.dot(memberships.T, x) / \
    np.sum(memberships.T, axis=1)[..., np.newaxis]
    alldistances = cdist(actual_centers, centers, metric='euclidean')
    mindist = np.min(alldistances,axis=1)
    maxdist = np.max(mindist)
    avgdist = np.average(mindist)
    sumdist = np.sum(mindist)
    return maxdist, avgdist, sumdist





def main():
    df = pd.DataFrame(columns=['algorithm', 'dataset','datasize','clscnt', 'fcnt','runtime_ms_iter','runtime_ms','iterno','numlocal','m', 'vmi','ami','ari','selh','maxdist', 'avgdist', 'sumdist'])
    #dsizenames = ['C2D100K', 'C2D200K', 'C2D300K', 'C2D400K', 'C2D500K']
    #dclsnames = []
    datasets= {  'iris':3
        #         'wine': 3, 'breast': 2, 'ionosphere': 2,
        #, 'shuttle': 5
                 #'F2C100D100K':102
    #, 'F4C100D100K':104, 'F6C100D100K':106, 'F8C100D100K':108, 'F10C100D100K':110,
    #'C2D100K':2, 'C5D100K':5, 'C10D100K':10, 'C20D100K':20, 'C30D100K':30, 'C40D100K':40, 'C50D100K':50}
    #'C100D100K':10000, 'C100D200K':20000, 'C100D300K':30000,'C100D400K':40000, 'C100D500K':50000,
    #           'C100D600K': 60000, 'C100D700K': 70000, 'C100D800K': 80000, 'C100D900K': 90000, 'C100D1000K': 100000
    }

    algorithms= ['PCM','FCM','SPCM','PFCM','WPCM','HCM']
    #for algo in range(6):
    for algo in range(6):
        #for dsize in range(9):
        for m in  [2.1]: #np.arange(1.1,5,0.2):
            for key,val in datasets.items():
                #datasize=datasetsize[dsize]
                if (val<1000):
                    datasize=10000
                    if (val>100):
                      f=val-100
                      clsno=10
                    else:
                      f=2
                      clsno=val

                else:
                    datasize = val
                    clsno = 10
                    f=2

                #X = normalize(X)

                if (key=='iris'):
                    d = load_iris()
                elif (key=='wine'):
                    d= load_wine()
                elif (key=='breast'):
                    d=load_breast_cancer()
                if key in ('iris', 'wine', 'breast'):
                    data, labels = np.array(d.data),np.array(d.target)
                elif key in ('ionosphere'):
                    dir_name = './data'
                    data, labels = read_csvfile(dir_name, key, False)
                elif key in ('shuttle'):
                    dir_name = './data'
                    data, labels = read_csvfile(dir_name, key, False)
                else:
                    data, labels = dt.make_blobs(n_features=f,
                                                 centers=clsno,
                                                 n_samples=datasize,
                                                 cluster_std=0.8,
                                                 random_state=11)

                sc = StandardScaler()
                # Compute the mean and standard deviation based on the training data
                sc.fit(data)
                # Scale the training data to be of mean 0 and of unit variance
                datastd = sc.transform(data)
                if (datastd.shape[1]>10):
                    pca=PCA(n_components=10)
                    datastd=pca.fit_transform(datastd)
                start=time.time()


                if (algo==0):
                    #p = skmeans.Possibilistic(n_init=1,n_clusters=2)
                    p = skmeans.Possibilistic(n_clusters=clsno)
                    u,centers, objlist,iterno=p.fit(datastd)
                elif (algo==1):
                    #p = skmeans.Possibilistic(n_init=1,n_clusters=2)
                    p = skmeans.Probabilistic(n_clusters=clsno)
                    u,centers, objlist,iterno=p.fit(datastd)
                elif (algo == 2):
                    # p = SoftPossClustering(num_clusters=clsno, bits=4)
                    p = skmeans.SPCM(n_clusters=clsno,m=m)
                    u, centers, objlist, iterno = p.fit(datastd)
                elif (algo == 3):
                    # p = SoftPossClustering(num_clusters=clsno, bits=4)
                    centers, u, T, obj_fcn,iterno = pfcm(data=datastd, c=clsno, expo=2, max_iter=1000, min_impro=0.005, a=1, b=4, nc=3)
                    u=np.transpose(u)
                elif (algo ==4):
                    p = skmeans.WPCM(n_clusters=clsno)
                    u, centers, objlist, iterno = p.fit(datastd)
                elif (algo ==5):
                    p = skmeans.Hard(n_clusters=clsno)
                    u, centers, objlist, iterno = p.fit(datastd)
                end = time.time()
                labels_pred = np.argmax(u, axis=1)
                maxdist, avgdist, sumdist = compute_cstar(datastd,centers,labels, labels_pred)
                p = len(df) + 1
                df.loc[p, 'runtime_ms_iter'] =  (end-start)*1000/iterno
                df.loc[p, 'runtime_ms'] = (end - start) * 1000
                df.loc[p,'iterno'] = iterno
                df.loc[p,'m'] = m
                df.loc[p, 'numlocal'] = 5
                #df.loc[p, 'comp'] = metrics.completeness_score(labels, labels_pred)
                #df.loc[p, 'homo'] = metrics.homogeneity_score(labels, labels_pred)

                df.loc[p, 'ari'] = round(metrics.adjusted_rand_score(labels, labels_pred),4)
                df.loc[p, 'vmi'] = round(metrics.v_measure_score(labels, labels_pred),4)
                #df.loc[p, 'acc'] = metrics.accuracy_score(labels, labels_pred)
                df.loc[p,'ami'] = round(metrics.adjusted_mutual_info_score(labels, labels_pred),4)
                df.loc[p, 'selh'] = round(metrics.silhouette_score(datastd,labels_pred), 4)
                df.loc[p, 'maxdist'] = round(maxdist, 4)
                df.loc[p, 'avgdist'] = round(avgdist, 4)
                df.loc[p, 'sumdist'] = round(sumdist, 4)
                df.loc[p, 'algorithm'] = algorithms[algo]
                df.loc[p, 'dataset'] = key
                df.loc[p,'clscnt']=clsno
                df.loc[p, 'fcnt'] = f
                df.loc[p,'datasize']=datasize
                # print("verify clusters: %0.3f" % verify_clusters(labels, labels_pred, datastd, v=centers, c=3))
                print("adjusted rand score: %1.3f" % metrics.adjusted_rand_score(labels, labels_pred))
                # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, labels_pred))
                # print("Completeness: %0.3f" % metrics.completeness_score(labels, labels_pred))
                # print("V-measure: %0.3f" % metrics.v_measure_score(labels, labels_pred))
                # print()
                print(end - start)
    df.to_csv(r'output/{}.csv'.format('pcm_alldata'), sep=',')


class TestMe:
    if __name__ == "__main__":
        main()

        #load_plot(data_X12)
        #load_plot(data_example)