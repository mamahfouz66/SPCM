from pandas import read_csv

from _base import BaseBiclusteringAlgorithm
from models import Bicluster, Biclustering
from sklearn.utils.validation import check_array
from statsmodels.stats.weightstats import DescrStatsW
from datasets import load_yeast_tavazoie
import numpy as np


class SoftFLocAlgorithm(BaseBiclusteringAlgorithm):

    def __init__(self, num_biclusters=2, msr_threshold='estimate', half=3):
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.halfshift = half

    def read_csvfile(dir_name, file_name, category_first=False):
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
    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold != 'estimate' and self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be equal to 'estimate' or a numeric value >= 0.0, got {}".format(
                self.msr_threshold))

        if self.halfshift > 7:
            raise ValueError(
                " half the number of possible membership should be less than <= 128  got {}".format(self.halfshift))

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """

        data = check_array(data, dtype=np.double, copy=True)
        self._validate_parameters()

        num_rows, num_cols = data.shape
        min_value = np.min(data)
        max_value = np.max(data)

        msr_thr = (((
                            max_value - min_value) ** 2) / 12) * 0.005 if self.msr_threshold == 'estimate' else self.msr_threshold

        biclusters = []

        for i in range(self.num_biclusters):
            rows = np.ones(num_rows, dtype=np.int) << self.halfshift
            cols = np.ones(num_cols, dtype=np.int) << self.halfshift
            while True:
                msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)
                #msr, row_msr, col_msr = self._calculate_corr(data, rows, cols)
                #   compute necessary actions by comparing their order in the discretized residue to their memberships
                full = 1 << (self.halfshift + 1)
                half = 1 << self.halfshift
                row_msr_indices = full - np.digitize(row_msr, np.linspace(np.min(row_msr), np.max(row_msr), full))
                row_actions = np.sign(row_msr_indices - rows)
                col_msr_indices = full - np.digitize(col_msr, np.linspace(np.min(col_msr), np.max(col_msr), full))

                # actions either -1 mean decrement membership or 1 increment or zero no change
                col_actions = np.sign(col_msr_indices - cols)
                row_actions[(row_actions + rows) < 0] = 0
                row_actions[(row_actions + rows) >= full] = 0
                col_actions[(col_actions + cols) < 0] = 0
                col_actions[(col_actions + cols) >= full] = 0

                # stop if no more actions i.e. all zeros
                row_actions_indices = np.nonzero(row_actions)[0]
                col_actions_indices = np.nonzero(col_actions)[0]
                if len(row_actions_indices) == 0 or len(col_actions_indices) == 0:
                    break
                else:
                    #rows = rows + row_actions
                    #cols = cols + col_actions
                    rows = rows - row_actions
                    cols = cols - col_actions
            # defuzzify the output
            rows = np.where(rows < half, 0, 1)
            cols = np.where(cols < half, 0, 1)
            row_indices = np.nonzero(rows)[0]
            col_indices = np.nonzero(cols)[0]
            # masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (len(row_indices), len(col_indices))
                data[row_indices[:, np.newaxis], col_indices] = np.random.uniform(low=min_value, high=max_value,
                                                                                  size=bicluster_shape)

            biclusters.append(Bicluster(row_indices, col_indices))

        return Biclustering(biclusters)

    def _calculate_msr(self, data, rows, cols):
        """Calculate the mean squared residues of the rows, of the columns and of the full data matrix."""
        mem_data = np.broadcast_to(cols, (len(rows), len(cols))) * np.broadcast_to(rows,
                                                                                   (len(cols), len(rows))).transpose()
        # no need for sub_data = data*mem_data

        data_mean = np.average(data, weights=mem_data)
        row_means = np.average(data, axis=1, weights=mem_data)
        col_means = np.average(data, axis=0, weights=mem_data)
        residues = data - row_means[:, np.newaxis] - col_means + data_mean
        squared_residues = residues * residues

        msr = np.average(squared_residues, weights=mem_data)
        row_msr = np.average(squared_residues, axis=1, weights=mem_data)
        col_msr = np.average(squared_residues, axis=0, weights=mem_data)

        return msr, row_msr, col_msr

    def _calculate_corr(self, data, rows, cols):
        """Calculate the mean squared residues of the rows, of the columns and of the full data matrix."""
        row_corr = np.zeros(len(rows))
        col_corr = np.zeros(len(cols))
        a=np.average(data, axis=1, weights=cols)[:, np.newaxis]
        mat = data - a
        rowsegma_vec = np.sqrt(np.average(mat * mat, axis=1, weights=cols))
        for i in range(len(rows)):
                #b2=(data[i] - a[i])  #np.average(data[i], weights=cols) then b2[np.newaxis,:] * mat
                rowcov_vec = np.average(mat[i] * mat, axis=1, weights=cols)
                b1=np.abs(rowcov_vec / (rowsegma_vec[i]*rowsegma_vec))
                b3 = rows[~np.isnan(b1)]
                if  np.sum(b3) > 0.0:
                    row_corr[i] = np.average(b1[~np.isnan(b1)], weights=b3)

        a=np.average(data, axis=0, weights=rows)
        mat = data -  a
        colsegma_vec = np.sqrt(np.average(mat * mat, axis=0, weights=rows))
        for i in range(len(cols)):
                #b1=(data[:,i] - a[i])[:, np.newaxis]
                colcov_vec = np.average(mat[:,i][:, np.newaxis] * mat, axis=0, weights=rows)
                b1=np.abs(colcov_vec / (colsegma_vec[i]*colsegma_vec))
                b3=cols[~np.isnan(b1)]
                if np.sum(b3) > 0.0:
                    col_corr[i] = np.average(b1[~np.isnan(b1)], weights=b3)
        avg_corr = np.average(row_corr, weights=rows)
        return avg_corr, row_corr, col_corr

class testme:

    def main():
        # load yeast data used in the original Cheng and Church's paper
        data = load_yeast_tavazoie().values
        # data = np.arange(100).reshape(10, 10)
        # data = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
        # missing value imputation suggested by Cheng and Church
        missing = np.where(data < 0.0)
        data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

        # creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
        p = SoftFLocAlgorithm(num_biclusters=2, msr_threshold=300.0, half=3)
        biclustering = p.run(data)
        print(biclustering)

    if __name__ == "__main__":
        main()
