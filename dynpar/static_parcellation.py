import gc
import random
import numpy as np
import nibabel as nb
from joblib import Memory
import dask.array as da
from nilearn.image import coord_transform
from nilearn.regions import connected_label_regions
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids
from multiprocessing import Pool
from contextlib import closing

class staticpar(object):

    __memory = Memory(location='~/.cachedir', mmap_mode='r')

    def __init__(self):
        self.__cachedir = '~/.cachedir'


    def __del__(self):
        gc.collect()
        #print('Delete static parcellation instance')


    def __get_sliding_windows(self, timepoints, window_length=100, shift=90):
        """
        Returns the indexes of sliding windows
        :param timepoints: int
        :param window_length: int
        :param shift: int
        :return: list
        """
        slices = [slice(start, start + window_length)
                  for start in range(0, timepoints, shift)]

        return slices

    def kmeans(self, tseries, n_clusters, seed, n_init=300, n_jobs=-1, max_iter=300, chunksize_voxels=3000):
        """
        Parcellate the functional MRI signal into functional parcels using the kmeans algorithm

        :param tseries: darray of shape (n_voxels, timepoints)
            functional MRI signal
        :param n_clusters: int
            Number of clusters
        :param seed:  See the documentation of dask_ml.cluster.KMeans
        :param init:  See the documentation of dask_ml.cluster.KMeans
        :param num_workers:  See the documentation of dask_ml.cluster.KMeans
        :param max_iter: See the documentation of dask_ml.cluster.KMeans
        :param tol:  See the documentation of dask_ml.cluster.KMeans
        :param init_max_iter:  See the documentation of dask_ml.cluster.KMeans
        :param precompute_distances:  See the documentation of dask_ml.cluster.KMeans
        :param copy_x:  See the documentation of dask_ml.cluster.KMeans
        :param chunksize_voxels: int
            Size of a chunk of an array
        :return: darr_labels: darray of size (n_voxels, )
            Parcellation of the brain cortex into functional parcels
        """
        random_state = seed + random.randint(0, 10000)

        random.seed(random_state)

        model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, n_init=n_init, max_iter=max_iter,
                       random_state=random_state,
                       algorithm='auto').fit(tseries.compute())

        darr_labels = da.from_array(model.labels_ + 1, chunks=(chunksize_voxels,), asarray=True)

        del model;
        gc.collect()

        return darr_labels


    def replicate_parcellation_on_sliding_windows(self, tseries, n_replications, n_clusters, chunksize_voxels=3000,
                                          seed=1, n_init = 1,  window_length=100, shift=90, num_workers=16):
        """
        Generate many parcellations across several sliding windows by replicating the same algorithm on sliding windows

        Parameters
        ----------------------------
        :param tseries: float, numpy.ndarray, shape(n_voxels, n_timepoints)
          fMRI time series

        :param n_replications int
            The number of repetitions for each sliding window parcellation

        :param n_clusters int
            The number of clusters in a seed-voxel parcellation for the whole brain

        :param chunksize_voxels: int
            Tells how many voxels are included in a chunk. The underlying array will be
            broken up into many chunks

        :param seed: n_init, (default, 1)
            See documentation of sklearn.cluster.KMeans

        :param seed: int, (default, 1)
            Initialize the random number generator

        :param scheduler: string, default, 'processes'
            The Dask scheduler used to run tasks in parallel on a machine

        :param window_length: int (default, 100)
            The number of timepoints in a sliding window
        :param shift: int (default, 90)
            The number of the non-overlapping timpoints between
            two sliding windows


        Return
        ------------------------------
        :return: Dask array, shape(n_voxels, n_replications*n_slices)
            Parcellations across sliding windows
        Notes
        ---------------------------------
        The documentation of sklearn.cluster.KMeans
       https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        """
        random.seed(seed)

        timepoints = tseries.shape[1]

        slices = self.__get_sliding_windows(timepoints, window_length=window_length, shift=shift)

        self.__n_slices = len(slices)

        with closing(Pool(processes=num_workers)) as p:

            l_rep_part_sw = p.starmap(self.kmeans, [(tseries[:, slices[rep_n%len(slices)]], n_clusters,
                                                 (seed + random.randint(0, 1000)), n_init)

                                                for rep_n in range(n_replications * len(slices))])

        del p
        gc.collect()

        parcellations_sw = da.vstack(l_rep_part_sw)

        del l_rep_part_sw;
        gc.collect()

        parcellations_sw = parcellations_sw.rechunk(chunks=(parcellations_sw.shape[0], chunksize_voxels))

        return parcellations_sw


    def get_seeds(self, nifti_img_path, connect_diag=True, min_size=100):
        """
        Get the seeds from regions in a group parcellation Nifti image
        :param nifti_img_path: str
            Path to the group parcellation Nifti image
        :param connect_diag: See the documentation of nilearn.regions.connected_label_regions

        :param min_size: See the documentation of nilearn.regions.connected_label_regions

        :return:
            l_medoids: list of tuple of shape (x,y,z)
                Each tuple represents the coordinates of the seed voxel of a region
            l_medoids_coord_mniL list of tuple of shape (x,y,z)
                Each tuple represents the coordinates of the seed voxel of a region in MNI coordinates
        """

        mist_img = nb.load(nifti_img_path)

        regions = connected_label_regions(nifti_img_path, connect_diag=connect_diag, min_size=min_size).get_data()

        labels_regions = np.unique(regions)

        # Apply the kmedoids clustering for each region
        l_medoids = []
        for label in range(1, len(labels_regions)):
            label_coords = np.argwhere(regions == label)

            kmedoids_instance = kmedoids(label_coords, [0])

            kmedoids_instance.process()

            medoid = kmedoids_instance.get_medoids()

            elem = tuple(label_coords[medoid[0]])

            l_medoids.append(elem)

        l_medoids_coord_mni = [coord_transform(medoid[0], medoid[1], medoid[2], mist_img.get_affine())
                               for medoid in l_medoids]

        return l_medoids, l_medoids_coord_mni
