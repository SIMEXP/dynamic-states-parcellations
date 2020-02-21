import gc
import os
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import seaborn as sns
import nibabel as nb
import dask.array as da
from dask import compute
from joblib import Memory
import pandas as pd
from nilearn.input_data import NiftiMasker
from multiprocessing import Process, Pool
import SharedArray
from contextlib import closing
from dynpar import staticpar

sns.set()


class dynamic_parcellation(staticpar):

    def __init__(self):

        staticpar.__init__(self)

        self.__memory = Memory(location=os.getcwd() + '/cachedir',
                               mmap_mode='r')

    def __del__(self):

        staticpar.__del__(self)


    def __get_mask_medoid(self, mask_img, coords):

        mask_medoid = np.zeros(mask_img.shape)

        mask_medoid[coords] = 1

        mask_medoid = nb.Nifti1Image(mask_medoid, mask_img.affine)

        masker = NiftiMasker(mask_img)

        masker.fit(mask_img)

        masker.standardize = False

        mask_medoid = masker.transform(mask_medoid)

        mask_medoid = mask_medoid.reshape([mask_medoid.shape[1]]) > 0

        del masker;
        gc.collect()

        return mask_medoid


    def get_onhot_vect(self, parts, mask_medoid, sw_idx):

        one_hot = parts[sw_idx, :] == parts[sw_idx, :][mask_medoid]

        return one_hot


    def mask_seed_region(self, parts, mask_img, medoid_coords, num_workers=16):
        """
        Generate encoders such that all voxels included in the cluster of the
        medoid are assigned the one value and zero otherwise

        :param parts: Dask array, shape(n_sliding windows, n_voxels)
            Parcellations across sliding windows
        :param mask_img: Nifti Image, shape (x,y,z)
            Mask of the data
        :param medoid_coords: tuple, shape (x,y,z)
            medoid of a region
        :param chunksize_voxels: int
            Number of voxels in one chunk
        :return: Dask array, shape(n_sliding windows, n_voxels)
            Masked parcellations for one medoid
        """

        n_sw = parts.shape[0]

        mask_medoid = self.__get_mask_medoid(mask_img, medoid_coords)

        tmp = np.unique(mask_medoid)

        if tmp.shape[0] == 1:

            return None

        else:

            with closing(Pool(processes=num_workers)) as p:

                l_masked_parts = p.starmap(self.get_onhot_vect, [(parts, mask_medoid, sw_idx)

                                           for sw_idx in range(0, n_sw)])

            del mask_medoid;

            gc.collect()

            darr_masked_parts = da.vstack(l_masked_parts)

            del l_masked_parts
            gc.collect()

            return darr_masked_parts


    def mask_brain_regions_across_medoids(self, parts, mask_img, medoids, num_workers=8):
        """
        Generate encoders such that all voxels included in the cluster of
        medoid are assigned the one value and zero otherwise. This process is
        repeated for all medoids extracted from brain regions using a template parcellation.

        :param medoids:
       :param parts: Dask array, shape(n_sliding windows, n_voxels)
            Parcellations across sliding windows
        :param part_template_img: Nifti image
            A parcellation template
         :param mask_img: Nifti Image, shape (x,y,z)
            Mask of the data
        :param chunksize_voxels: int
            Number of voxels in one chunk
        :return: list,  shape(n_medoids, n_sliding windows, n_voxels)
            Masked parcellations across medoids
        """
        l_masked_part_medoids = [(self.mask_seed_region)(parts, mask_img, medoid, num_workers=num_workers)
                                 for medoid in medoids]

        l_masked_part_medoids = compute(*l_masked_part_medoids, num_workers=num_workers)

        return l_masked_part_medoids


    def dice(self, part1, part2):
        d = 2 * np.sum((part1 == 1) & (part2 == 1)) / (np.sum(part1) + np.sum(part2))

        return d


    def calc_row(self, masked_parts, n_part, sw1):

        vect = np.zeros((n_part, ))

        for sw2 in range(sw1 + 1, n_part):

            vect[sw2] = self.dice(masked_parts[sw1, :], masked_parts[sw2, :])

        return vect


    def get_similarity_sliding_windows(self, masked_parts, num_workers=16):
        """
        Compute the dice matrix between encoders of sliding window parcellations
        :param masked_labels:
        :return: Dask array, shape (n_parcellations, n_parcellations)
            The spatial similarity matrix between sliding window binary
            parcellations (or encoders) for one cluster
        """

        if masked_parts is not None:

            n_part = masked_parts.shape[0]

            with closing(Pool(processes=num_workers)) as p:

                l_dice_mtx = p.starmap(self.calc_row, [(masked_parts, n_part, sw1)

                                                        for sw1 in range(n_part)])

            dice_mtx = np.asarray(l_dice_mtx)

            del l_dice_mtx
            gc.collect()

            # The dice scores between pairwise sliding windows
            dice_mtx = dice_mtx + np.transpose(dice_mtx)

            idx = np.diag_indices(dice_mtx.shape[0])

            dice_mtx[idx] = 1

            gc.collect()

            return dice_mtx
        return None


    def __compute_spatial_corr_mtx_sliding_windows_region(self, masked_parts):
        """
        Generate the spatial correlation matrix between encoders of sliding windows
        of one cluster

        :param encoders: Dask aray of shape (n_sliding window, n_voxels)
            Seed-based parcellations fro sliding windows

        :param cluster_label: str
            The name of the seed network

        :return: Dask array, shape (n_parcellations, n_parcellations)

            The spatial similarity matrix between sliding window binary
            parcellations (or encoders) for one cluster
        """
        n_part = masked_parts.shape[0]

        spatial_corr_mtx = np.zeros((n_part, n_part))

        for sw1 in range(n_part - 1):
            for sw2 in range(sw1 + 1, n_part):
                corr, _ = scipy.pearsonr(masked_parts.compute()[sw1, :],
                                         masked_parts.compute()[sw2, :])
                spatial_corr_mtx[sw1, sw2] = corr

        # The spatial correlation between pairwise sliding windows
        spatial_corr_mtx = spatial_corr_mtx + np.transpose(spatial_corr_mtx)

        idx = np.diag_indices(n_part)

        spatial_corr_mtx[idx] = 1

        darr_spatial_corr_mtx = da.from_array(spatial_corr_mtx, chunks=(n_part, n_part))

        del spatial_corr_mtx

        gc.collect()

        return darr_spatial_corr_mtx

    def compute_similarity_mtx(self, masked_parts_medoid, similarity='dice'):

        if similarity == 'pearson':

            spatial_similarity_mtx = self.__compute_spatial_corr_mtx_sliding_windows_region(masked_parts_medoid)

        else:

            spatial_similarity_mtx = self.get_similarity_sliding_windows(masked_parts_medoid)

        return spatial_similarity_mtx

    def assign_sliding_windows_to_states(self, similarity_mtx, cluster_size_threshold=0.1, min_dice_threshold=0.3, method='average', vmin=0, vmax=1, square=True,
                                         cmap="viridis", get_leaves=True, no_plot=True, leaf_rotation=90.,
                                         leaf_font_size=12., show_contracted=True, ):
        """
        Apply a ward clustering on the spatial similarity matrix to assign each sliding window to a state/cluster
        :param similarity_mtx: ndarray of shape (n_parcellations, n_parcellations)
            Dice similarity matrix of pairs of one-hot arrays (seed-based parcellations)
        :param vmin: int
            Minimum of the Dice score
        :param vmax: int
            Maximum of the Dice score
        :param square: bool
            Square to highlight the cluster in the similarity matrix
        :param cmap: str
            color map
        :param path_figure: str
            Folder path to save the figures of the similarity matrices
        :param network_label: str
            Seed network label
        :param subject: str
            Subject name
        :param get_leaves: See the documentation of scipy.cluster.hierarchy.dendrogram
        :param no_plot: See the documentation of scipy.cluster.hierarchy.dendrogram
        :param leaf_rotation: See the documentation of scipy.cluster.hierarchy.dendrogram
        :param leaf_font_size: See the documentation of scipy.cluster.hierarchy.dendrogram
        :param show_contracted: See the documentation of scipy.cluster.hierarchy.dendrogram
        :param cophenetic_dist: See the documentation of scipy.cluster.hierarchy.dendrogram
        :return:
            states
            final_n_states
            states_idx
            cdw_sorted
            ordered_labels
        """
        cophenetic_distance = 1 - min_dice_threshold

        if similarity_mtx is not None:

            iu = np.triu_indices(similarity_mtx.shape[0], 1)

            dist_part = 1 - similarity_mtx[iu]

            # Compute the linkage matrix
            hier_clustering = linkage(dist_part, method=method, optimal_ordering=True)

            print("Seed-parcellations similarity matrix")

            res = dendrogram(hier_clustering,  show_leaf_counts=True, no_labels=True,
                             get_leaves=get_leaves, no_plot=no_plot, leaf_rotation=leaf_rotation, p=15,
                             leaf_font_size=leaf_font_size, show_contracted=show_contracted)

            # Identify the number of states according to a threshold, cut on this distance
            # Get flat clusters (by excluding leaves associated to a distance > dist)
            states = fcluster(hier_clustering, cophenetic_distance, criterion='distance')

            order = res.get('leaves')

            # Cut the tree to get the number of clusters(or states) according to the specified threshold dist
            final_n_states = max(states)

            states -= 1

            # Medoid parcels ordered according to their association to states
            ordered_states = states[order]

            part = pd.DataFrame(data=ordered_states, columns=["Parcel"], index=order)

            # Generate and reorder the similarity matrix
            sim_order = similarity_mtx[order, :][:, order]

            sns.heatmap(sim_order, square=square, cmap=cmap, annot=False, vmin=vmin, vmax=vmax)

            val, states_unique_val = scipy.unique(part, return_index=True)

            states_idx = scipy.sort(states_unique_val)

            ordered_labels = []

            for ii in range(0, states_idx.shape[0] - 1):

                if (states_idx[ii + 1] - states_idx[ii] >= (cluster_size_threshold * similarity_mtx.shape[0])):
                    ordered_labels.append(ordered_states[states_idx[ii]])

            dim = similarity_mtx.shape[0] - states_idx[final_n_states - 1]

            if (dim >= (cluster_size_threshold * similarity_mtx.shape[0])):

                ordered_labels.append(ordered_states[states_idx[final_n_states - 1]])

            del hier_clustering;
            del res;
            gc.collect()

            return  states, ordered_labels

        return np.array([])


    def generate_stability_for_medoid(self, masked_parts_medoid, spatial_states_for_sw, l_labels_sorted):
        """
        Generate Stability maps for dynamic parcels for one seed

        :param masked_parts_medoid: Dask array
        :param spatial_states_for_sw:
        :param l_labels_sorted: list
        :param chunksize_voxels: int
            Size of the chunk in an array
        :return:
            darr_stab_maps
            len(l_labels_sorted)
        """
        try:
            SharedArray.delete('stab_maps')
        except:
            pass

        stab_maps = SharedArray.create('stab_maps', (len(l_labels_sorted), masked_parts_medoid.shape[1]))

        def compute_stability_map(masked_parts_medoid, spatial_states_for_sw, state, idx):

            stab_maps[idx, :] = masked_parts_medoid[spatial_states_for_sw == state, :].mean(axis=0)

        processes = []
        idx = 0
        for state in l_labels_sorted:
            process = Process(target=compute_stability_map, args=(masked_parts_medoid, spatial_states_for_sw, state, idx))
            processes.append(process)
            process.start()
            idx +=1

        for process in processes:
            process.join()

        SharedArray.delete('stab_maps')

        return stab_maps


    def get_stability_maps_medoid(self, part_sw_sessions, mask_img, medoid, similarity, cluster_size_threshold=0.1,
                                  min_dice_threshold=0.7):

        masked_parts_medoid = self.mask_seed_region(part_sw_sessions, mask_img, medoid)

        if masked_parts_medoid is not None:

            dice_mtx_medoid = self.compute_similarity_mtx(masked_parts_medoid, similarity=similarity)

        if dice_mtx_medoid is not None:

            spatial_states_for_sw_medoid, \
            l_labels = self.assign_sliding_windows_to_states(dice_mtx_medoid,
                                                              cluster_size_threshold=cluster_size_threshold,
                                                              min_dice_threshold=min_dice_threshold)

        del dice_mtx_medoid;
        gc.collect()

        stab_maps_medoid = self.generate_stability_for_medoid(masked_parts_medoid,
                                                                   spatial_states_for_sw_medoid,
                                                                   l_labels)

        del spatial_states_for_sw_medoid;
        del l_labels
        gc.collect()

        return stab_maps_medoid


    def generate_stability_for_dynamic_brain_parcels(self, darr_tseries, mask_img, medoids, n_replications, n_clusters,
                                                     cluster_size_threshold=0.1, min_dice_threshold=0.7,
                                                     similarity='dice', chunksize_voxels=3000, seed=1, n_init=1,
                                                     window_length=100, shift=90):
        """
        Generate stability maps for dynamic parcels across states and medoids
        :param n_replications:
        :param darr_tseries:
        :param mask_img: Nifti Image Object, shape (x,y,z)
            The mask
        :param medoids: list of tuple (x,y,z)
            The coordinates of the medoids of regions
        :param similarity: str, default ('dice')
            The similarity measure between sliding window parcellations
            Possible values ('dice', 'pearson')
        :param chunksize_voxels: int
            Tells how many voxels are included in a chunk. The underlying array will be
            broken up into many chunks
        :param seed: int, (default, 1)
            Initialize the random number generator
        :param scheduler: string, default, 'processes'
            The Dask scheduler used to run tasks in parallel on a machine
        :param num_workers: int, (default, 4)
            Sets the number of processes or threads to use
        :param init: string, (default, 'k-means++')
            See the documentation of dask_ml.cluster.KMeans
        :param oversampling_factor: int (default, 2)
            See the documentation of dask_ml.cluster.KMeans
         :param max_iter: int (default, 20)
            See the documentation of dask_ml.cluster.KMeans
        :param tol: float (default, 0.0001)
            See the documentation of dask_ml.cluster.KMeans
        :param init_max_iter: int (default, 100)
            See the documentation of dask_ml.cluster.KMeans
        :param precompute_distances: string (default, 'auto')
            See the documentation of dask_ml.cluster.KMeans
        :param copy_x: bool (default, True)
            See the documentation of dask_ml.cluster.KMeans
        :param window_length: int (default, 100)
            The number of timepoints in a sliding window
        :param shift: int (default, 90)
            The number of the non-overlapping timpoints between
            two sliding windows
        :return:
            l_darr_stab_maps: list of Dask arrays
                Stability maps for each medoid (n_states, n_voxels)
            l_dwt_medoids: list of dynamic states dwell time
                List of dwell time for each state
            l_labels: list of arrays
                List including states labels
        ---------------------------
        Note: Some medoids not falling in the grey matter are not included
        -----------------------------
        Note for myself:
        You cannot use directly the dynamic parcels as the stability map is based on the use of sliding windows maps
        averaged.generate_stability_for_dynamic_brain_parcels
        """
        part_sw_sessions = self.replicate_parcellation_on_sliding_windows(darr_tseries, n_replications, n_clusters,
                                                                          chunksize_voxels=chunksize_voxels, seed=seed,
                                                                          n_init=n_init, window_length=window_length,  shift=shift)


        l_maps = []
        for medoid in medoids:

            l_maps.append(self.get_stability_maps_medoid(part_sw_sessions, mask_img, medoid,
                                                        similarity, cluster_size_threshold=cluster_size_threshold,
                                                        min_dice_threshold=min_dice_threshold))


        del part_sw_sessions;

        gc.collect()

        return l_maps


