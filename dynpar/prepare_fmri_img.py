import os
import gc
import glob

import sklearn as sk
import nibabel as nb
from joblib import Memory
from nilearn import plotting
from nilearn.signal import clean
from nilearn.masking import unmask
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
import dask.array as da
import matplotlib.pyplot as plt

from dynpar.confounds_loader import load_confounds

class FuncMRIManagement(object):
    __memory = Memory(location='~/.cachedir', mmap_mode='r')


    def __init__(self):
        self.__cachedir = os.getcwd() + '/cachedir'
        self.__memory = Memory(location=os.getcwd() + '/cachedir',
                               mmap_mode='r')

    def __del__(self):
        gc.collect()


    def __locate_files_subject(self, path_to_data, pattern):
        '''Locate all files matching supplied filename pattern'''

        pattern = path_to_data + pattern

        list_files_subject = glob.glob(pattern, recursive=True)

        return sorted(list_files_subject)


    def __normalize_data(self, tseries):
        """Returns the normalized time series to the zero mean and unit variance
        """
        scaler = sk.preprocessing.StandardScaler()

        scaler.fit(tseries)

        tseries = scaler.transform(tseries)

        tseries = tseries.T

        return tseries


    def load_mask(self, part_img_filename):
        """
        Load mask and resample it on the same resolution as one of studied fMRI data
        """
        img = nb.load(part_img_filename)

        mask_data = img.get_data() > 0

        mask_img = nb.Nifti1Image(mask_data.astype(int), img.affine)

        return mask_img


    def __img_to_tseries(self, filename_img, mask_img, confounds_filename=None, smoothing_fwhm=None,
                         strategy=["minimal"], chunksize=3000):

        img = nb.load(filename_img)

        img = resample_to_img(img, mask_img)

        if confounds_filename is not None:

             confounds = load_confounds(confounds_filename, strategy=strategy)

        masker = NiftiMasker(mask_img, smoothing_fwhm=smoothing_fwhm)

        masker.fit(img)

        masker.mask_img_ = mask_img

        tseries = masker.transform(img)

        if confounds_filename is not None:

            confounds=confounds.values

            tseries = clean(tseries, confounds=confounds)

        else:

            tseries = clean(tseries)

        tseries = self.__normalize_data(tseries)

        darr_tseries = da.from_array(tseries, chunks=(chunksize, tseries.shape[1]))

        return darr_tseries


    def load_tseries(self, path_to_data, mask_img, idx_first_session=0, idx_last_session=1, smoothing_fwhm=6,
                     regress_confounds=False, strategy=["minimal"], pattern_nii="*bold.nii.gz",
                     pattern_confounds="*regressors.tsv"):
        """
        Extract time series and conatenate data from a list of function MRI runs
        :param path_to_data: str
            Folder path to the Nifti functional MRI images
        :param mask_img: Nifti image
            Mask image
        :param idx_first_session: int, default=0
            Index of the first functional MRI image
        :param idx_last_session: int, default = 1
            Index of the last functional MRI image
        :param smoothing_fwhm: int
            See documentation of nilearn.input_data.NiftiMasker
        :param regress_confounds: bool
            Confounds regression
        :param strategy: str : "minimal", "motion", "matter", "high_pass_filter", "compcor", default= "minimal",
            Confounds regression strategy, "minimal": white matter regression, high_pass_filter and motion regression
        :param pattern_nii: str, default="*bold.nii.gz"
            Pattern of Nifti functional MRI filenames
        :param pattern_confounds: str, default="*regressors.tsv"
            Pattern of csv confounds files
        :return:

            arr_tseries: Dask arrays of shape (Voxels, timepoints)

        See documentation https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiMasker.html
        """
        try:

            path_to_data = os.path.join(path_to_data, '')

            l_files_nii = self.__locate_files_subject(path_to_data, pattern_nii)

            if idx_last_session > len(l_files_nii):

                idx_last_session = len(l_files_nii)

            if regress_confounds:

                l_confounds = self.__locate_files_subject(path_to_data, pattern_confounds)

                assert len(l_files_nii) == len(l_confounds)

                # Regressing out confounds
                l_tseries = [(self.__img_to_tseries)(l_files_nii[idx], mask_img, confounds_filename=l_confounds[idx],
                                                            smoothing_fwhm=smoothing_fwhm, strategy=strategy)

                             for idx in range(idx_first_session, idx_last_session)]
            else:
                # Without further preprocessing except the spatial smoothing
                l_tseries = [(self.__img_to_tseries)(l_files_nii[idx], mask_img,
                                                            smoothing_fwhm=smoothing_fwhm)
                             for idx in range(idx_first_session, idx_last_session)]

            arr_tseries = da.concatenate(l_tseries, axis=1)

            del l_tseries;
            gc.collect()

        except:
            raise Exception

        return arr_tseries


    def visualize(self, path_output_results, n_states, network_label, mask_img,
                  subject='', threshold=0.5, annotate=False, vmax=1, display_mode='x',
                  cut_coords=[-52, -38, -28, -9, -4, 2, 7, 34, 38],
                  resampling_interpolation='continuous', movie='Rest'):

        cmap_name = ["Blues", "Reds", "Greens", "Purples", "hot"]

        if os.path.exists(path_output_results) == False:
            os.mkdir(path_output_results)

        for id_state in range(n_states):

            nii_filename = 'map_' + subject + '_net' + str(network_label) + '_state' + str(id_state) + '.nii'

            nii_path = os.path.join(path_output_results, nii_filename)

            state_img = nb.load(nii_path)

            cmap = plt.cm.get_cmap(cmap_name[id_state])

            title = "State " + str(id_state + 1) + "-" + movie + "-" + network_label + "-" + subject

            plotting.plot_stat_map(state_img, bg_img=mask_img, threshold=threshold,
                                   annotate=annotate, cmap=cmap, vmax=vmax,
                                   display_mode=display_mode, cut_coords=cut_coords,
                                   title=title, resampling_interpolation=resampling_interpolation)
            plt.show()

    def from_array_to_vol(self, array, nii_output_filename, mask_img):
        """
        Convert an array to a volume
        :param array:
        :param nii_output_filename:
        :param mask_img:
        :return:
        """

        img_3D = unmask(array, mask_img, order='F')

        nb.save(img_3D, nii_output_filename)


    def from_array_to_nifti(self, nii_output_filename, array, mask_img):
        """
        Convert an array to a nitfi image and save it
        :param nii_output_filename:
        :param array: Dask array, shape(n_voxels)
        :param mask_img:
        :return:
        """
        img_3D = unmask(array, mask_img, order='F')

        nb.save(img_3D, nii_output_filename)


    def save_stability_maps(self, darr_stab_maps_medoid, network_label, path_maps, mask_img, subject=''):

        if os.path.exists(path_maps) == False:
            os.makedirs(path_maps)

        for idx_state in range(darr_stab_maps_medoid.shape[0]):
            nii_output_filename = os.path.join(path_maps,
                                               'map_' + subject + '_net' + str(network_label) + '_state' + str(
                                                   idx_state)[0])
            img_3D = unmask(darr_stab_maps_medoid[idx_state, :], mask_img, order='F')

            nb.save(img_3D, nii_output_filename)


