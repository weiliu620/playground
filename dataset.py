import h5py
import numpy as np
import os

from parsing import parse_contour_file, parse_dicom_file, poly_to_mask

def build_dataset(ct_dir, dcm_dir, link_file, out_h5):
    """ build a h5 file with two datasets in it: 'image' and 'label'.


    ct_dir: contour file root dir, i.e. 'contourfiles'
    dcm_dir: dicom file root dir, i.e. 'dicoms'

    out_h5: output h5 file. 
    """

    h5_fh = h5py.File(out_h5, 'w-')

    links = np.genfromtxt(link_file, skip_header = 1, delimiter = ',', dtype = 'unicode')
    links_dic = {row[1]:row[0] for row in links}

    for subject_dir in sorted(os.listdir(ct_dir)):

        # Find original dicom ID. Will be used as dir name
        orig_id = links_dic[subject_dir]
        
        sub_con_dir = os.path.join(ct_dir, subject_dir, 'i-contours')
        
        for fname in sorted(os.listdir(sub_con_dir)):

            if fname.endswith('.txt') and not fname.startswith('.'):

                print("Working on subject {}, slice {}".format(subject_dir, fname))
                
                coords_list = parse_contour_file(os.path.join(sub_con_dir, fname) )
            
                slice_idx = fname[8:12].strip('0')

                dcm_file = os.path.join(dcm_dir, orig_id, slice_idx + '.dcm')
            
                dcm_im = parse_dicom_file(dcm_file)['pixel_data']

                mask_im = poly_to_mask(coords_list, dcm_im.shape[0], dcm_im.shape[1])
                mask_im = mask_im.astype(np.bool8)


                # Need dicom image shape to create dataset. 
                if 'image' in h5_fh:
                    pass
                
                else:
                    image_dset = h5_fh.create_dataset(
                        'image', 
                        shape =  (0, dcm_im.shape[0], dcm_im.shape[1]),
                        maxshape = (None, dcm_im.shape[0], dcm_im.shape[1]), 
                        dtype = np.float32)
                    
                    label_dset = h5_fh.create_dataset(
                        'label',
                        shape =  (0, dcm_im.shape[0], dcm_im.shape[1]),
                        maxshape = (None, dcm_im.shape[0], dcm_im.shape[1]),
                        dtype = np.bool8)

                sh = image_dset.shape
                image_dset.resize((sh[0]+1, sh[1], sh[2]))
                label_dset.resize((sh[0]+1, sh[1], sh[2]))

                image_dset[-1, ] = dcm_im
                label_dset[-1, ] = mask_im

    h5_fh.close()

    
class dataset(object):
    """ A simple dataset class that just provide next_batch method. """


    def __init__(self, h5_file):
        """ 
        h5_file: input hdf5 file with 'image' and 'label' datasets. """
        fh = h5py.File(h5_file, 'r')
        self._image = fh['image']
        self._label = fh['label']

        self._num_epoch = 0
        self._cur_pos = 0

        self._num_samples = fh['image'].shape[0]

        self._perm = list(range(self._num_samples))

        # shuttle the set of index is equivalent to random sampling.
        np.random.shuffle(self._perm)

    def __del__(self):
        """ """
        pass

    def next_batch(self, batch_size):
        """
        return a tuple of (image, label). 

        image: NWH
        label: NWH

        """
        if (self._cur_pos + batch_size >= self._num_samples):
            self._num_epoch += 1
            self._perm = list(range(self._num_samples))
            np.random.shuffle(self._perm)
            self._cur_pos = 0

        end = self._cur_pos + batch_size

        self._orig_indicies = self._perm[self._cur_pos:end]
        self._orig_indicies.sort()

        image_patch = self._image[self._orig_indicies,]
        label_patch = self._label[self._orig_indicies,]

        self._cur_pos += batch_size

        return (image_patch, label_patch)
        

    
