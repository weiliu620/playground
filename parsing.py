"""Parsing code for DICOMS and contour files"""

import dicom
from dicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw

import os
import matplotlib.pyplot as plt


def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))

    return coords_lst


def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        dcm_dict = {'pixel_data' : dcm_image}
        return dcm_dict
    except InvalidDicomError:
        return None


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask


def viz(dcm_dir, ct_dir, R = 4, C = 4):
    """ 
    visualize contour overlayed  on Dicom image for single subject.
    
    """

    all_im = []

    n = 0


    for fname in sorted(os.listdir(ct_dir)):
        
        if fname.endswith('.txt') and not fname.startswith('.'):
            coords_list = parse_contour_file(os.path.join(ct_dir, fname) )
            
            slice_idx = fname[8:12].strip('0')
            
            dcm_file = os.path.join(dcm_dir, slice_idx + '.dcm')
            
            dcm_im = parse_dicom_file(dcm_file)['pixel_data']

            mask_im = poly_to_mask(coords_list, dcm_im.shape[0], dcm_im.shape[1])

            all_im.append( (dcm_im, mask_im, slice_idx) )

            print("Done with {} and {}".format(fname, dcm_file))
            n += 1
            if n == R*C:
                break
            

            
    fig, ax = plt.subplots(R, C)
    # cmap = plt.cm.get_cmap("autumn")
    # cmap.set_under(alpha = 0)  
    for r in range(R):
        for c in range(C):
            n = r * C + c
            if n <= len(all_im):
                ax[r, c].imshow(all_im[n][0], cmap = 'gray')
                ax[r, c].imshow(all_im[n][1], alpha = 0.1, cmap = 'autumn')
                ax[r, c].set_title("{}".format(all_im[n][2]))
                ax[r, c].axis('off')

    plt.show(block = False)

            

