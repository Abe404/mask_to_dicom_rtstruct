"""
Copyright (C) 2022 Abraham George Smith
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import time
import datetime
import pydicom
import numpy as np
import cv2 as cv
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid 


def copy_attributes(rt_ds, im_ds):
    """ copy the image attributes to the struct dataset
        many of these should be the same.
    """
    for attribute in [
        'AccessionNumber',
        'FrameOfReferenceUID',
        'InstitutionAddress',
        'InstitutionName',
        'InstitutionalDepartmentName',
        'Manufacturer',
        'ManufacturerModelName',
        'PatientBirthDate',
        'PatientID',
        'PatientName',
        'PatientSex',
        'PositionReferenceIndicator',
        'ReferringPhysicianName',
        'SeriesDate',
        'SeriesNumber',
        'SeriesTime',
        'SeriesInstanceUID',
        # consider modifying later as structure may be created by different software
        'SoftwareVersions', 
        'SpecificCharacterSet',
        'StudyDate',
        'StudyDescription',
        'StudyID',
        'StudyInstanceUID',
        'StudyTime']:
        if hasattr(im_ds, attribute):
            setattr(rt_ds, attribute, getattr(im_ds, attribute))
        
    return rt_ds

def create_rt_struct_ds(im_ds, filename, label='label', name='name'):
    """ create rt struct dicom dataset """
    SOPClassUIDForRTSTRUCT = '1.2.840.10008.5.1.4.1.1.481.3'
    # SOPInstanceUID is a globally unique identifier for a DICOM file
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SOPClassUIDForRTSTRUCT
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid 

    rt_ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    # copy patient related attributes etc from an existing image dicom file.
    rt_ds = copy_attributes(rt_ds, im_ds)
    rt_ds.SOPInstanceUID = sop_instance_uid
    rt_ds.InstanceCreationTime = str(time.time())

    # Time at which the structures were last modified.
    rt_ds.StructureSetTime = rt_ds.InstanceCreationTime
    # YYYYMMDD
    rt_ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
    rt_ds.StructureSetDate = rt_ds.InstanceCreationDate

    rt_ds.StructureSetLabel = label 
    rt_ds.StructureSetName = name
    rt_ds.Modality = 'RTSTRUCT'
    rt_ds.SOPClassUID = SOPClassUIDForRTSTRUCT
    rt_ds.is_implicit_VR = False
    rt_ds.is_little_endian = True
    add_referenced_frame_of_reference(rt_ds, im_ds)
    
    rt_ds.ROIContourSequence = Sequence()
    rt_ds.RTROIObservationsSequence = Sequence()
    rt_ds.StructureSetROISequence = Sequence()
    return rt_ds


def add_referenced_frame_of_reference(rt_ds, im_ds):
    """ tell the rtstruct to reference the image series """
    # Referenced Frame of Reference Sequence (link structure to sequence)
    frame_of_ref_sequence = Sequence()
    rt_ds.ReferencedFrameOfReferenceSequence = frame_of_ref_sequence
    frame_of_ref = Dataset()
    frame_of_ref.FrameOfReferenceUID = im_ds.FrameOfReferenceUID
    referenced_study_sequence = Sequence()
    frame_of_ref.RTReferencedStudySequence = referenced_study_sequence
    referenced_study = Dataset()
    referenced_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'

    # I'm not sure what this should be.
    # I think rt_referenced series instances can be used to store
    # the UIDs of all the corresponding serires files. I'm hoping it
    # isn't essential as it seems useless and overkill to add this here.
    # The series is already referenced by it's SeriesInstanceUID
    referenced_study.ReferencedSOPInstanceUID = None

    referenced_series_sequence = Sequence()
    referenced_study.RTReferencedSeriesSequence = referenced_series_sequence
    referenced_series = Dataset()
    referenced_series.SeriesInstanceUID = im_ds.SeriesInstanceUID
    referenced_series_sequence.append(referenced_series)
    referenced_study_sequence.append(referenced_study)
    frame_of_ref_sequence.append(frame_of_ref)


def add_contour(rt_ds, contour, name, slice_ds, color=[255,0, 0]):
    """
    rt_ds is an existing rt struct dataset
    contour is a list of lists
    each list in contours is a list of contour_points
    with the format [x,y,z,x,y,z....]
    """
    # get the max contour number found so far
    # largest ROI number may be greater than the length of the sequence (and some numbers might be missing)
    contour_num = 1
    if len(rt_ds.StructureSetROISequence):
        max_contour_num = max([s.ROINumber for s in rt_ds.StructureSetROISequence])
        contour_num = max_contour_num + 1

    roi_contour = Dataset()
    roi_contour.ROIDisplayColor = color
    contour_sequence = Sequence()
    roi_contour.ContourSequence = contour_sequence
    
    # each list of contour points_mm represent a closed polygon
    # in a single slice
    # contours do not exist accross multiple axial slices
    # maybe they should?
    for points_mm in contour:
        contour_ds = Dataset()
        contour_ds.ContourGeometricType = 'CLOSED_PLANAR'
        contour_ds.NumberOfContourPoints = str(len(points_mm) // 3)
        contour_ds.ContourData = points_mm

        # Sequence of images containing the contour.
        contour_ds.ContourImageSequence = Sequence()
        contour_im = Dataset()
        contour_im.ReferencedSOPClassUID = slice_ds.SOPClassUID
        contour_im.ReferencedSOPInstanceUID = slice_ds.SOPInstanceUID
        contour_ds.ContourImageSequence.append(contour_im)
        contour_sequence.append(contour_ds)

    roi_contour.ReferencedROINumber = contour_num
    rt_ds.ROIContourSequence.append(roi_contour)

    roi_observations = Dataset()
    roi_observations.ObservationNumber = contour_num
    roi_observations.ReferencedROINumber = contour_num
    roi_observations.ROIObservationLabel = name
    roi_observations.RTROIInterpretedType = ''
    roi_observations.ROIInterpreter = ''

    rt_ds.RTROIObservationsSequence.append(roi_observations)
    structure_set_roi = Dataset()
    structure_set_roi.ROIName = name
    structure_set_roi.ROINumber = contour_num
    structure_set_roi.ROIGenerationAlgorithm = ''
    structure_set_roi.ReferencedFrameOfReferenceUID = rt_ds.FrameOfReferenceUID
    rt_ds.StructureSetROISequence.append(structure_set_roi)


def load_image_series(dicom_dir):
    """
    Get all dicom image dataset files for a dicom series in a dicom dir.
    """
    image_series = []
    dicom_files = sorted(os.listdir(dicom_dir))
    for f in dicom_files:
        fpath = os.path.join(dicom_dir, f)
        if os.path.isfile(fpath):
            fdataset = pydicom.dcmread(fpath)
            # Computed Radiography Image Storage SOP Class UID
            # https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
            mr_sop_class_uid = '1.2.840.10008.5.1.4.1.1.4'
            ct_sop_class_uid = '1.2.840.10008.5.1.4.1.1.2'
            #  PET Image Storage
            pet_sop_class_uid = '1.2.840.10008.5.1.4.1.1.128'
            # Enhanced PET Image Storage
            pet_enhanced_sop_class_uid = '1.2.840.10008.5.1.4.1.1.130'
            # legacy PET Image Storage
            pet_legacy_sop_class_uid = '1.2.840.10008.5.1.4.1.1.128.1'
            if fdataset.SOPClassUID in [mr_sop_class_uid,
                                        ct_sop_class_uid,
                                        pet_sop_class_uid,
                                        pet_enhanced_sop_class_uid,
                                        pet_legacy_sop_class_uid]:
                image_series.append(fdataset)
            else:
                print('Excluding slice from image series as SOPClassUID is', fdataset.SOPClassUID)
                
    return image_series


def get_z_spacing_mm(image_series):
    """ this is the difference in mm between the central pixel
        in the same x,y location in two adjacent slices. """
    z_min = image_series[0].ImagePositionPatient[2]
    z_max = image_series[-1].ImagePositionPatient[2]
    z_diff = abs(z_max - z_min)
    slice_diff_mm = z_diff / (len(image_series) - 1)
    return slice_diff_mm

def pixel_to_mm_points(contour_points, image_series):
    """
    convert contour points in pixel location
    to the location in mm (relative to origin)
    i.e in patient/dicom space
    """
    first_slice_ds = image_series[0]
    # physical distance of each pixel in mm. 
    # row space is y and column space is x
    pixel_row_spacing, pixel_column_spacing = first_slice_ds.PixelSpacing

    mm_per_z_pixel = get_z_spacing_mm(image_series)
    mm_per_x_pixel = pixel_column_spacing
    mm_per_y_pixel = pixel_row_spacing

    pixel_per_x_mm = 1/mm_per_x_pixel
    pixel_per_y_mm = 1/mm_per_y_pixel
    pixel_per_z_mm = 1/mm_per_z_pixel

    offset_x_mm, offset_y_mm, offset_z_mm = first_slice_ds.ImagePositionPatient
    i = 0
    while i < len(contour_points):
        x, y, z = contour_points[i:i+3]
        # first convert contour points from pixel to mm scale.
        x_mm = x * mm_per_x_pixel
        y_mm = y * mm_per_y_pixel
        z_mm = z * mm_per_z_pixel
        # and now assign the offset so they start at 0 
        x_mm = x_mm + offset_x_mm
        y_mm = y_mm + offset_y_mm
        z_mm = z_mm + offset_z_mm
        contour_points[i:i+3] = x_mm, y_mm, z_mm
        i += 3
    return contour_points


def get_slice_contours(mask_axial_slice, z):
    if not np.any(mask_axial_slice):
        return []
    mask_axial_slice = mask_axial_slice.astype(np.uint8)
    # From https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    # contours is a Python list of all the contours in the image.
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    # For mode information see:
    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    contour_mode = cv.RETR_LIST
    cv_contours, _ = cv.findContours(mask_axial_slice, contour_mode, cv.CHAIN_APPROX_SIMPLE)
    contours = []
    for cv_contour in cv_contours:
        contour_points = []
        for pair in cv_contour:
            x, y = pair[0]
            contour_points += [x, y, z]
        contours.append(contour_points)
    return contours


def get_3d_image(dicom_series_path):
    """ return dicom images as 3D numpy array """
    image_series_files = load_image_series(dicom_series_path)
    first_im = image_series_files[0]
    height, width = first_im.pixel_array.shape
    depth = len(image_series_files)
    image = np.zeros((depth, height, width))
    for i, im in enumerate(image_series_files):
        image[i] = im.pixel_array
    return image


def add_dicom_struct(series_path, struct_path, struct_name, mask):
    """ series path - full path to the folder containing the dicom series images
        struct path - full path to the existing rtstruct file.
        struct_name - name of the new struct to be added.
        mask - numpy binary ndarray for the structure
    """
    image_series_files = load_image_series(series_path)
    struct_ds = pydicom.dcmread(struct_path)
    contours_mm = mask_to_contours_mm(mask, image_series_files)
    add_contour(struct_ds, contours_mm, struct_name, struct_ds)
    print('save dicom struct as', struct_path)
    struct_ds.save_as(struct_path, write_like_original=True)
     

def mask_to_contours_mm(mask, image_series_files):
    # list of lists of points 
    contours_mm = []
    for i in range(mask.shape[0]):
        axial_slice = mask[i, :, :]
        series_slice_ds = image_series_files[i]
        cv_contours = get_slice_contours(axial_slice, i)
        for points_list in cv_contours:
            points_mm = pixel_to_mm_points(points_list, image_series_files)
            contours_mm.append(points_mm)
    return contours_mm

    
def save_dicom_struct(dicom_series_path, mask, fname, struct_name):
    """"
        # Example usage
        import dicom_utils as dcm
        dcm.save_dicom_struct(dicom_series_path, numpy_binary_mask,
                              fname='struct.dcm',
                              struct_name='struct_name')

    """

    image_series_files = load_image_series(dicom_series_path)
    new_struct_ds = create_rt_struct_ds(image_series_files[0], fname,
                                        label=struct_name, name=struct_name)
    contours_mm = mask_to_contours_mm(mask, image_series_files)
    add_contour(new_struct_ds, contours_mm, struct_name, series_slice_ds)
    output_path = os.path.join(dicom_series_path, fname)
    print('save dicom struct as', output_path)
    new_struct_ds.save_as(output_path, write_like_original=False)


