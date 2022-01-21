# mask_to_dicom_rtstruct
Convert numpy mask to dicom rtstruct

Example use case: Creating a new file named struct.dcm in the dicom_series_path directory.

```
import dicom_utils as dcm
dcm.save_dicom_struct(dicom_series_path, numpy_binary_mask, fname='struct.dcm', struct_name='struct_name')
```
