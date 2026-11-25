"""Write a slab of a public NIfTI CT volume as a DICOM CT series in a zip.

Used to make demo uploads from public research data (for example CT-ORG,
TCIA, CC BY 4.0), which ships as NIfTI while the workbench takes DICOM zips.

    python research/nifti_to_dicom.py volume.nii.gz Z0 Z1 out.zip "Description"

Z0 and Z1 are slice indices along the volume's third axis (Z1 exclusive).
"""
import datetime
import io
import os
import sys
import zipfile

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def convert(src, z0, z1, out, description):
    img = nib.load(src)
    vol = np.asanyarray(img.dataobj)
    A = img.affine
    lps = np.diag([-1, -1, 1])                               # NIfTI RAS -> DICOM LPS
    rowdir = lps @ (A[:3, 0] / np.linalg.norm(A[:3, 0]))     # along i (DICOM columns)
    coldir = lps @ (A[:3, 1] / np.linalg.norm(A[:3, 1]))     # along j (DICOM rows)
    si, sj, sk = (float(np.linalg.norm(A[:3, c])) for c in range(3))
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    today = datetime.date.today().strftime('%Y%m%d')
    tag = os.path.basename(src).split('.')[0]
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, k in enumerate(range(z0, z1)):
            ipp = lps @ (A @ [0, 0, k, 1])[:3]
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            d = Dataset()
            d.file_meta = meta
            d.SOPClassUID, d.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
            d.Modality = 'CT'
            d.PatientName, d.PatientID = f'PUBLIC^{tag}', f'PUBLIC-{tag}'
            d.StudyDescription = description
            d.SeriesDescription = f'Thin slice {sk:.1f} mm'
            d.StudyDate = d.SeriesDate = today
            d.StudyInstanceUID, d.SeriesInstanceUID, d.FrameOfReferenceUID = study, series, frame
            d.SeriesNumber, d.InstanceNumber = 1, n + 1
            d.ImagePositionPatient = [round(float(v), 4) for v in ipp]
            d.ImageOrientationPatient = [round(float(v), 6) for v in (*rowdir, *coldir)]
            d.PixelSpacing = [sj, si]
            d.SliceThickness = sk
            d.SliceLocation = float(ipp[2])
            px = np.clip(vol[:, :, k].T, -2048, 4095).astype(np.int16)    # rows = j, cols = i
            d.Rows, d.Columns = px.shape
            d.SamplesPerPixel = 1
            d.PhotometricInterpretation = 'MONOCHROME2'
            d.BitsAllocated = d.BitsStored = 16
            d.HighBit = 15
            d.PixelRepresentation = 1
            d.RescaleIntercept, d.RescaleSlope = 0, 1
            d.WindowCenter, d.WindowWidth = 40, 400
            d.PixelData = px.tobytes()
            buf = io.BytesIO()
            pydicom.dcmwrite(buf, d, enforce_file_format=True)
            z.writestr(f'CT{n + 1:04d}.dcm', buf.getvalue())
    print(f'wrote {z1 - z0} slices to {out}')


if __name__ == '__main__':
    convert(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
            sys.argv[5] if len(sys.argv) > 5 else 'Public CT (converted from NIfTI)')
