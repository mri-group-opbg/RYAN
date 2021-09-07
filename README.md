# STABILITYCALC

Stabilitycalc is a package we used to monitor the stability of fMRI system.  

We use a version of the fBIRN procedure to run monthly phantom FUNSTAR quality 
assurance tests.
These phantom scans are then converted to NIFTI format and analyzed 
with stabilitycalc to look for system instabilities that could degrade
our fMRI data.

The stabilitycalc package generates reports (pdf file) for individual scans, 
and tabulates performance over time, generating summary reports that put
current performance into the context of normal operation.  

## Requirements

Stabilitycalc checks for some of its requirements before running. In general,
it requires a standard numpy/scipy/matplotlib environment, as well as
`nibabel`, `nipype`, `dicom`, `pandas`, `seaborn`, `pdfkit` and `mako`.

## How to use

1) Clone GitFolder

2) Create an input folder and a subfolder named "nii". Copy one dicom acquisition file inside input folder and copy nifti files inside "nii" subfolder

3) Check nifti files' names: 
	- acquisition file should contain "acquisition" in its name
	- shimming file (if used) should contain "shimming" in its name 
	- no shimming file (if used) should contain "no_shimming" in its name

4) Launch command:

		python path_git_folder/script/stabilitycalc.py path_to_output path_to_input_folder 1 0

## INPUT

### REQUIREMENT INPUTS:
        - dirname: output path to store results (path)
        - dicompath: path to find dicom file/s (path)
        - starttime: start time point to begin analysis (int)
        - sliceshift: shift from center axial slice (int)

### OPTIONAL INPUTS:
        - shimmingfilename (optional): name of shimming file nii (string)
        - noshimmingfilename (optional): name of no-shimming file nii (string)
        - initxcenter (optional): xcenter (int)
        - initycenter (optional): ycenter (int)
        - initzcenter (optional): zcenter (int)

## OUTPUT

A folder will be create on the selected output path (dirname) and result of computation will be stored inside:

	- images (.png)
	- analysissummary.txt
	- dataquality.txt
	- output.html 
	- output.pdf



