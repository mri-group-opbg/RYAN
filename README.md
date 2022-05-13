# RYAN - Rin stAbilitY quAlity coNtrol

RYAN is a package we used to monitor the stability of fMRI system.  

We use a version of the fBIRN procedure to run monthly phantom FUNSTAR quality 
assurance tests.
These phantom scans are then converted to NIFTI format and analyzed 
with stabilitycalc to look for system instabilities that could degrade
our fMRI data.

The stabilitycalc package generates reports (pdf file) for individual scans, 
and tabulates performance over time, generating summary reports that put
current performance into the context of normal operation.  

## Requirements

<p>RYAN checks for some of its requirements before running. In general,
it requires a standard numpy/scipy/matplotlib environment, as well as
`nibabel`, `nipype`, `dicom`, `pandas`, `seaborn`, `pdfkit` and `mako`.</p>
<p><b>NB</b>: pdfkit may request `wkhtmltopdf` tool to work correctly. If the PDF report
file is not generated, install <a href="https://wkhtmltopdf.org/downloads.html" target="_blank">wkhtmltopdf</a>. 
The script will automatically serch inside the computer for the wkhtmltopdf folder: it is strongly recommended to give the wkhtmltopdf folder path as input to a faster script run</p>


## How to use

1) Clone GitFolder

2) Create an input folder and a subfolder named "nii". Copy one dicom acquisition file inside input folder and copy nifti files inside "nii" subfolder.

	<ins>Folder organization example</ins>:
	* input folder
		* nii
			* first_acquisition.nii
			* shimming.nii (optional)
			* no-shimming.nii (optional)
		* 001.dcm

3) Check nifti files' names: 
	- acquisition file should contain "acquisition" in its name
	- shimming file (if used) should contain "shimming" in its name 
	- no shimming file (if used) should contain "no_shimming" in its name

4) Launch RYAN_runner.py script: 
	* with only required inputs:
		```
		cd path path_git_folder/script
		python script/RYAN_runner.py path_to_output path_to_input_folder 1 0
		```
		
	* with required inputs and wkhtmltopdf path (<b><ins>strongly recommended</ins></b>):
		```
		cd path path_git_folder/script
		python RYAN_runner.py path_to_output path_to_input_folder 1 0 path_to_wkhtmltopdf_folder
		```

## INPUT

### REQUIRED INPUTS
- dirname: output path to store results (path)
- dicompath: path to find dicom file/s (path)
- starttime: start time point to begin analysis (int)
- sliceshift: shift from center axial slice (int)

### OPTIONAL INPUTS
- wkh: path to wkhtmltopdf installation (string)
- shimmingfilename: name of shimming file nii (string)
- noshimmingfilename: name of no-shimming file nii (string)
- initxcenter: xcenter (int)
- initycenter: ycenter (int)
- initzcenter: zcenter (int)

## OUTPUT

A folder will be create on the selected output path (dirname) and result of computation will be stored inside:

* dirname
	* images (.png)
	* analysissummary.txt
	* dataquality.txt
	* output.html
	* output.pdf


