#!/usr/bin/env python

import os
import logging
import pydicom
import nibabel as nib


def embed_studyinfo(niftifile, studyinfo_dict):
    """embed a studyinfo dict into a nifti image header extension comment (type 6)"""

    NEinfo = nib.nifti1.Nifti1Extension('comment', str(studyinfo_dict))
    image = nib.load(niftifile)
    image.header.extensions.append(NEinfo)
    nib.save(image, niftifile)


def extract_studyinfo(niftifile):
    """extract a studyinfo dict from a nifti image header extension comment"""

    info = {'Coil': '', 'StudyDate': '', 'StudyTime': '', 'ElementName': ''}
    image = nib.load(niftifile)
    for extension in image.header.extensions:
        content = extension.get_content()
        if 'StabilityCalcStudyinfo' in content:
            info = eval(content)
    return info

def getsiemensmrheader(dicomfilename):
        theplan = dicom.read_file(dicomfilename)
        siemens_csa_header2 = theplan[0x0029, 0x1020].value
        print (siemens_csa_header2)
        startposition = siemens_csa_header2.find('### ASCCONV BEGIN ###') + len('### ASCCONV BEGIN ###')
        partialheader = siemens_csa_header2[startposition:]
        endposition = partialheader.find('### ASCCONV END ###')
        splitheader = partialheader[:endposition].splitlines()
        d = {}
        for line in splitheader[1:]:
            pair = line.split()
            d[pair[0]] = pair[2]    
        #siemensheader = getsiemensmrheader(plan)
        #print (plan)
        #tr=plan.RepetitionTime/1000
        #print tr
        return d

def studyinfo_from_dicom(dicomfilename):
    """generate a studyinfo dict from a dicom file"""
    plan = pydicom.read_file(dicomfilename)

    info = {#'Coil': siemensheader['asCoilSelectMeas[0].asList[0].sCoilElementID.tCoilID'].strip('"'),
            'Coil': ' ',
            'StudyDate': plan.StudyDate,
            'StudyTime': plan.StudyTime,   
            #'TR' : plan[0x0018, 0x0080].value}
            'StabilityCalcStudyinfo': '1'}
    try:
        info['ElementName'] = plan[0x0051, 0x100f].value
    except KeyError:
        info['ElementName'] = 'UNKNOWN'

    try:
        info['RepetitionTime'] = plan[0x0018, 0x0080].value
    except KeyError:
        info['RepetitionTime'] = 'UNKNOWN'

    try:
        info['IRCCS'] = plan.InstitutionName
    except AttributeError:
        info['IRCCS'] = 'UNKNOWN'

    try:
        info['Manufacturer'] = plan.Manufacturer
    except AttributeError:
        info['Manufacturer'] = 'UNKNOWN'

    try:
        info['EchoTime'] = plan.EchoTime
    except AttributeError:
        info['EchoTime'] = 'UNKNOWN'

    try:
        info['PixelBandwidth'] = plan.PixelBandwidth
    except AttributeError:
        info['PixelBandwidth'] = 'UNKNOWN'

    return info


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Pull header data from a DICOM file.')
    parser.add_argument('dicomfile', help='The dicom to examine.')
    args = parser.parse_args()

    if not os.path.exists(args.dicomfile):
        logging.critical('studyinfo: dicomfile {} not found'.format(args.dicomfile))
        exit(1)

    for k, v in studyinfo_from_dicom(args.dicomfile).items():
        print('{}: {}'.format(k, v))
