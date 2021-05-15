from pathlib import Path
import os
import wget
import zipfile

'''
Download and extract the L3DAS21 dataset into a user-defined directory.
Command line arguments define which dataset partition to download and where to
save the unzipped folders.
'''

def download_l3das_dataset(set_type):
    if not os.path.exists(os.getcwd()+os.sep+'L3DAS_'+'Task2'+'_'+set_type+'.zip'):
        print ('Downloading')
        URL = 'https://zenodo.org/record/4642005/files/'
        zip_name= 'L3DAS_'+'Task2'+'_'+set_type+'.zip'
        wget.download(URL+zip_name)
        print("\n")
    else:
        print("Existing folder\n")


def extract_dataset(set_type, output_path):
    ZIP_PATH = os.getcwd() / Path("L3DAS_Task2_"+set_type+".zip")
    print("Extracting the archive")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    print("Done")
    os.remove(ZIP_PATH)


def download_and_extract(set_type, output_path):
    """ set_type: train or dev
        output_path: path to extract dataset files
    """

    download_l3das_dataset(set_type)
    extract_dataset(set_type, output_path)


