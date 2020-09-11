# TileQC
Tile Quality Control for piTEAM

The repository provides the python script for evaluating the EM montage quality, example QC output images, example dataset for running the QC script and example high-res images collected from production dataset.

![](demo/demo.gif)

The images available via this repository are subject to the Allen Institute Terms of Use, which are currently available at
https://alleninstitute.org/legal/terms-use/

“tile_qc_meta.py” is a python script running on metadata file from every montage and performing the QC evaluation including tile overlapping, image quality, focus etc. This script has been integrated into imaging pipeline as a key function to check the montage quality after each section being imaged. It can also run as a stand-alone script. Here we provide a step-by-step guide and some QC_Example_Dataset to run the script.

  1.	Install python3.6.
  2.	Install the following packages: numpy-1.15.4+mkl-cp36-cp36m-win_amd64.whl; opencv_python-3.4.5-cp36-cp36m-win_amd64.whl; requirements.txt
  3.	To run the tile_qc_meta.py script, go to the directory where the script is saved, then run: python tile_qc_meta_demo.py “directory of sample dataset”
  4.	Expected QC output files are saved in the same folder as the dataset.


