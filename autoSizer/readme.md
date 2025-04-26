LabelingGui.py runs the labeling GUI as a standalone, for labeling xy points of oyster hinges.
labelingGuiWithMaskClean is still under work. Intention is to allow for basic mask cleaning features such as removing spurious objects and small object removal.
However, is not done yet. requires some work, specifically the saving routine and workflow - how to save and merge the edited binary image, since will affect how next step is called.  
SAM_Autocrop.py takes original SAM script for segmentation and adds auto stage cropping. 
condaRequirements.yml gives minimum conda environment for running.  Includes torchvision-cuda for acceleration
