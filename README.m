# Dynamic states of parcellations
This repository contains scripts, and notebooks for the following article **Unraveling reproducible dynamic states of individual brain functional parcellation**.

Click on this link to show an example of generating dynamic states of parcellations:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SIMEXP/dynamic-states-parcellations/980d28d9af72500c3beb7adecf609c27fe3522b0)

Here is a brief description of each item in the repository:

**dynpar**
* **prepare_fmri_img.py** - Python script used to extract functional MRI signal from functional scans.
* **static_parcellation.py** - Python script used to replicate static parcellations. 
* **dynamic_parcellation.py** - Python script that inherits from the **static_parcellation.py** and it includes further functions to generate state stability maps. 

**Notebooks**

* **Generate_states_stability_maps_dynpar.ipynb** - Jupyter notebook that generates state stability maps for one individual from the test-retest NYC dataset. All the data is automatically downloaded by the Nilearn grabber.




