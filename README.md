# connectomes_harmonization
Connectomes harmonization
version 1.0.0

================================================================================

This software requires the Python librairies:

	neuroCombat pandas scipy numpy os matplotlib

neuroCombat can be installed by executing:

	pip install neurocombat

================================================================================

harmonization.py requires the following parameters:

_ a csv file that (1) lists text or npy files storing fMRI time series, (2) contains associated clinical variables to preserve during the harmonization such as age, (3) indicates what group the scan/time series belong to

_ the framework to use for the harmonization (no framework, Fisher z-transform, Affine-Invariant geometric framework, Bures-Wasserstein geometric framework)

_ the method to use (setting group means to the same values, harmonizing group means and dispersion, ComBat )

_ the reference connectome to use (the identity marix, the average of the correlation matrices, or the Frechet mean. In the Fisher z-transform framework, the Frechet mean is calculated as the mean of the connectomes after Fisher z-transform. In the "no framework" framework the Frechet mean is the standard mean between correlation matrices).

_ an output folder to store the results
 
and carry out the following steps:

_ calculation of the correlation matrices (including OAS shrinkage)

_ harmonization of these matrices

_ storing the harmonized matrices in the output folder

_ storing the flattened and concatenated upper triangular part of the log of the correlation matrices before and after harmonization


To check the usage: 

	python harmonization.py -h


================================================================================

To run a full processing on synthetic data (connectomes of dimension 25x25):

	python syntheticHarmonization.py -d 25 
	
This code (syntheticHarmonization.py) takes as parameter the dimension of the connectomes (smaller connectomes will be processed faster) and:

_ generates 10 random data sets (of 475 time series, each time series containing 1000 time points)

_ for each data set, runs three different harmonizations (mean harmonization, mean and scatter harmonization, ComBat harmonization) with the fastest method: the Bures-Wasserstein framework using the identity matrix as a reference

_ displays the harmonization.py commands used

_ generates a visualization of the log of the connectomes before and after harmonization (PCA)  

To check the usage: 
	python syntheticHarmonization.py -h

================================================================================

For more details about the method, please read (and cite): 
	Riemannian Frameworks for the Harmonization of resting-state Functional MRI Scans
	Nicolas Honnorat, Sudha Seshadri, Ron Killiany, John Blangero, David C. Glahn, Peter Fox, Mohamad Habes
	Medical Image Analysis (in revision)
