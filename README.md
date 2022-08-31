# voltage-imaging-analysis
Pre-processing, semi-automated segmentation, signal extraction (volpy), and data visualization for voltage imaging datasets.

To run the code in this repository, use the 'Demo notebook to run Volpy'. 
The steps are as follows:
1. Pre-processing: Convert pixel intensity to photon counts, remove any part of the data without illumination
2. Motion correction and cell segmentation: Register raw data using caiman.motion_correction. Draw initial neuron ROIs using mean projections of registered movies.
3. Run Volpy: Extract dF/F, spike times, spatial filter and SNR for each neuron and display the data.


