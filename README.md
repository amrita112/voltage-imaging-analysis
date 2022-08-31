# voltage-imaging-analysis
Pre-processing, semi-automated segmentation, signal extraction (volpy), and data visualization for voltage imaging datasets.

To run the code in this repository, use the 'Demo notebook to run Volpy'. 
The steps are as follows:
1. Pre-processing: Convert pixel intensity to photon counts, remove any part of the data without illumination
2. Motion correction and cell segmentation: Register raw data using caiman.motion_correction. Draw initial neuron ROIs using mean projections of registered movies.
3. Run Volpy: Extract dF/F, spike times, spatial filter and SNR for each neuron and display the data.

References
1. Spike Pursuit: Abdelfattah et al., “Bright and Photostable Chemigenetic Indicators for Extended in Vivo Voltage Imaging.” DOI: 10.1126/science.aav6416
2. Volpy: Cai C et al., "VolPy: Automated and scalable analysis pipelines for voltage imaging datasets." DOI: https://doi.org/10.1371/journal.pcbi.1008806
3. Caiman: Andrea Giovannucci et al., "CaImAn an open source tool for scalable calcium imaging data analysis." DOI: https://doi.org/10.7554/eLife.38173
