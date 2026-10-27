# msr
This repository contains the software related to the Multi-strategy Serial Refine (MSR) algorithm for wind farm flow control optimization. This optimization algorithm allows an user-defined black-box objective function and it is demonstrated with the application of wake steering and helix wake mixing as control techniques.

# Related publications
The theory underlying these metrics and their applications are described in the following publications:
- \[1\] M. Baricchio, D. van der Hoek, T. Dammann, P.M.O. Gebraad, J. Iori and J.W. van Wingerden. ‘Multi-strategy wind farm control: alternating wake steering and helix wake mixing on a large-scale wind farm’. In: \textit{Wind Energy Science} 11.8 (2026), pp. 2817–2843. DOI: [10.5194/wes-11-2817-2026](https://doi.org/10.5194/wes-11-2817-2026)
- \[2\] M. Baricchio, D. van der Hoek, T. Dammann, J. Iori, P.M.O. Gebraad and J.W. van Wingerden. ‘An approximation of the optimal combined helix and yaw control for wind farm co-design applications’. In: *Journal of Physics: Conference Series* 3224.3 (2026), p. 032019. DOI: [10.1088/1742-6596/3224/3/032019](https://doi.org/10.1088/1742-6596/3224/3/032019)

# How to use
The code related to the optimization algorithm can be found in the file **msr.py**, and a basic example how to use these functions is made available in the jupyter notebook **tutorial_msr.ipynb** (recommended to get started). 
The folder **case/HKNscaled** contains the software and data related to \[1\].

# Dependecies
The optimization algoirhtm contained in **msr.py** only requires the python package numpy (>= 2.2.6 recommended).
To run all other scripts the following packages are required:
- scipy (>= 1.15.3 recommended)
- pathos (>= 0.3.4 recommended)
- pandas (>= 2.2.3 recommended)
- py_wake (>= 2.6.11 recommended)
- matplotlib (>= 3.10.3 recommended)
- xarray (>= 2025.4.0 recommended)
- utm  (>= 0.8.1 recommended)

To run the code included in the folder **case/HKNscaled** several additional packages are required. All the packages (with their versions) present in the environment adopted for this example are listed in the corresponding folder.


Technische Universiteit Delft hereby disclaims all copyright interest in the
program Layout Field supported by the SUDOCO project, which receives the funding
from the European Union’s Horizon Europe Programme under the grant No. 101122256.
written by the Author(s).
Prof. Dr. Ir. Fred van Keulen, Dean of Mechanical Engineering