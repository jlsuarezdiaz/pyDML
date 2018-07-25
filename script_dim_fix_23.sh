#!/bin/bash

#Name of the job
#SBATCH -J dim23

#Queue
#SBATCH -p muylarga

#Output stream
#SBATCH -o large_output_dim23.info

#Error stream
#SBATCH -e large_error_dim23.info

#Current working directory


#pip install --user --upgrade Cython
#pip install --user --upgrade pandas
#pip install --user --upgrade arff
#pip install --user --upgrade scikit-learn

python3.6 -m pip install --user --upgrade numpy
python3.6 -m pip install --user --upgrade pandas
python3.6 -m pip install --user --upgrade Cython
python3.6 -m pip install --user --upgrade arff
python3.6 -m pip install --user --upgrade scikit-learn
python3.6 -m pip install --user --upgrade matplotlib
python3.6 -m pip install --user --upgrade seaborn

python3.6 --version
python3.6 setup.py build_ext --inplace
export PYTHONPATH=`pwd`
python3.6 test/test_tfg_dim_fix.py 2 3