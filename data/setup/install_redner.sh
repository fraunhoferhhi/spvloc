git clone --branch v143 --recursive https://github.com/gard-n/redner.git
# CUDA is not needed for pyredner and also did not lead to performance improvements
export REDNER_CUDA=0 
cd redner 
python setup.py install
python -m pip wheel -w dist --verbose .
cd ..
rm -rf redner