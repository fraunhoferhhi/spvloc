git clone --branch v143 --recursive https://github.com/gard-n/redner.git
cd redner 
python setup.py install
python -m pip wheel -w dist --verbose .
cd ..
rm -rf redner