git clone --recursive https://github.com/gard-n/pyrender.git
cd pyrender
python setup.py install
python -m pip wheel -w dist --verbose .
cd ..
rm -rf pyrender
