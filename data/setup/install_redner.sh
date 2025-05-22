git clone --recursive https://github.com/BachiLi/redner.git
cd redner 
python setup.py install
python -m pip wheel -w dist --verbose .
cd ..
rm -rf redner