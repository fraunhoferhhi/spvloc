# https://github.com/xuxy09/RSC-Net/blob/master/doc/pyrender_woscreen_install.md
# copy directly into conda environment
wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
mkdir -p mesa-package
dpkg-deb -x ./mesa_18.3.3-0.deb mesa-package
cp -r mesa-package/usr/local/* $CONDA_PREFIX
rm -r mesa-package
rm mesa_18.3.3-0.deb