
pip install -v -e .

cd ./mmsegextlib/dcnv3
python setup.py build install
cd ../../

cd ./mmsegextlib/msda
python setup.py build install
cd ../../

cd ./mmsegextlib/swattention
python setup.py build install
cd ../../

