pip install ../
sphinx-apidoc -o source/ ../deepblocks/ -Mef -d 1
make clean
make html

cd build/html 
cp * -rf ../../../docs