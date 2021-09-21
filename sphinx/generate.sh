pip install ../
sphinx-apidoc -o source/ ../deepblocks/ -Mef -d 1
make clean
make html

rm -rf ../docs
cp -r build/html ../docs