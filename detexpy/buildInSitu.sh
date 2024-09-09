flex --nodefault --noline -o detexpysource.c detexpy.l
python3 setup.py build
cp build/lib.*/detexpy.* ..
rm -R build
