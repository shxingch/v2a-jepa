sudo apt-get install xvfb
cd /root/jepa/libero
xvfb-run -a "-screen 0 1400x900x24" python example/example_evaluate_libero.py