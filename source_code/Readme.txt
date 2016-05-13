Dependencies
============
- opencv-python
- python-collections
- python-numpy
- ale_python_interface win64
- Theano
- lasagne

Once the dependencies are installed. Run: python ale_ql.py > debug_output
Models will be saved in the model folder and watch out for the episode scores and mean loss in debug_output

Notes:
- System that we used is: Windows 10 64 bit
- It takes a few days to train on cpu and about a day or two on a gpu.
- Mean Global error will stay nan for first 25000 frames (because model collects some data before starting training)