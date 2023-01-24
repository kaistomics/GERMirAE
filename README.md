# GERMirAE

GERMirAE is a tool for predicting the occurrences of irAE (immune-related adverse events) using WES-derived germline variants in patients treated with immune checkpoint blockade

Prerequisites

This program requires the following modules.

Python (=> 2.7 ): https://www.python.org/
NumPy : http://www.numpy.org/
scikit-learn : http://scikit-learn.org/stable/index.html

Options

type_ : string, type of irAE (sys.argv[1] # ANY, SKIN, ENDO etc.)

labmd : string, mode of labelling (sys.argv[2] # all, within, onlyCtr)

cores : integer, number of cores used for the calculation (sys.argv[3])

n_iter : integer, number of iterations for machine learning algorithms (sys.argv[4])

example :  python GERMirAE.py ANY onlyCtr 88 100

