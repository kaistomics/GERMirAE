# GERMirAE

GERMirAE is a tool for predicting the occurrences of irAE (immune-related adverse events) using WES-derived germline variants in patients treated with immune checkpoint blockade

## Prerequisites

This program requires the following modules.

Python (=> 2.7 ): https://www.python.org/

NumPy : http://www.numpy.org/

scikit-learn : http://scikit-learn.org/stable/index.html


## Options

type_ : string, type of irAE (sys.argv[1] # ANY, SKIN, ENDO etc.)

labmd : string, mode of labelling (sys.argv[2] # all, within, onlyCtr)

cores : integer, number of cores used for the calculation (sys.argv[3])

n_iter : integer, number of iterations for machine learning algorithms (sys.argv[4])
```
example :  python GERMirAE.py ANY onlyCtr 88 100
```

## Outputs
Binary classification and the probability of occurrences for the irAE of interest according to the samples

<img width="279" alt="image" src="https://user-images.githubusercontent.com/35682945/214225841-50b8b5f4-1981-4d96-a001-2df973af4f32.png">
