# GERMirAE

GERMirAE is a tool for predicting the occurrences of irAE (immune-related adverse events) using WES-derived germline variants in patients treated with immune checkpoint blockade

Prerequisites
This program requires the following modules.

Python (=> 2.7 ): https://www.python.org/
NumPy : http://www.numpy.org/
scikit-learn : http://scikit-learn.org/stable/index.html

Options

-m INT : OCSVM-Splicing model number (1,2,or 3)

-i STR : Input file name

-o STR : Output file name

example:  python one_calss_SVM.py -m 1 -i model_1_input_example.txt -o model_1_output.txt
