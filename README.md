# KMeansDimReduction

This project is part of the course "Sketching Methods for Matrix and Data Analysis" by [Haim Avron](http://www.math.tau.ac.il/~haimav/index.html)

The objective of the project is reproducing the results of the paper [Randomized Dimensionality Reduction
for k-Means Clustering](https://www.boutsidis.org/Boutsidis_IEEE_IT_15.pdf)

## How to use

For help you can use the ```-h``` command line argument

```bash
python3 main.py -h
```

For configuring data set use (in the following example the synthetic data set)

```bash
python3 main.py --data-set SYNTH
```

## Graph produced
The code runs a simulation for comparing different dimension reduction methods.
Example for  the graphs producing by the simulation over a specified data set (in this case [ORL](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) data set).
![alt text](https://imgur.com/XhWaMcK.png) ![alt text](https://imgur.com/Iq4uvE0.png) 
![alt text](https://imgur.com/hdB1awM.png) ![alt text](https://imgur.com/G8SLBm6.png)
