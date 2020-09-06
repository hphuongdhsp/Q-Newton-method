# A New Q-Newton's method avoiding saddle points

### This repository is accompanying with [paper](https://arxiv.org/pdf/2006.01512.pdf):
### A modification of quasi-Newton's methods helping to avoid saddle points, 
Truong Trung Tuyen, Tat Dat To, Tuan Hang Nguyen, Thu Hang Nguyen, Hoang Phuong Nguyen, Maged Helmy.



## Prerequisites 

```
scipy
algopy
numdifftools
```
## Doing experiments

To test the  Newton's method, BFGS, New Q-Newton's method, Random damping Newton's method and Inertial Newton's method 


```bash 

$ python src/main.py
```

To test more examples (with Backtracking GD, in addition to Newton's method and New Q-Newton's method) 

```bash 

$ python src/functionsDirectRun.py
```
