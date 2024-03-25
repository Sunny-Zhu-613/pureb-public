# Pureb-public

🚀 Exciting News! We've launched the `numqi` package [github/numqi](https://github.com/husisy/numqi), combining all the functionalities of this repository and even more! 🌟 To dive into these features, just install `numqi` using `pip install numqi`, and explore the relevant functions within the `numqi.entangle` module. 🛠️

## Quick start

Installation for the package
```bash
pip install .

# for developer
pip install -e .
```

Unittest
    
```bash
pytest -v
```

matlab setup for SDP

Installation

- `cvx` [cvxr.com/cvx](http://cvxr.com/cvx/)
- `cvxquad` [github/cvxquad](https://github.com/hfawzi/cvxquad)
- `quantinf` [dr-qubit/link](https://www.dr-qubit.org/matlab.html)

```bash
matlab -nodisplay

# matlab
cvx_setup # cvx
addpath("/path/to/cvxquad")
addpath("/path/to/quantinf")
addpath("/path/to/QETLAB-0.9")
```
## Workspace
```bash
cd ws_paper/
```
Main results in the paper

- `isotropic_ree.py` and `werner_ree.py` for computing REE of Isotropic and Werner states.
- `werner2_kext_boundary.py` and `kext_boundary_accuracy.py` for computing the boundaries across different k-ext along the direction of Werner states or random states.
- `upb_bes_boundary.py` for drawing a plane constructed by Tiles and pyramid UPB BES.
- `purebQ_werner.py` is the quantum circuit version of PureB-ext.
