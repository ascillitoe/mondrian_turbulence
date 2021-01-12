## Data-driven turbulence modelling with random and Mondrian forests

This python package trains random forests and Mondrian forests on high fidelity LES/DNS data. The trained models can then be used to predict turbulence parameters for a new RANS flowfield. For more details see:

Ashley Scillitoe, Pranay Seshadri, Mark Girolami,
*Uncertainty quantification for data-driven turbulence modelling with mondrian forests*,
Journal of Computational Physics,
2021,
110116,
ISSN 0021-9991,
doi: [10.1016/j.jcp.2021.110116](https://doi.org/10.1016/j.jcp.2021.110116). 
arXiv: [2003.01968](http://arxiv.org/abs/2003.01968).

### How to use
Instructions and examples coming soon!

### Notes
* Regressors and classifiers are implemented, however the classifer code is out of date and should be used with caution!
* requirements.txt file to enable easy installation is in the works.

### Key dependencies
* `scikit-learn`: For Random forest classifier and regressor - https://scikit-learn.org/stable/
* `scikit-garden`: For Mondrian forest regressor - https://scikit-garden.github.io
* `pyvista`: For reading and writing vtk files, and built into the `CaseData` class - https://github.com/pyvista/pyvista
* `forestci`: For calculating infinitesimal jackknife uncertainty estimates for random forests - https://github.com/scikit-learn-contrib/forest-confidence-interval
* `shap`: For calculating SHAP values - https://github.com/ascillitoe/shap (forked from https://github.com/slundberg/shap)
* `eli5`: For calculating permutation importance - https://github.com/TeamHG-Memex/eli5

### Acknowledgments
This work was supported by wave 1 of The UKRI Strategic Priorities Fund under the EPSRC grant EP/T001569/1, particularly the [Digital Twins in Aeronautics](https://www.turing.ac.uk/research/research-projects/digital-twins-aeronautics) theme within that grant, and [The Alan Turing Institute](https://www.turing.ac.uk).
