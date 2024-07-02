# Universal OCO Simulations
 A base implementation of "A simple yet Universal Strategy for Online Convex Optimization" (Zhang et. al. 2022) with some simple examples.

Problems are specified with a domain in the form $\\{x:f(x)<= \ell\\}$, for a convex function $f$ and vector of levels $\ell$. In the [domain visualization notebook](Notebooks/domain_visualization.ipynb), some examples of sets are provided and features like projection are shown. 

In the [gradient descent notebook](Notebooks/gradient_descent.ipynb), a simple online convex optimization (OCO) problem is set up where the function is given as $f_t(x) =f(t,x)$ for convience. A comparison of projected subgradient descent (PSGD) with default steps and steps under the strong convexity assumption are shown, although the difference is minimal considering the simplicity of the example. 

In the [universal notebook](Notebooks/universal_alg_simple.ipynb), a simple example of the universal algorithm from the [Zhang paper](zhang2022.pdf) (Algorithm 1) is created over a box and compared against the best possible strongly convex expert in hindsight (close to the guarantee of best discretized expert in the paper).  


# Usage 

To adapt the code to your problem, add your optimization algorithms to mimick the PSGD class in [Utils](Notebooks/Utils.py) and if necessary, replicate the strongly-convex expert section to $\exp$-concave experts in the **__init__** of the **UniversalAlgorithm** class in the same [Utils](Notebooks/Utils.py) file. 
