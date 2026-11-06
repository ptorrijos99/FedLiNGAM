# Data

- `BNs/*.xbif`: the eight network topologies of the extended article (Asia, Sachs, Child,
  Insurance, Water, Mildew, Alarm, Barley), from the bnlearn Bayesian network repository
  (https://www.bnlearn.com/bnrepository/). Only the graphs are used; weights and noise are
  drawn by `experiments/benchmark/data.py`.
- `sachs_data_cdt_7466_11.csv`: the Sachs et al. (2005) flow cytometry assay as distributed by
  the Causal Discovery Toolbox (Kalainathan et al., JMLR 2020). The 7466 rows are the first nine
  reagent conditions in order; `experiments/benchmark/data.py` splits them into one client per
  condition.
- `danube/danube_data_raw.csv` and `danube/danube_edges.csv`: daily discharge at 31 gauging
  stations of the upper Danube and the river network (Asadi et al., 2015), as distributed by the
  R package graphicalExtremes (Engelke et al., CRAN, GPL-3). Edges point downstream.
