##
##

from ctdg.data import JODIEDataset
from ctdg.utils import LazyCall as L

dataset = L(JODIEDataset)(
    path="data/wikipedia",
    name="Wikipedia",
    val_ratio=0.15,
    test_ratio=0.15,
    inductive_ratio=0.1,
    seed=2020,
)
