from pysofor.helpers import *


class IForest:
    ntrees: int
    phi: int
    trees: List[ITree]
    scale_factor: float

    def __init__(self, ntrees: int, phi: int):
        self.trees = []
        self.ntrees = ntrees
        self.phi = phi
        self.scale_factor = c(phi)

    def fit(self, x: np.ndarray):
        s = np.random.randint(x.shape[0], size=self.phi)
        for i in range(self.ntrees):
            self.trees.append(ITree(x[s, :], self.phi))

    def predict(self, x: np.ndarray):
        pls = np.array([itree.pathlength_multiple(x) for itree in self.trees])
        return 2 ** (-np.mean(pls, axis=0) / self.scale_factor)
