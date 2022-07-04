import numpy as np
from danmf_crfr import DANMF_CRFR
from tqdm import tqdm
from utils import preproccessing


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(lambda0, lambda1, lambda2):

    layer = [120, 100, 80]
    dataset = "mnist"
    lambda0 = lambda0 # DI SIMILARITY
    lambda1 = lambda1 # SIMILARITY
    lambda2 = lambda2 # FRP

    args = Namespace(
        k_neigh=5,
        k_kmeans=10,
        delta=0.3,
        dataset=dataset,
        calculate_loss=False,
        dataset_path="datasets/DB_normalized/"+dataset+".mat",
        iterations=500,
        pre_iterations=500,
        lamb0=lambda0,
        lamb1=lambda1,
        lamb2=lambda2,
        layers=layer,
        seed=None)

    data = preproccessing(args)
    model = DANMF_CRFR(data, args)
    model.pre_training()
    return model.training()


# range for obtaining values as grid search strategy
lambda_0_arr = [0, 0.000001, 0.00001, 0.0001, 0.001]  # dis
lambda_1_arr = [0, 0.01, 0.1, 1, 10, 100, 1000]  # sim
lambda_2_arr = [0, 0.5, 0.75, 1, 1.25, 1.5, 2]  # FRP

nmi_arr, ari_arr, acc_arr = [], [], []

if __name__ == "__main__":
    # run the code n times for achieving standard deviation and average measurements
    for i in tqdm(range(3)):
        NMI, ARI, ACC = main(0.000001, 5, 0)
        nmi_arr.append(NMI)
        ari_arr.append(ARI)
        acc_arr.append(ACC)

print(f"Average NMI: {np.average(np.array(nmi_arr))}, ARI: {np.average(np.array(ari_arr))}, ACC: {np.average(np.array(acc_arr))}")
print(f"STD NMI: {np.std(np.array(nmi_arr))}, ARI: {np.std(np.array(ari_arr))}, ACC: {np.std(np.array(acc_arr))}")