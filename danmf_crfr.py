from tqdm import tqdm
from sklearn.decomposition import NMF as NMF_SKL
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from acc import cluster_acc
import numpy as np
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.cluster import KMeans




class DANMF_CRFR():

    def __init__(self, data, args):
        self.X = data['X']
        self.Y = data['Y'][0]
        self.XXT = self.X @ self.X.T
        self.D = data['D']
        self.S = data['S']
        self.L = data['L']
        self.args = args
        self.pre_iter = args.pre_iterations
        self.lamb0 = args.lamb0
        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2
        self.U_s = []
        self.V_s = []
        self.Z = []
        self.p = len(args.layers)
        self.Dis = pairwise_distances(self.X.T, self.X.T)
        self.k_kmeans = args.k_kmeans

    # Target matrix
    def setup_z(self, i):
        if i == 0:
            self.Z = self.X
        else:
            self.Z = self.V_s[i-1]

    # Shallow Autoencoder-like NMF for pre-training
    def auto_encoder_pretrain(self, i):
        U = np.random.rand(self.Z.shape[0], self.args.layers[i])
        V = np.random.rand(self.args.layers[i], self.Z.shape[1])
        for i in range(self.args.pre_iterations):
            u_u = 2 * self.Z @ V.T
            u_d = U @ V @ V.T + self.Z @ self.Z.T @ U
            U = U * (u_u / np.maximum(u_d, 10 ** -10))

            v_u = 2 * U.T @ self.Z
            v_d = U.T @ U @ V + V
            V = V * (v_u / np.maximum(v_d, 10 ** -10))
        return U, V

    # Pre-training process
    def pre_training(self):
        print("\nLayer pre-training started. \n")
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            U, V = self.auto_encoder_pretrain(i)
            self.U_s.append(U)
            self.V_s.append(V)

    # update Basis Matrix (U or W) according to Paper
    def update_U(self, i):

        if i == 0:
            # the next line is just for simplifying U in following lines
            U = self.U_s[i] @ self.PHI[i + 1]
            bottom = self.U_s[i] @ self.PHI[i+1] @ self.Vp @ self.Vp.T @ self.PHI[i+1].T
            bottom = bottom + self.X @ self.X.T @ self.U_s[i] @ self.PHI[i+1] @ self.PHI[i+1].T + 2 * self.lamb2 ** 2 * U @ U.T @ U @ self.PHI[i+1].T
            top = 2 * self.X @ self.Vp.T @ self.PHI[i+1].T + 2 * self.lamb2 * self.XXT @ U @ self.PHI[i+1].T
            self.U_s[0] = self.U_s[0] * (top / np.maximum(bottom, 10 ** -10))
        else:
            # the next line is just for simplifying U in following lines
            U = self.SII @ self.U_s[i] @ self.PHI[i + 1]
            bottom = self.SII.T @ self.SII @ self.U_s[i] @ self.PHI[i + 1] @ self.Vp @ self.Vp.T @ self.PHI[i + 1].T
            bottom = bottom + self.SII.T @ self.X @ self.X.T @ self.SII @ self.U_s[i] @ self.PHI[i + 1] @ self.PHI[i + 1].T \
                     + 2 * self.lamb2**2 * self.SII.T @ U @ U.T @ U @ self.PHI[i+1].T
            top = 2 * self.SII.T @ self.X @ self.Vp.T @ self.PHI[i + 1].T + 2 * self.lamb2 * self.SII.T @ self.XXT @ U @ self.PHI[i+1].T
            self.U_s[i] = self.U_s[i] * (top / np.maximum(bottom, 10 ** -10))
        return

    # Update coefficient Matrix (H or V) according with paper
    def update_V(self, i):
        if i == self.p - 1:
            top = 2 * self.SII.T @ self.X                      + 2 * self.lamb1 * self.V_s[-1] @ self.S
            bottom = self.SII.T @ self.SII @ self.V_s[-1] + self.V_s[-1] + 2 * self.lamb1 * self.V_s[-1] @ self.D + self.lamb0 * self.V_s[-1] @ self.Dis
            self.V_s[i] = np.multiply(self.V_s[i], top) / np.maximum(bottom, 10 ** -10)
        return

    # Set PHI ( Φ ) Matrix based on Ui+1 to Up
    def set_PHI(self):
        self.PHI = [None for _ in range(self.p + 1)]
        self.PHI[self.p] = np.eye(self.args.layers[self.p - 1])
        for i in range(self.p - 1, -1, -1):  # Question
            self.PHI[i] = np.dot(self.U_s[i], self.PHI[i + 1])

    # Set PSI ( Ψ ) Matrix based on U1 to Ui-1
    def set_SII(self, i):
        if i == 0:
            self.SII = self.U_s[0]
        else:
            self.SII = self.SII.dot(self.U_s[i])


    # Main training process after pre-training
    def training(self):

        print("\n\nTraining started. \n")
        self.loss = np.zeros(self.args.iterations)
        for iteration in tqdm(range(self.args.iterations), desc="Training pass: ", leave=True):

            self.Vp = self.V_s[self.p-1]
            for i in range(self.p):
                self.set_PHI()
                self.update_U(i)
                self.set_PHI()
                self.set_SII(i)
                self.update_V(i)
            if  self.args.calculate_loss:
                self.calculate_cost(iteration)

        # normalizing Vp
        sum = np.sum(self.V_s[-1], axis=0)
        normalized_Vp = self.V_s[-1] / sum

        kmeans = KMeans(self.k_kmeans, random_state=20).fit(normalized_Vp.T)
        print(normalized_mutual_info_score(self.Y, kmeans.labels_), adjusted_rand_score(self.Y, kmeans.labels_), cluster_acc(self.Y, kmeans.labels_))
        # return NMI, ARI, ACC
        return normalized_mutual_info_score(self.Y, kmeans.labels_), adjusted_rand_score(self.Y, kmeans.labels_), cluster_acc(self.Y, kmeans.labels_)


