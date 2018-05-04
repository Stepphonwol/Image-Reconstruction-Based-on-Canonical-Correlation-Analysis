import numpy as np
from PIL import Image

class CCA:
    def __init__(self, subspace_num):
        self.subspace_num = subspace_num
        self.src_list = []
        for i in range(self.subspace_num):
            im_name = input("Type in the name of the image: ")
            im = Image.open(im_name)
            src = np.array(im.convert('L'))
            #src = np.transpose(src.flatten())
            print(src.shape)
            self.src_list.append(src)

    def zeroMean(self):
        self.phi_list = []
        self.mean_list = []
        print("Calculating the mean value for each variable...")
        for src in self.src_list:
            mean_val = np.mean(src, axis=0)
            self.mean_list.append(mean_val)
            print(mean_val)
            phi_data = src - mean_val
            print(phi_data)
            self.phi_list.append(phi_data)

    def build_cov(self):
        left_cov_list = [] # save the covariance matrix of Cnn
        left_dim = 0
        print("Calculating the left matrix...")
        for src in self.phi_list: # calculate the covariance matrix of each
            cov_matrix = np.cov(src, rowvar=0)
            print(cov_matrix.shape)
            left_cov_list.append(cov_matrix)
            left_dim = left_dim + cov_matrix.shape[0]
        self.left_matrix = np.zeros((left_dim, left_dim))
        base = 0
        # calculate the left matrix
        for cov in left_cov_list:
            self.left_matrix[base:base+cov.shape[0],base:base+cov.shape[1]] = cov
            base = base + cov.shape[0]
        print(self.left_matrix.shape[0])
        print(self.left_matrix.shape[1])
        print(self.left_matrix)
        #----------------calculate the right matrix
        print("Calculating the right matrix...")
        right_dim = 0
        # calculate the dimension of the right matrix
        calc_base = self.phi_list[0]
        for calc in self.phi_list:
            #cov_matrix = np.cov(calc_base, calc, rowvar=0)
            print(calc.shape)
            cov_matrix = np.dot(calc_base.T, calc) / (calc.shape[1] - 1)
            right_dim = right_dim + cov_matrix.shape[0]
        print(right_dim)
        # calculate the right matrix
        self.right_matrix = np.zeros((right_dim, right_dim))
        x_base = 0
        y_base = 0
        for row_src in self.phi_list:
            column_increment = 0
            for column_src in self.phi_list:
                #cov_matrix = np.cov(row_src, column_src, rowvar=0)
                cov_matrix = np.dot(row_src.T, column_src) / (column_src.shape[1] - 1)
                print(cov_matrix.shape)
                print(cov_matrix)
                self.right_matrix[x_base:x_base+cov_matrix.shape[0], y_base:y_base+cov_matrix.shape[1]] = cov_matrix
                x_base = x_base + cov_matrix.shape[0]
                column_increment = cov_matrix.shape[1]
            x_base = 0
            y_base = y_base + column_increment
        print(self.right_matrix.shape[0])
        print(self.right_matrix.shape[1])
        print(self.right_matrix)

    def ED(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(np.dot(np.linalg.pinv(self.left_matrix), self.right_matrix))

    def reconstruct(self):
        #self.eigenvalues = np.ones((1, 12096)).flatten()
        #self.eigenvectors = np.ones((12096, 12096))
        print(self.eigenvalues)
        print(self.eigenvectors.shape)
        print("%d bases" % np.size(self.eigenvectors))
        new_n = int(input("The number of new bases: "))
        eig_seq = np.argsort(self.eigenvalues)
        eig_seq_indice = eig_seq[-1:-(new_n + 1):-1]
        new_eig_vec = self.eigenvectors[:, eig_seq_indice]
        print(new_eig_vec.shape)
        lower_data_list = []
        lower_base = 0
        for index, data in enumerate(self.phi_list):
            lower_data = np.dot(data, new_eig_vec[lower_base:lower_base + data.shape[1], :])
            print(lower_data.shape)
            lower_data_list.append(lower_data)
            lower_base = lower_base + data.shape[1]
        lower_base = 0
        self.recon_list = []
        for index, data in enumerate(lower_data_list):
            recon = np.dot(data, new_eig_vec[lower_base:lower_base + self.phi_list[index].shape[1], :].T) + self.mean_list[index]
            lower_base = lower_base + self.phi_list[index].shape[1]
            self.recon_list.append(recon)

    def show(self):
        for data in self.recon_list:
            show_data = data.astype('uint8')
            new_im = Image.fromarray(show_data, mode="L")
            new_im.show()

    def analyze(self):
        self.zeroMean()
        self.build_cov()
        self.ED()
        self.reconstruct()
        self.show()

if __name__ == "__main__":
    n = int(input("Number of views: "))
    cca = CCA(n)
    cca.analyze()