### Canonical Correlation Analysis and Linear Discriminant Analysis

#### A simple implementation of CCA
##### Input description
- three images of the face of one person, but from different angles.
##### Algorithm description
```
    def analyze(self):
        self.zeroMean()
        self.build_cov()
        self.ED()
        self.reconstruct()
        self.show()
```
- zeroMean() does exactly what its name implies
- build_cov() constructs the two matrices we need to calculate the eigenvalues and eigenvectors.
```
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
```
```
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
```
- ED() does exactly what its name implies.
- reconstruct() tries to reconstruct the image from the eigenvalues and the eigenvectors acquired in the ED() step, however, the result is not so promising.
```
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
```

##### Questions
- How to reconstruct the image properly from the low dimensional data based on CCA?
- Is it necessary to flatten the input image to an one dimensional vector? In my previous PCA experiment, I didn't flatten the image, but the whole dimensionality-reduction process and image reconstruction process work pretty smoothly. However, this time, in CCA, the results are clearly far-fetched from the correct ones. So, I wonder is it because I didn't flatten my images?


#### Q1: What are advantages of CCA over PCA?
- CCA is able to handle data drawn from multiple subspaces,  which is something PCA couldn't do.

#### Q2：What are the limitations/requirement of CCA？
- the data must be drawn from multiple subspaces.
- CCA is a linear and global algorithm.
    - linear : CCA aims at finding a *linear* transformation for each subspace, such that the correlation between each transformed subspace is maximized.
    - global : the linear mapping is uniform everywhere, i.e, the same transformation is applied to all data points from the same view.

#### Q3：The objective function of CCA, and how to solve it?(binary and multiple case)
##### binary case
- **target** : Consider two sets of variables, $x\in R_1$, $y\in R_2$, CCA is formulated as:
$$arg\max_{w_x, w_y}w_x^TC_{xy}w_y$$
$$s.t.\ \ w_x^TC_{xx}w_x=1, \ \ w_y^TC_{yy}w_y=1 $$
- **solution** : 
    - incorporating the two constraints by lagrangian multipliers.
    - solving the generalized eigenvalue problem as followed.
    $$ \left[
        \begin{matrix} 
        0 & C_{xy} \\
        C_{yx} & 0 \\
        \end{matrix}
        \right] \left[
                \begin{matrix}
                w_x \\
                w_y \\
                \end{matrix}
                \right]=\lambda \left[
                                \begin{matrix}
                                C_{xx} & 0 \\
                                0 & C_{yy} \\
                                \end{matrix}
                                \right] \left[
                                        \begin{matrix}
                                        w_x \\
                                        w_y \\
                                        \end{matrix}
                                        \right] $$

##### multiple case
- **target** : Consider n multiple sets of variables, $x_1\in R_1$, $x_2\in R_2$, ..., $x_n\in R_n$, CCA is formulated as:
$$ arg\max_{w_1...w_n} \sum_{i=1}^n\sum_{j=1}^nw_i^TC_{ij}w_j$$
$$ s.t. \sum_{i=1}^nw_i^TC_{ii}w_i=1$$
- **solution** : 
    - incorporating the constraint by a lagrangian multiplier.
    - solving the generalized eigenvalue problem similar to the binary one.

#### Q5: What is the major difference between LDA and PCA, give two at least.
- LDA : supervised, utilizing the tags.
PCA : unsupervised, no tags
- LDA : applicable to multiple-subspaces situation
PCA : not applicable to multiple-subspaces situation

#### Q6 : What is the key idea of LDA and how does LDA utilize the label to perform DR?
- learning a projection matrix W so that the within-class data points are as close as possible and between-class data points are as far as possible.
- only with the help of labels could we define the within-class scatter and between-class scatter properly.
    - within-class scatter(within-class variance) : 
    $$ S_W=\sum_{k=1}^KS_k$$
    where
    $$ S_k=\sum_{n\in C_k}(x_n-m_k)(x_n-m_k)^T$$
    $$m_k=\frac{1}{N}\sum_{n\in C_k}x_n$$
    - between-class scatter(between-class covariance matrix) : 
    $$ S_B=\sum_{k=1}^KN_k(m_k-m)(m_k-m)^T$$
    where $N_k$ is the number of data points in class $C_k$

#### Q7 : the objective function of LDA, and how to solve it? (binary and multiple case)
##### binary case
- **objective function** : $$ J(w) = \frac{(m_2-m_1)^2}{s_1^2+s_2^2} = \frac{w^TS_Bw}{w^TS_Ww}$$
- **solution** : 
    - diffrentiating $J(w)$ with respect to $w$
    - we find that $J(w)$ is maximized when
    $$ (w^TS_Bw)S_Ww=(w^TS_Ww)S_Bw$$
    - based on the definition of $S_B$ and the fact that we only need the direction of $w$, therefore, we can forget about the scalar factors $w^TS_Bw$ and $w^TS_Ww$, multiplying both sides of the last formula by $S_W^{-1}$:
    $$ w\propto S_W^{-1}(m_2-m_1)$$
##### multiple case
- **objective function** : 
    $$J(w) = Tr\{(WS_WW^T)^{-1}(WS_BW^T)\} $$
    where $Tr$ means the trace of the matrix (sum of the eigenvalues of the matrix)
- **solution** : maximization of the objective function is quite straightforward and is discussed at length in *Fukunaga (1990)*.

#### Q8 : Why could LDA reduce the data into a K-1 dimensional space at most?
- $S_B$ is composed of the sum of K matrices, each of which is an outer product of two vectors. At the same time, only $(K-1)$ matrices are independent due to the definition of $m_k$(sorry, failed to understand this part). Thus, $S_B$ has rank at most equal to $(K-1)$ and so there are at most $(K-1)$ nonzero eigenvalues. Therefore, we are unable to find more than $(K-1)$ linear features by this means.

