import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from .activations import swish


class DimeNetPP(tf.keras.Model):
    """
    DimeNet++ model.

    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    envelope_exponent
        Shape of the smooth cutoff
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    extensive
        Whether the output should be extensive (proportional to the number of atoms)
    output_init
        Initialization method for the output layer (last layer in output block)
    """

    def __init__(
            self, emb_size, out_emb_size, int_emb_size, basis_emb_size,
            num_blocks, num_spherical, num_radial,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_targets=12,
            activation=swish, extensive=True, output_init='zeros',
            name='dimenet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Embedding and first output block
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(
            OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets,
                          activation=activation, output_init=output_init))

        # Interaction and remaining output blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip,
                                   num_after_skip, activation=activation))
            self.output_blocks.append(
                OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets,
                              activation=activation, output_init=output_init))

    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        # ReLU prevents negative numbers in sqrt
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    def call(self, inputs):
        Z, R                         = inputs['Z'], inputs['R']
        batch_seg                    = inputs['batch_seg']
        idnb_i, idnb_j               = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji   = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        n_atoms = tf.shape(Z)[0]

        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(
            R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Embedding block
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms])

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i+1]([x, rbf, idnb_i, n_atoms])

        if self.extensive:
            P = tf.math.segment_sum(P, batch_seg)
        else:
            P = tf.math.segment_mean(P, batch_seg)
        return P
