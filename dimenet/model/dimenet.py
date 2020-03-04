import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .activations import swish


class DimeNet(tf.keras.Model):
    """
    DimeNet model.

    Parameters
    ----------
    emb_size
        Embedding size used throughout the model
    num_blocks
        Number of building blocks to be stacked
    num_bilinear
        Third dimension of the bilinear layer tensor
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
    seed
        Random seed for weight initialization
    """

    def __init__(
            self, emb_size, num_blocks, num_bilinear, num_spherical,
            num_radial, cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_targets=12,
            activation=swish, name='dimenet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Embedding and first output block
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(
            OutputBlock(emb_size, num_dense_output, num_targets, activation=activation))

        # Interaction and remaining output blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                InteractionBlock(emb_size, num_bilinear, num_before_skip,
                                 num_after_skip, activation=activation))
            self.output_blocks.append(
                OutputBlock(emb_size, num_dense_output, num_targets, activation=activation))

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
        R2 = Rk - Ri
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    @tf.function(input_signature=[{
                'Z': tf.TensorSpec(name='Z', shape=[None], dtype=tf.int32),
                'R': tf.TensorSpec(name='R', shape=[None, 3], dtype=tf.float32),
                'batch_seg': tf.TensorSpec(name='batch_seg', shape=[None], dtype=tf.int32),
                'idnb_i': tf.TensorSpec(name='idnb_i', shape=[None], dtype=tf.int32),
                'idnb_j': tf.TensorSpec(name='idnb_j', shape=[None], dtype=tf.int32),
                'id_expand_kj': tf.TensorSpec(name='id_expand_kj', shape=[None], dtype=tf.int32),
                'id_reduce_ji': tf.TensorSpec(name='id_reduce_ji', shape=[None], dtype=tf.int32),
                'id3dnb_i': tf.TensorSpec(name='id3dnb_i', shape=[None], dtype=tf.int32),
                'id3dnb_j': tf.TensorSpec(name='id3dnb_j', shape=[None], dtype=tf.int32),
                'id3dnb_k': tf.TensorSpec(name='id3dnb_k', shape=[None], dtype=tf.int32)
                }])
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

        P = tf.squeeze(tf.math.segment_sum(P, batch_seg))
        return P
