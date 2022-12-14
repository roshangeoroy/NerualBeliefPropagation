
import tensorflow as tf
import numpy as np
import scipy as sp # for sparse H matrix computations
from keras.layers import Layer
import matplotlib.pyplot as plt
class LDPCBPDecoder(Layer):
    # pylint: disable=line-too-long
    def __init__(self,
                 pcm,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=output_dtype, **kwargs)

        assert isinstance(trainable, bool), 'trainable must be bool.'
        assert isinstance(hard_out, bool), 'hard_out must be bool.'
        assert isinstance(track_exit, bool), 'track_exit must be bool.'
        assert isinstance(cn_type, str) , 'cn_type must be str.'
        assert isinstance(num_iter, int), 'num_iter must be int.'
        assert num_iter>=0, 'num_iter cannot be negative.'
        assert isinstance(stateful, bool), 'stateful must be bool.'
        assert isinstance(output_dtype, tf.DType), \
                                'output_dtype must be tf.Dtype.'

        if isinstance(pcm, np.ndarray):
            assert np.array_equal(pcm, pcm.astype(bool)), 'PC matrix \
                must be binary.'
        elif isinstance(pcm, sp.sparse.csr_matrix):
            assert np.array_equal(pcm.data, pcm.data.astype(bool)), \
                'PC matrix must be binary.'
        elif isinstance(pcm, sp.sparse.csc_matrix):
            assert np.array_equal(pcm.data, pcm.data.astype(bool)), \
                'PC matrix must be binary.'
        else:
            raise TypeError("Unsupported dtype of pcm.")

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder uses tf.float32 for internal calculations.')

        # init decoder parameters
        self._pcm = pcm
        self._trainable = trainable
        self._cn_type = cn_type
        self._hard_out = hard_out
        self._track_exit = track_exit
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._stateful = stateful
        self._output_dtype = output_dtype

        # clipping value for the atanh function is applied (tf.float32 is used)
        self._atanh_clip_value = 1 - 1e-7
        # internal value for llr clipping
        self._llr_max = 20

        # init code parameters
        self._num_cns = pcm.shape[0] # total number of check nodes
        self._num_vns = pcm.shape[1] # total number of variable nodes

        # make pcm sparse first if ndarray is provided
        if isinstance(pcm, np.ndarray):
            pcm = sp.sparse.csr_matrix(pcm)
        # find all edges from variable and check node perspective
        self._cn_con, self._vn_con, _ = sp.sparse.find(pcm)

        # number of edges equals number of non-zero elements in the
        # parity-check matrix
        self._num_edges = len(self._vn_con)

        # permutation index to rearrange messages into check node perspective
        self._ind_cn = np.argsort(self._cn_con)

        # inverse permutation index to rearrange messages back into variable
        # node perspective
        self._ind_cn_inv = np.argsort(self._ind_cn)

        # generate row masks (array of integers defining the row split pos.)
        self._vn_row_splits = self._gen_node_mask_row(self._vn_con)
        self._cn_row_splits = self._gen_node_mask_row(
                                                    self._cn_con[self._ind_cn])
        # pre-load the CN function for performance reasons
        if self._cn_type=='boxplus':
            # check node update using the tanh function
            self._cn_update = self._cn_update_tanh
        elif self._cn_type=='boxplus-phi':
            # check node update using the "_phi" function
            self._cn_update = self._cn_update_phi
        elif self._cn_type=='minsum':
            # check node update using the min-sum approximation
            self._cn_update = self._cn_update_minsum
        else:
            raise ValueError('Unknown node type.')

        # init trainable weights if needed
        self._has_weights = False  # indicates if trainable weights exist
        if self._trainable:
            self._has_weights = True
            self._edge_weights = tf.Variable(tf.ones(self._num_edges),
                                             trainable=self._trainable,
                                             dtype=tf.float32)

        # track mutual information during decoding
        self._ie_c = 0
        self._ie_v = 0

    @property
    def pcm(self):
        """Parity-check matrix of LDPC code."""
        return self._pcm

    @property
    def num_cns(self):
        """Number of check nodes."""
        return self._num_cns

    @property
    def num_vns(self):
        """Number of variable nodes."""
        return self._num_vns

    @property
    def num_edges(self):
        """Number of edges in decoding graph."""
        return self._num_edges

    @property
    def has_weights(self):
        """Indicates if decoder has trainable weights."""
        return self._has_weights

    @property
    def edge_weights(self):
        """Trainable weights of the BP decoder."""
        if not self._has_weights:
            return []
        else:
            return self._edge_weights

    @property
    def output_dtype(self):
        """Output dtype of decoder."""
        return self._output_dtype

    @property
    def ie_c(self):
        "Extrinsic mutual information at check node."
        return self._ie_c

    @property
    def ie_v(self):
        "Extrinsic mutual information at variable node."
        return self._ie_v

    @property
    def num_iter(self):
        "Number of decoding iterations."
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter):
        "Number of decoding iterations."
        assert isinstance(num_iter, int), 'num_iter must be int.'
        assert num_iter>=0, 'num_iter cannot be negative.'
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

    def show_weights(self, size=7):
        """Show histogram of trainable weights.
        Input
        -----
            size: float
                Figure size of the matplotlib figure.
        """
        # only plot if weights exist
        if self._has_weights:
            weights = self._edge_weights.numpy()

            plt.figure(figsize=(size,size))
            plt.hist(weights, density=True, bins=20, align='mid')
            plt.xlabel('weight value')
            plt.ylabel('density')
            plt.grid(True, which='both', axis='both')
            plt.title('Weight Distribution')
        else:
            print("No weights to show.")

    def _gen_node_mask(self, con):
        """ Generates internal node masks indicating which msg index belongs
        to which node index.
        """
        ind = np.argsort(con)
        con = con[ind]

        node_mask = []

        cur_node = 0
        cur_mask = []
        for i in range(self._num_edges):
            if con[i] == cur_node:
                cur_mask.append(ind[i])
            else:
                node_mask.append(cur_mask)
                cur_mask = [ind[i]]
                cur_node += 1
        node_mask.append(cur_mask)
        return node_mask

    def _gen_node_mask_row(self, con):
        """ Defining the row split positions of a 1D vector consisting of all
        edges messages.
        Used to build a ragged Tensor of incoming node messages.
        """
        node_mask = [0] # the first element indicates the first node index (=0)

        cur_node = 0
        for i in range(self._num_edges):
            if con[i] != cur_node:
                node_mask.append(i)
                cur_node += 1
        node_mask.append(self._num_edges) # last element must be the number of
        # elements (delimiter)
        return node_mask

    def _vn_update(self, msg, llr_ch):
        """ Variable node update function.
        This function implements the (extrinsic) variable node update
        function. It takes the sum over all incoming messages ``msg`` excluding
        the intrinsic (= outgoing) message itself.
        Additionally, the channel LLR ``llr_ch`` is added to each message.
        """
        # aggregate all incoming messages per node
        x = tf.reduce_sum(msg, axis=1)
        x = tf.add(x, llr_ch)

        # subtract extrinsic message from node value
        # x = tf.expand_dims(x, axis=1)
        # x = tf.add(-msg, x)
        x = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                      x + tf.gather(y, row_ind),
                                      -1.*msg,
                                      x,
                                      msg.value_rowids())
        return x

    def _extrinsic_min(self, msg):
        num_val = tf.shape(msg)[0]
        msg = tf.transpose(msg, (1,0))
        msg = tf.expand_dims(msg, axis=1)
        id_mat = tf.eye(num_val)

        msg = (tf.tile(msg, (1, num_val, 1)) # create outgoing tensor per value
               + 1000. * id_mat) # "ignore" intrinsic msg by adding large const.


        msg = tf.math.reduce_min(msg, axis=2)
        msg = tf.transpose(msg, (1,0))
        return msg

    def _where_ragged(self, msg):
        return tf.where(tf.equal(msg, 0), tf.ones_like(msg) * 1e-12, msg)

    def _where_ragged_inv(self, msg):
        msg_mod =  tf.where(tf.less(tf.abs(msg), 1e-7),
                            tf.zeros_like(msg),
                            msg)
        return msg_mod

    def _cn_update_tanh(self, msg):
        """Check node update function implementing the exact boxplus operation.
        This function implements the (extrinsic) check node update
        function. It calculates the boxplus function over all incoming messages
        "msg" excluding the intrinsic (=outgoing) message itself.
        The exact boxplus function is implemented by using the tanh function.
        The input is expected to be a ragged Tensor of shape
        `[num_cns, None, batch_size]`.
        Note that for numerical stability clipping is applied.
        """

        msg = msg / 2
        # tanh is not overloaded for ragged tensors
        msg = tf.ragged.map_flat_values(tf.tanh, msg) # tanh is not overloaded

        # for ragged tensors; map to flat tensor first
        msg = tf.ragged.map_flat_values(self._where_ragged, msg)

        msg_prod = tf.reduce_prod(msg, axis=1)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # ^-1 to avoid division
        # Note this is (potentially) numerically unstable
        # msg = msg**-1 * tf.expand_dims(msg_prod, axis=1) # remove own edge

        msg = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                        x * tf.gather(y, row_ind),
                                        msg**-1,
                                        msg_prod,
                                        msg.value_rowids())

        # Overwrite small (numerical zeros) message values with exact zero
        # these are introduced by the previous "_where_ragged" operation
        # this is required to keep the product stable (cf. _phi_update for log
        # sum implementation)
        msg = tf.ragged.map_flat_values(self._where_ragged_inv, msg)

        msg = tf.clip_by_value(msg,
                               clip_value_min=-self._atanh_clip_value,
                               clip_value_max=self._atanh_clip_value)

        # atanh is not overloaded for ragged tensors
        msg = 2 * tf.ragged.map_flat_values(tf.atanh, msg)
        return msg

    def _phi(self, x):
        """Helper function for the check node update.
        This function implements the (element-wise) `"_phi"` function as defined
        in [Ryan]_.
        """
        # the clipping values are optimized for tf.float32
        x = tf.clip_by_value(x, clip_value_min=8.5e-8, clip_value_max=16.635532)
        return tf.math.log(tf.math.exp(x)+1) - tf.math.log(tf.math.exp(x)-1)

    def _cn_update_phi(self, msg):
        """Check node update function implementing the exact boxplus operation.
        This function implements the (extrinsic) check node update function
        based on the numerically more stable `"_phi"` function (cf. [Ryan]_).
        It calculates the boxplus function over all incoming messages ``msg``
        excluding the intrinsic (=outgoing) message itself.
        The exact boxplus function is implemented by using the `"_phi"` function
        as in [Ryan]_.
        The input is expected to be a ragged Tensor of shape
        `[num_cns, None, batch_size]`.
        Note that for numerical stability clipping is applied.
        """

        sign_val = tf.sign(msg)

        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        # sign_val = sign_val * tf.expand_dims(sign_node, axis=1)
        sign_val = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                             x * tf.gather(y, row_ind),
                                             sign_val,
                                             sign_node,
                                             sign_val.value_rowids())

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # apply _phi element-wise (does not support ragged Tensors)
        msg = tf.ragged.map_flat_values(self._phi, msg)
        msg_sum = tf.reduce_sum(msg, axis=1)

        # TF2.9 does not support XLA for the addition of ragged tensors
        # the following code provides a workaround that supports XLA

        # msg = tf.add( -msg, tf.expand_dims(msg_sum, axis=1)) # remove own edge
        msg = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                        x + tf.gather(y, row_ind),
                                        -1.*msg,
                                        msg_sum,
                                        msg.value_rowids())

        # apply _phi element-wise (does not support ragged Tensors)
        msg = self._stop_ragged_gradient(sign_val) * tf.ragged.map_flat_values(
                                                            self._phi, msg)
        return msg

    def _stop_ragged_gradient(self, rt):
        """Helper function as TF 2.5 does not support ragged gradient
        stopping"""
        return rt.with_flat_values(tf.stop_gradient(rt.flat_values))

    def _sign_val_minsum(self, msg):
        """Helper to replace find sign-value during min-sum decoding.
        Must be called with `map_flat_values`."""

        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)
        return sign_val

    def _cn_update_minsum_mapfn(self, msg):
        """ Check node update function implementing the min-sum approximation.
        This function approximates the (extrinsic) check node update
        function based on the min-sum approximation (cf. [Ryan]_).
        It calculates the "extrinsic" min function over all incoming messages
        ``msg`` excluding the intrinsic (=outgoing) message itself.
        The input is expected to be a ragged Tensor of shape
        `[num_vns, None, batch_size]`.
        This function uses tf.map_fn() to call the CN updates.
        It is currently not used, but can be used as template to implement
        modified CN functions (e.g., offset-corrected minsum).
        Please note that tf.map_fn lowers the throughput significantly.
        """

        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_val = self._stop_ragged_gradient(sign_val) * tf.expand_dims(
                                                             sign_node, axis=1)

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # calculate extrinsic messages and include the sign
        msg_e = tf.map_fn(self._extrinsic_min, msg, infer_shape=False)

        # ensure shape after map_fn
        msg_fv = msg_e.flat_values
        msg_fv = tf.ensure_shape(msg_fv, msg.flat_values.shape)
        msg_e = msg.with_flat_values(msg_fv)

        msg = sign_val * msg_e

        return msg

    def _cn_update_minsum(self, msg):
        """ Check node update function implementing the min-sum approximation.
        This function approximates the (extrinsic) check node update
        function based on the min-sum approximation (cf. [Ryan]_).
        It calculates the "extrinsic" min function over all incoming messages
        ``msg`` excluding the intrinsic (=outgoing) message itself.
        The input is expected to be a ragged Tensor of shape
        `[num_vns, None, batch_size]`.
        """
        # a constant used overwrite the first min
        LARGE_VAL = 10000. # pylint: disable=invalid-name

        # clip values for numerical stability
        msg = tf.clip_by_value(msg,
                               clip_value_min=-self._llr_max,
                               clip_value_max=self._llr_max)

        # calculate sign of outgoing msg
        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # sign_val = self._stop_ragged_gradient(sign_val) \
        #             * tf.expand_dims(sign_node, axis=1)
        sign_val = tf.ragged.map_flat_values(
                                        lambda x, y, row_ind:
                                        tf.multiply(x, tf.gather(y, row_ind)),
                                        self._stop_ragged_gradient(sign_val),
                                        sign_node,
                                        sign_val.value_rowids())

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # Calculate the extrinsic minimum per CN, i.e., for each message of
        # index i, find the smallest and the second smallest value.
        # However, in some cases the second smallest value may equal the
        # smallest value (multiplicity of mins).
        # Please note that this needs to be applied to raggedTensors, e.g.,
        # tf.top_k() is currently not supported and the ops must support graph
        # # mode.

        # find min_value per node
        min_val = tf.reduce_min(msg, axis=1, keepdims=True)

        # TF2.9 does not support XLA for the subtraction of ragged tensors
        # the following code provides a workaround that supports XLA

        # and subtract min; the new array contains zero at the min positions
        # benefits from broadcasting; all other values are positive
        # msg_min1 = msg - min_val
        msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind:
                                             x- tf.gather(y, row_ind),
                                             msg,
                                             tf.squeeze(min_val, axis=1),
                                             msg.value_rowids())

        # replace 0 (=min positions) with large value to ignore it for further
        # min calculations
        msg = tf.ragged.map_flat_values(lambda x:
                                        tf.where(tf.equal(x, 0), LARGE_VAL, x),
                                        msg_min1)

        # find the second smallest element (we add min_val as this has been
        # subtracted before)
        min_val2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val

        # Detect duplicated minima (i.e., min_val occurs at two incoming
        # messages). As the LLRs per node are <LLR_MAX and we have
        # replace at least 1 position (position with message "min_val") by
        # LARGE_VAL, it holds for the sum < LARGE_VAL + node_degree*LLR_MAX.
        # if the sum > 2*LARGE_VAL, the multiplicity of the min is at least 2.
        node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*LARGE_VAL-1.)
        # indicator that duplicated min was detected (per node)
        double_min = 0.5*(1-tf.sign(node_sum))

        # if a duplicate min occurred, both edges must have min_val, otherwise
        # the second smallest value is taken
        min_val_e = (1-double_min) * min_val + (double_min) * min_val2

        # replace all values with min_val except the position where the min
        # occurred (=extrinsic min).
        msg_e = tf.where(msg==LARGE_VAL, min_val_e, min_val)

        # it seems like tf.where does not set the shape of tf.ragged properly
        # we need to ensure the shape manually
        msg_e = tf.ragged.map_flat_values(
                                    lambda x:
                                    tf.ensure_shape(x, msg.flat_values.shape),
                                    msg_e)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # and apply sign
        #msg = sign_val * msg_e
        msg = tf.ragged.map_flat_values(tf.multiply,
                                        sign_val,
                                        msg_e)

        return msg

    def _mult_weights(self, x):
        """Multiply messages with trainable weights for weighted BP."""
        # transpose for simpler broadcasting of training variables
        x = tf.transpose(x, (1, 0))
        x = tf.math.multiply(x, self._edge_weights)
        x = tf.transpose(x, (1, 0))
        return x

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        # Raise AssertionError if shape of x is invalid
        if self._stateful:
            assert(len(input_shape)==2), \
                "For stateful decoding, a tuple of two inputs is expected."
            input_shape = input_shape[0]

        assert (input_shape[-1]==self._num_vns), \
                            'Last dimension must be of length n.'
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

    def call(self, inputs):
        """Iterative BP decoding function.
        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.
        Args:
        llr_ch or (llr_ch, msg_vn):
            llr_ch (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.
            msg_vn (tf.float32) : Ragged tensor containing the VN
                messages, or None. Required if ``stateful`` is set to True.
        Returns:
            `tf.float32`: Tensor of shape `[...,n]` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits.
        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.
            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        # internal calculations still in tf.float32
        llr_ch = tf.cast(llr_ch, tf.float32)

        # clip llrs for numerical stability
        llr_ch = tf.clip_by_value(llr_ch,
                                  clip_value_min=-self._llr_max,
                                  clip_value_max=self._llr_max)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(llr_ch)[-1],
                                  self._num_vns,
                                  'Last dimension must be of length n.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)

        # must be done during call, as XLA fails otherwise due to ragged
        # indices placed on the CPU device.
        # create permutation index from cn perspective
        self._cn_mask_tf = tf.ragged.constant(self._gen_node_mask(self._cn_con),
                                              row_splits_dtype=tf.int32)

        # batch dimension is last dimension due to ragged tensor representation
        llr_ch = tf.transpose(llr_ch_reshaped, (1,0))

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # init internal decoder state if not explicitly
        # provided (e.g., required to restore decoder state for iterative
        # detection and decoding)
        # load internal state from previous iteration
        # required for iterative det./dec.
        if not self._stateful or msg_vn is None:
            msg_shape = tf.stack([tf.constant(self._num_edges),
                                   tf.shape(llr_ch)[1]],
                                   axis=0)
            msg_vn = tf.zeros(msg_shape, dtype=tf.float32)
        else:
            msg_vn = msg_vn.flat_values

        # track exit decoding trajectory; requires all-zero cw?
        if self._track_exit:
            self._ie_c = tf.zeros(self._num_iter+1)
            self._ie_v = tf.zeros(self._num_iter+1)

        # perform one decoding iteration
        # Remark: msg_vn cannot be ragged as input for tf.while_loop as
        # otherwise XLA will not be supported (with TF 2.5)
        def dec_iter(llr_ch, msg_vn, it):
            it += 1

            msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))
            # variable node update
            msg_vn = self._vn_update(msg_vn, llr_ch)

            # track exit decoding trajectory; requires all-zero cw
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1. * msg_vn.flat_values)
                self._ie_v = tf.tensor_scatter_nd_add(self._ie_v,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # scale outgoing vn messages (weighted BP); only if activated
            if self._has_weights:
                msg_vn = tf.ragged.map_flat_values(self._mult_weights,
                                                   msg_vn)
            # permute edges into CN perspective
            msg_cn = tf.gather(msg_vn.flat_values, self._cn_mask_tf, axis=None)

            # check node update using the pre-defined function
            msg_cn = self._cn_update(msg_cn)

            # track exit decoding trajectory; requires all-zero cw?
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1.*msg_cn.flat_values)
                # update pos i+1 such that first iter is stored as 0
                self._ie_c = tf.tensor_scatter_nd_add(self._ie_c,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # re-permute edges to variable node perspective
            msg_vn = tf.gather(msg_cn.flat_values, self._ind_cn_inv, axis=None)
            return llr_ch, msg_vn, it

        # stopping condition (required for tf.while_loop)
        def dec_stop(llr_ch, msg_vn, it): # pylint: disable=W0613
            return tf.less(it, self._num_iter)

        # start decoding iterations
        it = tf.constant(0)
        # maximum_iterations required for XLA
        _, msg_vn, _ = tf.while_loop(dec_stop,
                                     dec_iter,
                                     (llr_ch, msg_vn, it),
                                     parallel_iterations=1,
                                     maximum_iterations=self._num_iter)


        # raggedTensor for final marginalization
        msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))

        # marginalize and remove ragged Tensor
        x_hat = tf.add(llr_ch, tf.reduce_sum(msg_vn, axis=1))

        # restore batch dimension to first dimension
        x_hat = tf.transpose(x_hat, (1,0))

        x_hat = -1. * x_hat # convert llrs back into logits

        if self._hard_out: # hard decide decoder output if required
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = llr_ch_shape
        output_shape[0] = -1 # overwrite batch dim (can be None in Keras)

        x_reshaped = tf.reshape(x_hat, output_shape)

        # cast output to output_dtype
        x_out = tf.cast(x_reshaped, self._output_dtype)

        if not self._stateful:
            return x_out
        else:
            return x_out, 
    
def llr2mi(llr, s=None, reduce_dims=True):
        # pylint: disable=line-too-long
        
        if s is None:
            s = tf.ones_like(llr)

        if llr.dtype not in (tf.float16, tf.bfloat16, tf.float32, tf.float64):
            raise TypeError("Dtype of llr must be a real-valued float.")

        # ensure that both tensors are compatible
        s = tf.cast(s, llr.dtype)

        # scramble sign as if all-zero cw was transmitted
        llr_zero = tf.multiply(s, llr)
        llr_zero = tf.clip_by_value(llr_zero, -20., 20.) # clip for stability
        x = np.log2(1. + tf.exp(1.* llr_zero))
        if reduce_dims:
            x = 1. - tf.reduce_mean(x)
        else:
            x = 1. - tf.reduce_mean(x, axis=-1)
        return 
    