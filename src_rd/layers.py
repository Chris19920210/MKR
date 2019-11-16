import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Layer(object):
    def __init__(self, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.vars = []

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs

    @abstractmethod
    def _call(self, inputs):
        pass


class Dense(Layer):
    def __init__(self, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, name=None):
        super(Dense, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name):
            self.weight = tf.get_variable(name='weight', shape=(input_dim, output_dim), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias', shape=output_dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight]

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        output = tf.matmul(x, self.weight) + self.bias
        return self.act(output)


class CrossCompress2Unit(Layer):
    def __init__(self, dim, lmbd=1, real_label=0.9, name=None):
        super(CrossCompress2Unit, self).__init__(name)
        self.dim = dim
        self.lmbd = lmbd
        self.real_label = real_label
        with tf.variable_scope(self.name):
            self.weight_v = tf.get_variable(name='weight_v', shape=(2 * dim, 1), dtype=tf.float32)
            self.weight_e = tf.get_variable(name='weight_e', shape=(2 * dim, 1), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias_v', shape=2 * dim, initializer=tf.zeros_initializer())

            self.wg_v = tf.get_variable(name='wg_v', shape=(dim, dim), dtype=tf.float32)
            self.wf_e = tf.get_variable(name='wf_e', shape=(dim, dim), dtype=tf.float32)
            self.bg_v = tf.get_variable(name='bg_v', shape=dim, initializer=tf.zeros_initializer())
            self.bf_e = tf.get_variable(name='bf_e', shape=dim, initializer=tf.zeros_initializer())

            self.wd_v = tf.get_variable(name='wd_v', shape=(dim, 1), dtype=tf.float32)
            self.wd_e = tf.get_variable(name='wd_e', shape=(dim, 1), dtype=tf.float32)
            self.bd_v = tf.get_variable(name='bd_v', shape=1, initializer=tf.zeros_initializer())
            self.bd_e = tf.get_variable(name='bd_e', shape=1, initializer=tf.zeros_initializer())
        self.vars = [self.weight_v, self.weight_e, self.wg_v, self.wf_e, self.wd_v, self.wd_e]
        
    def _generator(self, inp, mode):
        weight = self.wg_v if mode == 've' else self.wf_e
        bias = self.bg_v if mode == 've' else self.bf_e
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def _discriminator(self, inp, mode):
        weight = self.wd_v if mode == 'v' else self.wd_e
        bias = self.bd_v if mode == 'v' else self.bd_e
        return tf.nn.sigmoid(tf.matmul(inp, weight) + bias)

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs
        fv = self._generator(e, 'ev')
        fe = self._generator(v, 've')

        dv = tf.concat([v, fv], 1)
        de = tf.concat([e, fe], 1)

        # [batch_size, 2dim, 1], [batch_size, 1, 2dim]
        dv = tf.expand_dims(dv, dim=2)
        de = tf.expand_dims(de, dim=1)

        # [batch_size, 2dim, 2dim]
        c_matrix = tf.matmul(dv, de)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * 2dim, 2dim]
        c_matrix = tf.reshape(c_matrix, [-1, 2 * self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, 2 * self.dim])

        # [batch_size, 2dim]
        output = tf.reshape(tf.matmul(c_matrix, self.weight_v) + tf.matmul(c_matrix_transpose, self.weight_e),
                              [-1, 2 * self.dim]) + self.bias
        v_output, e_output = tf.split(output, 2, 1)
        return v_output, e_output, fv, fe

    def get_rs_loss(self, inputs, fakes):
        v, e = inputs
        fv, fe = fakes
        error_real = tf.reduce_mean(tf.squared_difference(self._discriminator(v, 'v'), self.real_label))
        error_fake = tf.reduce_mean(tf.square(self._discriminator(fv, 'v')))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fv, 've') - e))
        rs_loss = discriminator_loss + self.lmbd * consistency_loss
        return rs_loss

    def get_kge_loss(self, inputs, fakes):
        v, e = inputs
        fv, fe = fakes
        error_real = tf.reduce_mean(tf.squared_difference(self._discriminator(e, 'e'), self.real_label))
        error_fake = tf.reduce_mean(tf.square(self._discriminator(fe, 'e')))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fe, 'ev') - v))
        kge_loss = discriminator_loss + self.lmbd * consistency_loss
        return kge_loss


class CrossCompressUnit(Layer):
    def __init__(self, dim, name=None):
        super(CrossCompressUnit, self).__init__(name)
        self.dim = dim
        with tf.variable_scope(self.name):
            self.weight_vv = tf.get_variable(name='weight_vv', shape=(dim, 1), dtype=tf.float32)
            self.weight_ev = tf.get_variable(name='weight_ev', shape=(dim, 1), dtype=tf.float32)
            self.weight_ve = tf.get_variable(name='weight_ve', shape=(dim, 1), dtype=tf.float32)
            self.weight_ee = tf.get_variable(name='weight_ee', shape=(dim, 1), dtype=tf.float32)
            self.bias_v = tf.get_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee]

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = tf.expand_dims(v, dim=2)
        e = tf.expand_dims(e, dim=1)

        # [batch_size, dim, dim]
        c_matrix = tf.matmul(v, e)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])

        # [batch_size, dim]
        v_output = tf.reshape(tf.matmul(c_matrix, self.weight_vv) + tf.matmul(c_matrix_transpose, self.weight_ev),
                              [-1, self.dim]) + self.bias_v
        e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_transpose, self.weight_ee),
                              [-1, self.dim]) + self.bias_e

        return v_output, e_output
