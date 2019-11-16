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
        

class CycleGANUnit(Layer):
    def __init__(self, dim, lmbd=1, real_label=0.9, name=None):
        super(CycleGANUnit, self).__init__(name)
        self.dim = dim
        self.lmbd = lmbd
        self.real_label = real_label
        with tf.variable_scope(self.name):
            self.generator_ve = tf.get_variable(name='gw_ve', shape=(dim, dim), dtype=tf.float32)
            self.generator_ev = tf.get_variable(name='gw_ev', shape=(dim, dim), dtype=tf.float32)
            self.discriminator_v = tf.get_variable(name='dw_v', shape=(dim, 1), dtype=tf.float32)
            self.discriminator_e = tf.get_variable(name='dw_e', shape=(dim, 1), dtype=tf.float32)
            self.mlp_v = tf.get_variable(name='mlp_v', shape=(2 * dim, dim), dtype=tf.float32)
            self.mlp_e = tf.get_variable(name='mlp_e', shape=(2 * dim, dim), dtype=tf.float32)
            self.gbias_ve = tf.get_variable(name='gb_ve', shape=dim, initializer=tf.zeros_initializer())
            self.gbias_ev = tf.get_variable(name='gb_ev', shape=dim, initializer=tf.zeros_initializer())
            self.dbias_v = tf.get_variable(name='db_v', shape=1, initializer=tf.zeros_initializer())
            self.dbias_e = tf.get_variable(name='db_e', shape=1, initializer=tf.zeros_initializer())
            self.mbias_v = tf.get_variable(name='mb_v', shape=dim, initializer=tf.zeros_initializer())
            self.mbias_e = tf.get_variable(name='mb_e', shape=dim, initializer=tf.zeros_initializer())
        self.vars = [self.generator_ve, self.generator_ev, self.discriminator_v, self.discriminator_e, self.mlp_v, self.mlp_e]

    def _generator(self, inp, mode='ve'):
        weight = self.generator_ve if mode == 've' else self.generator_ev
        bias = self.gbias_ve if mode == 've' else self.gbias_ev
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def _discriminator(self, inp, mode='v'):
        weight = self.discriminator_v if mode == 'v' else self.discriminator_e
        bias = self.dbias_v if mode == 'v' else self.dbias_e
        return tf.nn.sigmoid(tf.matmul(inp, weight) + bias)

    def _mlp(self, inp, mode='v'):
        weight = self.mlp_v if mode == 'v' else self.mlp_e
        bias = self.mbias_v if mode == 'v' else self.mbias_e
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs
        fake_v = self._generator(e, mode='ev')
        fake_e = self._generator(v, mode='ve')
        
        cv = tf.concat([v, fake_v], 1)
        ce = tf.concat([e, fake_e], 1)

        v_output = self._mlp(cv, mode='v')
        e_output = self._mlp(ce, mode='e')

        return v_output, e_output, fake_v, fake_e

    def get_rs_loss(self, inputs, fakes):
        v, e = inputs
        fv, fe = fakes
        error_real = tf.reduce_mean(tf.squared_difference(self._discriminator(v, mode='v'), self.real_label))
        error_fake = tf.reduce_mean(tf.square(self._discriminator(fv, mode='v')))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fv, mode='ve') - e))
        rs_loss = discriminator_loss + self.lmbd * consistency_loss 
        return rs_loss

    def get_kge_loss(self, inputs, fakes):
        v, e = inputs
        fv, fe = fakes
        error_real = tf.reduce_mean(tf.squared_difference(self._discriminator(e, mode='e'), self.real_label))
        error_fake = tf.reduce_mean(tf.square(self._discriminator(fe, mode='e')))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fe, mode='ev') - v))
        kge_loss = discriminator_loss + self.lmbd * consistency_loss 
        return kge_loss


class FCUnit(Layer):
    def __init__(self, dim, name=None):
        super(FCUnit, self).__init__(name)
        self.dim = dim
        with tf.variable_scope(self.name):
            self.weight_v = tf.get_variable(name='weight_v', shape=(2 * dim, dim), dtype=tf.float32)
            self.weight_e = tf.get_variable(name='weight_e', shape=(2 * dim, dim), dtype=tf.float32)
            self.bias_v = tf.get_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight_v, self.weight_e]

    def _call(self, inputs):
        # [batch_size, dim]
        inp = tf.concat(inputs, 1)
        v_output = tf.nn.relu(tf.matmul(inp, self.weight_v) + self.bias_v)
        e_output = tf.nn.relu(tf.matmul(inp, self.weight_e) + self.bias_e)
        return v_output, e_output


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
