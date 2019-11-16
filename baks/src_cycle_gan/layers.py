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


# use k-layer dense network as generator
class Generator(Layer):
    def __init__(self, input_dim, output_dim, num_layers, dropout=0.0, act=tf.nn.relu, name=None):
        super(Generator, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.num_layers = num_layers
        self.layers = []
        self.vars = []
        for i in range(num_layers):
            dense_name = self.name + "_gen_dense" + str(i) if name is not None else "gen_dense" + str(i)
            dim_out = input_dim if i < num_layers - 1 else output_dim
            dense = Dense(input_dim, dim_out, dropout=dropout, act=act, name=dense_name)
            self.layers.append(dense)
            self.vars += dense.vars

    def _call(self, inputs):
        output = inputs
        for dense in self.layers:
            output = dense(output)
        return output


# use k-layer dense network as discriminator
class Discriminator(Layer):
    def __init__(self, input_dim, num_layers, dropout=0.0, act=tf.nn.relu, name=None):
        super(Discriminator, self).__init__(name)
        self.input_dim = input_dim
        self.dropout = dropout
        self.act = act
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            dense_name = self.name + "_gen_dense" + str(i) if name is not None else "gen_dense" + str(i)
            dim_out = input_dim if i < num_layers - 1 else 1
            act_out = act if i < num_layers - 1 else tf.nn.sigmoid
            dense = Dense(input_dim, dim_out, dropout=dropout, act=act_out, name=dense_name)
            self.layers.append(dense)
            self.vars += dense.vars

    def _call(self, inputs):
        output = inputs
        for dense in self.layers:
            output = dense(output)
        return output

class CycleGANUnit(Layer):
    def __init__(self, dim, lmbd=1, real_label=0.9, name=None):
        super(CycleGANUnit, self).__init__(name)
        self.dim = dim
        self.lmbd = lmbd
        self.real_label = real_label
        with tf.variable_scope(self.name):
            self.dense_item = Dense(2 * dim, dim, name="dense_item")
            self.dense_head = Dense(2 * dim, dim, name="dense_head")
            self.generator_G = Generator(dim, dim, 2, name='generator_g')
            self.generator_F = Generator(dim, dim, 2, name='generator_f')
            self.discriminator_V = Discriminator(dim, 1, name='discriminator_v')
            self.discriminator_E = Discriminator(dim, 1, name='discriminator_e')
        self.vars = self.dense_item.vars + self.dense_head.vars \
                 + self.generator_G.vars + self.generator_F.vars \
                 + self.discriminator_V.vars + self.discriminator_E.vars
    
    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        fake_e = self.generator_G(v)
        fake_v = self.generator_F(e)

        cv = tf.concat([v, fake_v], 1)
        ce = tf.concat([e, fake_e], 1)

        update_v = self.dense_item(cv)
        update_e = self.dense_head(ce)

        return update_v, update_e, fake_v, fake_e

    def get_rs_loss(self, inputs, fakes):
        v, e = inputs
        fv, fe = fakes
        error_real = tf.reduce_mean(tf.squared_difference(self.discriminator_V(v), self.real_label))
        error_fake = tf.reduce_mean(tf.square(self.discriminator_V(fv)))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self.generator_G(fv) - e))
        rs_loss = discriminator_loss + self.lmbd * consistency_loss 
        return rs_loss

    def get_kg_loss(self, inputs, fakes):
        v, e = inputs
        fv, fe = fakes
        error_real = tf.reduce_mean(tf.squared_difference(self.discriminator_E(e), self.real_label))
        error_fake = tf.reduce_mean(tf.square(self.discriminator_E(fe)))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self.generator_F(fe) - v))
        kg_loss = discriminator_loss + self.lmbd * consistency_loss
        return kg_loss

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
