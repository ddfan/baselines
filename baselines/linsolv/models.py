import tensorflow as tf
import tensorflow.contrib as tc

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

class LinSolvActorCritic(Model):
    def __init__(self, nb_actions, name='actorcritic', layer_norm=True):
        super(LinSolvActorCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions

    def __call__(self, obs, action, reuse=False, rescale=False, return_action=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            q = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            
            if return_action:
                #calculate action as gradient of critic w.r.t action input, scaled.
                all_actions=tf.map_fn(lambda q_: tf.concat([tf.gradients(q_[i],action) for i in range(self.nb_actions)],axis=0), q)
                if rescale:
                    scale_action=tf.get_variable('scale_action',[self.nb_actions])
                    shift_action=tf.get_variable('shift_action',[self.nb_actions])
                    all_actions=tf.multiply(all_actions,scale_action)+shift_action

                return q,all_actions

            else:
                return q

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
