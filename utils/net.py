import numpy as np
import lasagne as l
from lasagne.regularization import regularize_layer_params, l2
import theano
import theano.tensor as T
from lasagne.utils import one_hot
import pandas as pd
import time

# if theano.config.device == 'cpu':
#     from l.layers import Conv2DLayer as ConvLayer
#     from l.layers import MaxPool2DLayer as PoolLayer
# else:
#     from l.layers.dnn import Conv2DDNNLayer as ConvLayer
#     from l.layers.dnn import MaxPool2DDNNLayer as PoolLayer

from collections import OrderedDict

import cPickle as pickle

import logging
log = logging.getLogger('make_net')


class Net(object):

    def __init__(self, net_obj, solver,
                 train_loader, val_loader, test_loader):

        self.batch_sz = 256
        self.shuffle_train = False#True
        self.weight_decay = 0.0005

        self.train_epochs = 1000
        self.val_freq = 25
        self.snapshot_freq = 5000
        self.snapshot_dir = 'data'

        self.solver = solver
        self.learning_rate_mul = None

        self.step_epoch = 60
        self.gamma = 0.1

        self.input_layer = 'input'
        self.output_layer = 'prob'

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # cache number of classes
        self.num_classes_ = self.test_loader.num_classes

        self.net_obj_ = net_obj

        self.iter_funcs_ = {
            'train': None,
            'val': None
        }

        self.init_iter_funcs_()

    def train(self):

        epoch = 0

        i = 0

        while epoch < self.train_epochs:
            epoch += 1

            log.info('Epoch %d / %d' % (epoch, self.train_epochs))

            # get data

            batches = self.train_loader.batch_gen(self.batch_sz,
                                                  shuffle=self.shuffle_train)

            # iterate

            t0_bl = time.time()

            for batch in batches:

                time_bl = time.time() - t0_bl

                x, y, paths = batch
                if len(paths) < x.shape[0]:
                    # non-full batch
                    x = x[:len(paths)]
                    y = y[:len(paths)]

                if i % self.val_freq == 0:
                    log.info('Calculating validation loss...')
                    val_loss = self.val()

                    if self.snapshot_freq > 0 and i % self.snapshot_freq == 0:

                        snapshot_file = ('snapshot_iter%d_%d_%d.pkl' %
                                         (i, epoch, val_loss))
                        snapshot_path = os.path.join(self.snapshot_dir,
                                                     snapshot_file)
                        log.info('Saving snapshot %s...' % snapshot_file)

                        self.save_weights(snapshot_path)

                # if i % self.val_freq == 0:
                #     x_arr = np.asarray(x, dtype=theano.config.floatX)
                #     #print(x_arr.shape)
                #     output_layer = self.net_obj_[self.output_layer]
                #     output_arr_t = l.layers.get_output(output_layer, x_arr)
                #     output_arr = output_arr_t.eval()
                #     #f = theano.function([], output_arr_t)
                #     #output_arr = f()
                #     y_arr = np.asarray(y, dtype=np.int32)
                #     y_arr_one_hot = one_hot(y_arr).eval()
                #     #print(type(output_arr))
                #     #print(output_arr.shape)
                #     #print(type(y_arr_one_hot))
                #     #print(y_arr_one_hot.shape)
                #     #print(output_arr.ndim)
                #     loss_arr = -np.sum(y_arr_one_hot*np.log(np.array(output_arr)), axis=(output_arr.ndim-1))
                #     log.info(loss_arr.shape)
                #     log.info(loss_arr.mean())
                #     input_var = T.tensor4('x_tmp')
                #     target_var = T.ivector('y_tmp')
                #     output_tmp = l.layers.get_output(output_layer, input_var)
                #     loss_ten = -T.sum(one_hot(target_var)*T.log(output_tmp), axis=(output_tmp.ndim - 1))
                #     target_var_one_hot = one_hot(target_var)
                #     f = theano.function([input_var, target_var], loss_ten)
                #     f2 = theano.function([target_var], target_var_one_hot)
                #     loss_arr = f(x_arr, y_arr)
                #     target_var_one_hot_arr = f2(y_arr)
                #     log.info(loss_arr.mean())
                #     log.info(target_var_one_hot_arr)

                x = np.asarray(x, dtype=theano.config.floatX)
                y = np.asarray(one_hot(y).eval(), dtype=theano.config.floatX)

                t0 = time.time()
                loss, output, mat, grad = self.iter_funcs_['train'](x, y)
                # print output.shape
                # print mat.shape
                # print 'loss is: %f' % loss
                # y_idx = np.argmax(y, axis=1)
                # for i in range(min(10,y_idx.size)):
                #     print '%d: loss %f - ' % (y_idx[i], mat[i, y_idx[i]])
                #     out_str = ''
                #     for j in range(mat.shape[1]):
                #         if j == y_idx[i]:
                #             out_str += '[%f] ' % mat[i, j]
                #         else:
                #             out_str += '%f ' % mat[i, j]
                #     print 'full loss mat row: %s' % out_str
                #     out_str = ''
                #     for j in range(mat.shape[1]):
                #         if j == y_idx[i]:
                #             out_str += '[%f] ' % output[i, j]
                #         else:
                #             out_str += '%f ' % output[i, j]
                #     print 'output mat row: %s' % out_str
                # print np.asarray(grad)

                #a = one_hot(y).eval()
                #print a.shape
                #print a[:10, :]
                #print output.shape
                #print output
                #print np.sum(a*np.log(output), axis=(a.ndim - 1))
                #(-T.sum(one_hot(t)*T.log(x), axis=(x.ndim - 1)))# +

                if i % self.val_freq == 0:
                    log.info('Iteration %d, train loss: %f \t [%.2f (%.2f) s]' %
                             (i, loss, time.time()-t0, time_bl))

                i += 1

                t0_bl = time.time()

    def val(self):

        return self.test_exec_('val')

    def test(self):

        return self.test_exec_('test')

    def test_exec_(self, set='test'):

        assert(set in ['val', 'test'])

        if self.num_classes_ > 2:
            n = self.num_classes_
        else:
            n = 1

        #pos_neg_acc = [[] for i in range(n)] # what is this??
        #test_acc_i = [[] for i in range(n)]

        batches = self.train_loader.batch_gen(self.batch_sz,
                                              shuffle=False)

        test_loss = 0
        test_acc = 0
        test_batches = 0

        for batch in batches:
            x, y, paths = batch
            if len(paths) < x.shape[0]:
                # non-full batch
                x = x[:len(paths)]
                y = y[:len(paths)]

            x = np.asarray(x, dtype=theano.config.floatX)
            y = np.asarray(one_hot(y).eval(), dtype=theano.config.floatX)

            t0 = time.time()
            retval = self.iter_funcs_['val'](x, y)
            #output_det, loss, acc, pred, true_y = retval
            loss, acc = retval

            # for i in range(n):

            #     tp = len(pred[(pred == i) & (true_y == i)])
            #     tn = len(pred[(pred != i) & (true_y != i)])
            #     fp = len(pred[(pred == i) & (true_y != i)])
            #     fn = len(pred[(pred != i) & (true_y == i)])

            #     #pos = tp*1.0/(tp+fn)
            #     #neg = tn*1.0/(fp+tn)

            #     #avg = 0.5*(pos+neg)
            #     #pos_neg_acc[i].append(avg)
            #     #test_acc_i[i].append((tp+tn)*1.0/(tp+tn+fp+fn))

            #     test_acc_i[i].append((tp+tn)*1.0/(tp+tn+fp+fn))

            #print 'loss: %f' % loss
            #print 'acc: %f' % acc
            test_loss += loss
            test_acc += acc
            test_batches += 1

        test_loss = test_loss / test_batches
        test_acc = test_acc / test_batches * 100.0

        log.info('%s loss: %f, accuracy %f %%' %
                 (set, test_loss, test_acc))

        # per_class_acc = []
        # for i in range(n):
        #     acc = np.mean(test_acc_i[i])
        #     #pn_acc = np.mean(pos_neg_acc[i])
        #     log.info('  class %d acc: %f' % (i, acc))
        #     #log.info('  class %d acc: %f, pos_neg_acc: %f' %
        #     #         (i, acc, pn_acc))
        #     per_class_acc.append(acc)

        # log.info('%s mean per-class acc: %f' %
        #          (set, np.mean(per_class_acc)))

        return test_loss

    def save_weights(self, path):
        save_weights(self.net_obj_, path, self.output_layer)

    def load_weights(self, path):
        load_weights(self.net_obj_, path, self.output_layer)

    def load_learning_rate_mul(self, path):
        self.learning_rate_mul = load_learning_rate_mul(self.net_obj_, path,
                                                        self.output_layer)

    def init_iter_funcs_(self):

        log.info('Compiling net...')

        output_layer = self.net_obj_[self.output_layer]

        # layers = l.layers.get_all_layers(self.net_obj_[self.output_layer])
        # def loss_function(x, t):
        #     return -T.sum(t*T.log(x), axis=(x.ndim - 1))
        #     #return (-T.sum(t*T.log(x), axis=(x.ndim - 1)) +
        #     #        self.weight_decay*regularize_layer_params(layers, l2))


        def log_softmax(x):
            assert(x.ndim == 2)
            xdev = x-x.max(1,keepdims=True)
            return xdev - T.log(T.sum(T.exp(xdev),axis=1,keepdims=True))
        def loss_function_lsm(lsm, t):
            return -T.sum(t*lsm, axis=(lsm.ndim - 1))

        # setup input symbolic variables

        input_var = T.tensor4('x')
        target_var = T.matrix('y')
        target_var_idx = T.argmax(target_var, axis=1)

        output = l.layers.get_output(output_layer, input_var)
        #loss_ken = loss_function(output, target_var)
        #loss_ken = l.objectives.aggregate(loss_ken, mode='mean')
        #loss = l.objectives.categorical_crossentropy(output, target_var_idx)
        #loss = loss.mean()
        loss = loss_function_lsm(log_softmax(output), target_var)
        loss = loss.mean()

        params = l.layers.get_all_params(output_layer, trainable=True)

        grads = theano.gradient.grad(loss, params)
        #grads = l.updates.get_or_compute_grads(loss, params)
        updates = self.solver.compute_updates(params, grads,
                                              learning_rate_mul=self.learning_rate_mul)
        #updates = l.updates.nesterov_momentum(loss, params,
        #                                      learning_rate=0.01,
        #                                      momentum=0.9)

        test_output = l.layers.get_output(output_layer, input_var, deterministic=True)
        #test_loss = loss_function(test_output, target_var)
        #test_loss = l.objectives.aggregate(test_loss, mode='mean')
        #test_loss = l.objectives.categorical_crossentropy(test_output, target_var_idx)
        #test_loss = test_loss.mean()
        test_loss = loss_function_lsm(log_softmax(test_output), target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), target_var_idx),
                          dtype=theano.config.floatX)

        mat = -target_var*T.log(output)
        grad = grads[-1]
        log.info('train_fn...')
        train_fn = theano.function([input_var, target_var], [loss, output, mat, grad], updates=updates)
        log.info('val_fn...')
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        log.info('DONE')

        # get output given X_batch input
        #output_train = l.layers.get_output(output_layer, input_var)
        #output_val = l.layers.get_output(output_layer, input_var,
        #                                 deterministic=True)

        # get loss given y_batch input
        #loss_train = loss_function(output_train, y_batch)
        #loss_train = l.objectives.aggregate(loss_train, mode='mean')
        #loss = l.objectives.categorical_crossentropy(output_train, target_var)
        #loss = loss.mean()

        #loss_val = loss_function(output_det, y_batch)
        #loss_val = l.objectives.categorical_crossentropy(output_val, target_var)
        #loss_val = loss_val.mean()

        #pred = T.argmax(output_val, axis=1)
        #true_y = T.argmax(target_var, axis=1)
        #acc = T.mean(T.eq(pred, true_y), dtype=theano.config.floatX)

        #grads = l.updates.get_or_compute_grads(loss_train, all_params)
        #updates = self.solver.compute_updates(all_params, grads,
        #                                      learning_rate_mul=self.learning_rate_mul)
        #updates = l.updates.nesterov_momentum(
        #    loss, all_params, learning_rate=0.01, momentum=0.9)

        # set iter funcs

        #log.info('Compiling train function...')
        #train_func = theano.function([input_var, target_var],
        #                             loss,
        #                             updates=updates)

        #log.info('Compiling val function...')
        #val_func = theano.function([input_var, target_var],
        #                           [output_val, loss_val, acc, pred, true_y])

        self.iter_funcs_ = {
            'train': train_fn,
            'val': val_fn
        }


####

def save_weights(self, net_obj, path,
                 output_layer='prob'):

    all_params = l.layers.get_all_params(net_obj[output_layer])

    weights = []
    for param in all_params:
        weights.append(param.eval())
    pickle.dump(weights, open(path, 'wb'))

def load_weights(self, net_obj, path,
                 output_layer='prob'):

    all_params = l.layers.get_all_params(net_obj[output_layer])

    weights = pickle.load(open(path))

    #if len(weights) > len(all_params)
    assert len(weights) == len(all_params)

    l.layers.set_all_param_values(net_obj[output_layer], weights)

def load_learning_rate_mul(self, net_obj, path,
                           output_layer='prob'):

    learning_rate_mul = dict()
    mismatches = 0

    names = l.layers.get_all_params(net_obj[output_layer])
    layer_names = pd.unique(names)

    with open(path, 'r') as f:
        for line, layer in zip(f, layer_names):

            (key, val) = line.strip().split()
            if key != layer:
                log.warning('Not matching: %s <-> %s' %
                            key, layer)
                mismatches += 1
                learning_rate_mul[key] = 1.0
                continue

            learning_rate_mul[key] = float(val)

        if mismatches > 0:

            log.warning('Please ensure names in learning rate multipliers file match names'
                        ' of layers EXACTLY - setting lr to 1.0 for mismatched layers')

        if len(self.learning_rate_mul.keys())!= len(layer_names):

            log.warning('Incorrect number of learning rate multipliers provided!'
                        'Require learning multipliers for:')
            for name in layer_names:
                log.warning(name)
