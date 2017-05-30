import matplotlib.animation as animation
import h5py
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy
import scipy.misc
import scipy.ndimage
from sklearn.metrics import roc_curve, auc,roc_auc_score
import tensorflow as tf
import socket
import sys
import time

from layers import *
from nets_classification import *
from data_stream import *
from ops import *

def create_exec_statement_test(opts):
    """
    Creates an executable statement string.
    Basically lets us keep everything general.
    Comments show an example.
    INPUTS:
    - opts: (object) command line arguments from argparser
    """
    exec_statement = "self.pred = "
    #self.pred =
    exec_statement += opts.network
    #self.pred = GoogLe
    exec_statement += "_Net(self.xTe, self.is_training, "
    #self.pred = GoogLe_Net(self.xTe, self.is_training,
    exec_statement += str(opts.num_class)
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2
    exec_statement += ", 1"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 1
    exec_statement += ", self.keep_prob)"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 1, self.keep_prob)
    return exec_statement

def create_exec_statement_train(opts):
    """
    Same as create_exec_statement_test but for multi
    gpu parsed training cycles.
    INPUTS:
    - opts: (object) command line arguments from argparser
    """
    exec_statement = "pred = "
    #pred =
    exec_statement += opts.network
    #pred = GoogLe
    exec_statement += "_Net(self.xTr, self.is_training, "
    #pred = GoogLe_Net(multi_inputs[i], self.is_training,
    exec_statement += str(opts.num_class)
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2
    exec_statement += ", "
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2,
    exec_statement += str(opts.batch_size / max(1,opts.num_gpu))
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2, 12
    exec_statement += ", self.keep_prob)"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 12, self.keep_prob)
    return exec_statement

def average_gradients(grads_multi):
    """
    Basically averages the aggregated gradients.
    Much was stolen from code from the Tensorflow team.
    Basically, look at the famous inceptionv3 code.
    INPUTS:
    - grads_multi: a list of gradients and variables
    """
    average_grads = []
    for grad_and_vars in zip(*grads_multi):
        grads = []
        for g,_ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        if grads == []:
            continue
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class classifier:
    def __init__(self, opts):
        """
        Initialization of all the fields.
        We also create the network.
        INPUTS:
        - opts: (object) command line arguments from argparser
        """
        self.opts = opts

        print(opts)
        # Creating the Placeholders.
        if self.opts.path_train:
            self.ds = DataStream(img_size=opts.image_size, batch_size=opts.batch_size,
                                 h5_path=opts.path_train, ap_only=opts.ap_only)
            self.matrix_size = opts.image_size
            self.num_channels = 1
        elif self.opts.path_test:
            self.matrix_size, self.num_channels = find_data_shape(self.opts.path_test)
        else:
            self.matrix_size, self.num_channels = 224,1
        each_bs  = self.opts.batch_size
        xTe_size = [None, self.matrix_size, self.matrix_size, self.num_channels]
        yTe_size = [None]
        xTr_size = [None, self.matrix_size, self.matrix_size, self.num_channels]
        yTr_size = [None]
        self.xTe = tf.placeholder(tf.float32, xTe_size)
        self.yTe = tf.placeholder(tf.int64, yTe_size)
        self.xTr = tf.placeholder(tf.float32, xTr_size)
        self.yTr = tf.placeholder(tf.int64, yTr_size)
        self.is_training = tf.placeholder_with_default(1, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        self.ap_only = opts.ap_only

        # Creating the Network for Testing
        exec_statement = create_exec_statement_test(opts)
        exec exec_statement
        self.L2_loss = get_L2_loss(self.opts.l2)
        self.L1_loss = get_L1_loss(self.opts.l1)
        self.ce_loss = get_ce_loss(self.pred, self.yTe)
        self.cost = self.ce_loss + self.L2_loss + self.L1_loss
        self.prob = tf.nn.softmax(self.pred)
        self.acc = get_accuracy(self.pred, self.yTe)
        self.n_epochs = opts.max_epoch
        self.print_every = opts.print_every
        self.n_batch_val = opts.n_batch_val

        # Listing the data.
        if self.opts.path_train:
            self.iter_count = self.ds.n_batches
            self.epoch_every = self.ds.n_batches
        else:
            self.iter_count, self.epoch_every, self.print_every = calculate_iters(1000, self.opts.max_epoch, self.opts.batch_size)
        if self.opts.path_test:
            list_imgs = listdir(self.opts.path_test)
            for name_img in list_imgs:
                if name_img[0] == '.':
                    list_imgs.remove(name_img)
            self.X_te = list_imgs
        optimizer,global_step = get_optimizer(self.opts.lr, self.opts.lr_decay, self.epoch_every)
        grads = optimizer.compute_gradients(self.cost)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)


        # Creating the Network for Training
        loss_multi = []
        grads_multi = []
        acc_multi = []
        tf.get_variable_scope().reuse_variables()

        for i in xrange(self.opts.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu%d' % i) as scope:
                    exec_statement = create_exec_statement_train(opts)
                    exec exec_statement
                    loss = get_ce_loss(pred, self.yTr)
                    loss_multi.append(loss)
                    cost = loss + self.L2_loss + self.L1_loss

                    grads_and_vars = optimizer.compute_gradients(cost)
                    grads_multi.append(grads_and_vars)

                    accuracy = get_accuracy(pred, self.yTr)
                    acc_multi.append(accuracy)
        if self.opts.num_gpu == 0:
            i = 0
            with tf.name_scope('cpu0') as scope:
                exec_statement = create_exec_statement_train(opts)
                exec exec_statement
                loss = get_ce_loss(pred, self.yTr)
                loss_multi.append(loss)
                cost = loss + self.L2_loss + self.L1_loss

                grads_and_vars = optimizer.compute_gradients(cost)
                grads_multi.append(grads_and_vars)

                accuracy = get_accuracy(pred, self.yTr)
                acc_multi.append(accuracy)
        grads = average_gradients(grads_multi)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)
        self.loss_multi = tf.add_n(loss_multi) / max(self.opts.num_gpu,1)
        self.acc_multi = tf.add_n(acc_multi) / max(self.opts.num_gpu,1)


        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=None)

        self.tr_acc = []
        self.tr_loss = []
        self.val_acc = []
        self.val_loss = []

        if self.opts.bool_display:
            self.f1 = plt.figure()
            self.plot_accuracy = self.f1.add_subplot(121)
            self.plot_loss = self.f1.add_subplot(122)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def average_accuracy(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        return np.mean(0.0 + (prediction == truth))
    
    def confusion_matrix(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        truth = truth.astype(np.int64)
        prediction = prediction.astype(np.int64)
        O = np.zeros((self.opts.num_class, self.opts.num_class))
        for i in range(len(truth)):
            O[truth[i], prediction[i]] += 1
        return O
    
    def quadratic_kappa(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        truth = truth.astype(np.int64)
        prediction = prediction.astype(np.int64)
        t_vec = np.zeros((self.opts.num_class))
        p_vec = np.zeros((self.opts.num_class))
        O = np.zeros((self.opts.num_class, self.opts.num_class))
        for i in range(len(truth)):
            O[truth[i], prediction[i]] += 1
            t_vec[truth[i]] += 1
            p_vec[prediction[i]] += 1
        W = np.zeros((self.opts.num_class, self.opts.num_class))
        for i in range(self.opts.num_class):
            for j in range(self.opts.num_class):
                W[i,j] = ((float(i) - j)**2) / ((self.opts.num_class - 1)**2)
        E = np.outer(t_vec, p_vec)
        E = E.astype(np.float32)
        O = O.astype(np.float32)
        W = W.astype(np.float32)
        E = np.sum(O) * E / np.sum(E)
        kappa = 1 - np.sum(W * O) / np.sum(W * E)
        return kappa
    
    def super_graph(self, save=True, name='0'):
        self.plot_accuracy.cla()
        self.plot_loss.cla()

        self.plot_accuracy.plot(self.tr_acc, 'b')
        if self.val_acc:
            self.plot_accuracy.plot(self.val_acc, 'r')
        self.plot_accuracy.set_ylim([0,1])
        self.plot_accuracy.set_xlabel('Epoch')
        self.plot_accuracy.set_ylabel('Accuracy')
        self.plot_accuracy.set_title('Accuracy')

        self.plot_loss.plot(self.tr_loss, 'b')
        if self.val_loss:
            self.plot_loss.plot(self.val_loss, 'r')
        ymax = 2 * np.log(self.opts.num_class)
        self.plot_loss.set_ylim([0, ymax])
        self.plot_loss.set_xlabel('Epoch')
        self.plot_loss.set_ylabel('-log(P(correct_class))')
        self.plot_loss.set_title('CrossEntropy Loss')
        
        if self.opts.path_visualization and save:
            path_save = join(self.opts.path_visualization, 'accuracy')
            if not isdir(path_save):
                mkdir(path_save)
            self.f1.savefig(join(path_save, name + '.png'))
        plt.pause(0.05)
        return 0

    def update_init(self):
        self.init = tf.global_variables_initializer()

    def super_print(self, statement):
        """
        This basically prints everything in statement.
        We'll print to stdout and path_log.
        """
        sys.stdout.write(statement + '\n')
        sys.stdout.flush()
        f = open(self.opts.path_log, 'a')
        f.write(statement + '\n')
        f.close()
        return 0

    def train_one_iter(self, i):
        """
        Basically trains one iteration.
        INPUTS:
        - self: (object)
        - i: (int) iteration
        """
        # Filling in the data.
        dataXX, dataYY = self.ds.next_batch()
        feed = {self.xTr: dataXX, self.is_training:1, self.yTr: dataYY, self.keep_prob:self.opts.keep_prob}
        _, loss_iter, acc_iter = self.sess.run((self.optimizer, self.loss_multi, self.acc_multi), feed_dict=feed)
        return loss_iter, acc_iter

    def inference_one_iter(self, path_file):
        """
        Does one forward pass and returns the segmentation.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """
        dataXX = np.zeros((1, self.matrix_size, self.matrix_size, self.num_channels))
        while(True):
            try:
                with h5py.File(path_file) as hf:
                    dataXX[0,:,:,:] = np.array(hf.get('data'))
                    break
            except:
                time.sleep(0.001)
        feed = {self.xTe:dataXX, self.is_training:0, self.keep_prob:1.0}
        prob = self.sess.run((self.prob), feed_dict=feed)
        prob = prob[0]
        return prob

    def test_one_iter(self, ds_test, name='0'):
        """
        Does one forward pass and returns the segmentation.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """
        dataXX, dataYY = ds_test.next_batch()
        feed = {self.xTe: dataXX, self.is_training:0, self.yTe: dataYY, self.keep_prob:1.0}
        loss, acc, pred = self.sess.run((self.ce_loss, self.acc, self.pred), feed_dict=feed)
        return loss, acc, pred, dataYY

    def test(self, path_X, n_batches=None):
        """
        Basically tests all the folders in path_X.
        INPUTS:
        - self: (object)
        - path_X: (str) file path to the data.
        - n_batches: (int/None) None tests all batches
        """
        # Initializing variables.
        ds_test = DataStream(img_size=self.opts.image_size, batch_size=self.opts.batch_size, h5_path=path_X, ap_only=self.ap_only)
        
        acc_te  = 0.0
        loss_te = 0.0
        preds = []
        truths = []
        counter = 0
        if n_batches is None:
            n_batches = ds_test.n_batches
        # Doing the testing.
        ds_test.prep_minibatches()
        for iter_data in range(n_batches):
            # Reading in the data.
            loss_iter_iter, acc_iter_iter,pred_iter_iter,truth_iter_iter = self.test_one_iter(ds_test)
            loss_te += loss_iter_iter
            acc_te += acc_iter_iter
            if counter == 0:
                preds = pred_iter_iter
                truths = truth_iter_iter
                counter += 1
            else:
                preds = np.concatenate((preds, pred_iter_iter), 0)
                truths = np.concatenate((truths, truth_iter_iter), 0)

        loss_te /= n_batches
        acc_te /= n_batches 
        
        return loss_te, acc_te, preds, truths
        
    
    def train_model(self):
        """
        Loads model and trains.
        """
        if not self.opts.path_train:
            return 0
        # Initializing
        start_time = time.time()
        loss_tr = 0.0
        acc_tr = 0.0
        if self.opts.bool_load:
            self.sess.run(self.init)
            self.saver.restore(self.sess, self.opts.path_model)
            #optimizer_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'optimizer')
            #print optimizer_scope
            #self.sess.run(tf.variables_initializer(optimizer_scope))
        else:
            self.sess.run(self.init)
        # Training
        self.super_print("Let's start the training!")
        loss_min = 1000000
        for epoch in range(self.n_epochs):
            self.ds.prep_minibatches()
            for iter in range(self.iter_count):
                loss_temp, acc_temp = self.train_one_iter(iter)
                loss_tr += loss_temp
                acc_tr += acc_temp
                if ((iter)%self.print_every) == 0 or iter == self.iter_count-1:
                    if iter != 0:
                        loss_tr /= self.print_every
                        acc_tr /= self.print_every
                    self.tr_loss.append(loss_tr)
                    self.tr_acc.append(acc_tr)
                    current_time = time.time()
                    statement = "\t"
                    statement += "Iter: " + str(iter) + " "
                    statement += "Time: " + str((current_time - start_time) / 60) + " "
                    statement += "Loss_tr: " + str(loss_tr) + " "
                    statement += "Acc_tr: " + str(acc_tr)
                    loss_tr = 0.0
                    acc_tr = 0.0
                    if self.opts.path_validation:
                        if iter == self.iter_count-1:
                            loss_val, acc_val,preds,truths = self.test(self.opts.path_validation)
                        else:
                            loss_val, acc_val,preds,truths = self.test(self.opts.path_validation, self.n_batch_val)
                        self.val_loss.append(loss_val)
                        self.val_acc.append(acc_val)
                        statement += " Loss_val: " + str(loss_val)
                        if self.opts.bool_kappa:
                            statement += " Kappa: " + str(self.quadratic_kappa(preds, truths))
                        if self.opts.bool_confusion:
                            print self.confusion_matrix(preds, truths)
                        if loss_val < loss_min:
                            loss_min = loss_val
                            self.saver.save(self.sess, self.opts.path_model)
                    if self.opts.bool_display:
                        self.super_graph()
                    self.super_print(statement)
            if (not self.opts.path_validation) and self.opts.path_model:
                self.saver.save(self.sess, self.opts.path_model)
                

    def test_model(self):
        """
        Loads model and test.
        """
        if not self.opts.path_test:
            return 0
        # Initializing
        start_time = time.time()
        loss_te = 0.0
        self.saver.restore(self.sess, self.opts.path_model)

    def do_inference(self):
        """
        Loads model and does inference.
        """
        if not self.opts.path_inference:
            return 0
        # Initializing
        start_time = time.time()
        loss_te = 0.0
        self.saver.restore(self.sess, self.opts.path_model)
        for name_folder in listdir(self.opts.path_inference):
            path_imgs = join(self.opts.path_inference, name_folder)
            for name_img in listdir(path_imgs):
                if name_img[0] == '.':
                    continue
                if name_img[-3:] != '.h5':
                    continue
                path_file = join(path_imgs, name_img)
                prob = self.inference_one_iter(path_file)
                h5f = h5py.File(path_file, 'a')
                h5f.create_dataset('label_pred', data=prob)
                h5f.close()
            
            
                
