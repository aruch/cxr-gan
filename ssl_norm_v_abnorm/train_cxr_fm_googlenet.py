import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.updates as lupd
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import cxr_data
from sklearn.metrics import confusion_matrix
from data_stream import DataStream
import cPickle

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--unlabeled_weight', type=float, default=0.0001)
parser.add_argument('--ap_only', type=int, default=0)
parser.add_argument('--train_path', type=str, default='/scratch/users/aruch/nerdd/train_sub.hdf5')
parser.add_argument('--val_path', type=str, default='/scratch/users/aruch/nerdd/val_sub.hdf5')
parser.add_argument('--test_path', type=str, default='/scratch/users/aruch/nerdd/test_sub.hdf5')
parser.add_argument('--print_every', type=int, default=500)
parser.add_argument('--n_val', type=int, default=100)
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--n_labeled_images', type=int, default=20000)
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
ds_train = DataStream(img_size=args.image_size,
                      batch_size=args.batch_size, h5_path=args.train_path)

ds_train2 = DataStream(img_size=args.image_size,
                      batch_size=args.batch_size, h5_path=args.train_path)

labels = np.array(ds_train.f["label"], dtype="int32")
labels[labels==2] = 1
train_x_lab = []
train_y_lab = []
for lab in range(args.n_class):
    idxs = np.arange(ds_train.n_images)[labels==lab]
    idxs = np.random.permutation(idxs)[:args.n_labeled_images]
    idxs = np.sort(idxs)
    for i in idxs:
        train_x_lab.append(np.array(ds_train.images[i]))
        train_y_lab.append(lab)

train_x_lab = np.array(train_x_lab)
train_x_lab = np.expand_dims(train_x_lab, axis=1)
train_y_lab = np.array(train_y_lab, dtype=np.int32)
ds_test = DataStream(img_size=args.image_size,
                     batch_size=args.batch_size, h5_path=args.val_path)

print("DATA LOADERS CREATED")

# specify generative model
noise_dim = (args.batch_size, 100)
noise = theano_rng.uniform(size=noise_dim)
orig_gen_n = 1024
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*orig_gen_n, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,orig_gen_n,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,orig_gen_n/2,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,orig_gen_n/4,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,orig_gen_n/8,32,32), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 16 -> 32 
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,1,64,64), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 32 -> 64 
gen_dat = ll.get_output(gen_layers[-1])

print("GENERATOR CREATED")

def inceptionModule(input_layer, nfilters):
    inception_net = []
    inception_net.append(dnn.MaxPool2DDNNLayer(input_layer, pool_size=3, stride=1, pad=1)) #0
    inception_net.append(dnn.Conv2DDNNLayer(inception_net[-1], nfilters[0], 1, flip_filters=False)) #1
    
    inception_net.append(dnn.Conv2DDNNLayer(input_layer, nfilters[1], 1, flip_filters=False)) #2
    
    inception_net.append(dnn.Conv2DDNNLayer(input_layer, nfilters[2], 1, flip_filters=False)) #3
    inception_net.append(dnn.Conv2DDNNLayer(inception_net[-1], nfilters[3], 3, pad=1, flip_filters=False)) #4
    
    inception_net.append(dnn.Conv2DDNNLayer(input_layer, nfilters[4], 1, flip_filters=False)) #5
    inception_net.append(dnn.Conv2DDNNLayer(inception_net[-1], nfilters[5], 5, pad=2, flip_filters=False)) #6

    inception_net.append(ll.ConcatLayer([inception_net[2], inception_net[4], inception_net[6], inception_net[1]])) #7

    return inception_net

disc_layers = [ll.InputLayer(shape=(None, 1, args.image_size, args.image_size))]
disc_layers.append(dnn.Conv2DDNNLayer(disc_layers[-1], 64, 7, stride=2, pad=3, flip_filters=False))
disc_layers.append(ll.MaxPool2DLayer(disc_layers[-1], pool_size=3, stride=2, ignore_border=False))
disc_layers.append(ll.LocalResponseNormalization2DLayer(disc_layers[-1], alpha=0.00002, k=1))
disc_layers.append(dnn.Conv2DDNNLayer(disc_layers[-1], 64, 1, flip_filters=False))
disc_layers.append(dnn.Conv2DDNNLayer(disc_layers[-1], 192, 3, pad=1, flip_filters=False))
disc_layers.append(ll.LocalResponseNormalization2DLayer(disc_layers[-1], alpha=0.00002, k=1))
disc_layers.append(ll.MaxPool2DLayer(disc_layers[-1], pool_size=3, stride=2, ignore_border=False))
disc_layers.extend(inceptionModule(disc_layers[-1], [32, 64, 96, 128, 16, 32]))
disc_layers.extend(inceptionModule(disc_layers[-1], [64, 128, 128, 192, 32, 96]))
disc_layers.append(ll.MaxPool2DLayer(disc_layers[-1], pool_size=3, stride=2, ignore_border=False))
disc_layers.extend(inceptionModule(disc_layers[-1], [64, 192, 96, 208, 16, 48]))
disc_layers.extend(inceptionModule(disc_layers[-1], [64, 160, 112, 224, 24, 64]))
disc_layers.extend(inceptionModule(disc_layers[-1], [64, 128, 128, 256, 24, 64]))
disc_layers.extend(inceptionModule(disc_layers[-1], [64, 112, 144, 288, 32, 64]))
disc_layers.extend(inceptionModule(disc_layers[-1], [128, 256, 160, 320, 32, 128]))
disc_layers.append(ll.MaxPool2DLayer(disc_layers[-1], pool_size=3, stride=2, ignore_border=False))
disc_layers.extend(inceptionModule(disc_layers[-1], [128, 256, 160, 320, 32, 128]))
disc_layers.extend(inceptionModule(disc_layers[-1], [128, 384, 192, 384, 48, 128]))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(ll.DenseLayer(disc_layers[-1], num_units=args.n_class, nonlinearity=lasagne.nonlinearities.linear))
disc_layers.append(ll.NonlinearityLayer(disc_layers[-1], nonlinearity=lasagne.nonlinearities.softmax))
disc_params = ll.get_all_params(disc_layers, trainable=True)

print("DISCRIMINATOR CREATED")

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()
temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(disc_layers[-2], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-2], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)


l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_pred = T.argmax(output_before_softmax,axis=1)
test_err = T.mean(T.neq(test_pred,labels))

print("ERROR FUNCTIONS CREATED")

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = lupd.adam(loss_lab + args.unlabeled_weight*loss_unl, disc_params, learning_rate=lr, beta1=0.5).items()
disc_param_avg =  []
for p in disc_params:
    disc_param_avg.append(th.shared(np.cast[th.config.floatX](0.*p.get_value()), broadcastable=p.broadcastable))
disc_avg_updates = [(a,a+0.01*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates, on_unused_input='ignore') # data based initialization
#train_batch_disc = th.function(inputs=[x_lab,labels,lr], outputs=[loss_lab, train_err], updates=disc_param_updates+disc_avg_updates)
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
# Need to tweak avg update weight if we want to use givens
test_batch = th.function(inputs=[x_lab,labels], outputs=[test_err, test_pred], givens=disc_avg_givens)
samplefun = th.function(inputs=[],outputs=gen_dat)

print("DISC TRAINING FUNCTIONS CREATED")

# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-3], x_unl, deterministic=False)
output_gen = ll.get_output(disc_layers[-3], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(abs(m1-m2)) # feature matching loss
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = lupd.adam(loss_gen, gen_params, learning_rate=lr, beta1=0.5)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=None, updates=gen_param_updates)

print("GEN TRAINING FUNCTIONS CREATED")

# //////////// perform training //////////////
for epoch in range(10):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))
    print("Starting epoch {:} with learning rate {:.4f}".format(epoch, float(lr)))

    ds_train.prep_minibatches()
    ds_train2.prep_minibatches()

    # create minibatches of labeled data
    lab_batch_inds = []
    for t in range(ds_train.n_batches):
        inds = np.random.choice(train_x_lab.shape[0], args.batch_size, replace=False)
        lab_batch_inds.append(inds)

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(ds_train.n_batches):
        batchx = train_x_lab[lab_batch_inds[t]]
        batchy = train_y_lab[lab_batch_inds[t]]
        batchunl, _ = ds_train.next_batch()
        batchunl2, _ = ds_train2.next_batch()
        if t == 0 and epoch==0:
            init_param(batchx) # data based initialization

        #ll, te = train_batch_disc(batchx,batchy,lr)
        ll, lu, te = train_batch_disc(batchx,batchy,batchunl,lr)
        loss_lab += ll
        loss_unl += lu
        train_err += te

        train_batch_gen(batchunl2,lr)

        if t % args.print_every == 0:
            test_err = 0.
            ds_test.prep_minibatches()
            pred = np.array([])
            actual = np.array([])
            for b in range(args.n_val):
                batchx, batchy = ds_test.next_batch()
                batch_err, batch_pred = test_batch(batchx, batchy)
                test_err += batch_err
                pred = np.concatenate([pred, batch_pred]) 
                actual = np.concatenate([actual, batchy]) 

            test_err /= args.n_val
            print(time.time()-begin) 
            print(test_err)
            print(confusion_matrix(actual, pred))
            # generate samples from the model
            sample_x = samplefun()
            img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
            img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title='CXR Samples', rgb=False)
            plotting.plt.savefig("cxr_sample_feature_match{:}.png".format(epoch))

            print("Loss Lab: {:.4f}, Loss Unl: {:.4f}".format(loss_lab/t, loss_unl/t))

    loss_lab /= ds_train.n_batches
    loss_unl /= ds_train.n_batches
    train_err /= ds_train.n_batches

    # test
    test_err = 0.
    ds_test.prep_minibatches()
    pred = np.array([])
    actual = np.array([])
    for t in range(ds_test.n_batches):
        batchx, batchy = ds_test.next_batch()
        batch_err, batch_pred = test_batch(batchx, batchy)
        test_err += batch_err
        pred = np.concatenate([pred, batch_pred]) 
        actual = np.concatenate([actual, batchy]) 

    test_err /= ds_test.n_batches

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, train err = %.4f, test err avg. = %.4f" % (epoch, time.time()-begin, loss_lab, train_err, test_err))
    print(confusion_matrix(actual, pred))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CXR Samples', rgb=False)
    plotting.plt.savefig("cxr_sample_feature_match{:}.png".format(epoch))

    # save params
    np.savez('disc_params{:}.npz'.format(epoch), *[p.get_value() for p in disc_params])
    np.savez('gen_params{:}.npz'.format(epoch), *[p.get_value() for p in gen_params])