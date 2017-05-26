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
from data_stream import DataStream
import cPickle

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--data_dir', type=str, default='/scratch/users/aruch/nerdd')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
ds_train = DataStream(d=args.data_dir, img_size=args.image_size,
                      batch_size=args.batch_size, h5_name="train_sub.hdf5")
ds_test = DataStream(d=args.data_dir, img_size=args.image_size,
                     batch_size=args.batch_size, h5_name="val_sub.hdf5")
print("DATA LOADERS CREATED")

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 1, args.image_size, args.image_size))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 32, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=2, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

print("DISCRIMINATOR CREATED")

# costs
labels = T.ivector()
x_lab = T.tensor4()
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

print("ERROR FUNCTIONS CREATED")

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = lupd.adam(loss_lab, disc_params, learning_rate=lr, beta1=0.5).items()
disc_param_avg =  []
for p in disc_params:
    disc_param_avg.append(th.shared(np.cast[th.config.floatX](0.*p.get_value()), broadcastable=p.broadcastable))
disc_avg_updates = [(a,a+0.01*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates) # data based initialization
train_batch_disc = th.function(inputs=[x_lab,labels,lr], outputs=[loss_lab, train_err], updates=disc_param_updates+disc_avg_updates)
# Need to tweak avg update weight if we want to use givens
test_batch2 = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err)

print("TRAINING FUNCTIONS CREATED")

# //////////// perform training //////////////
for epoch in range(15):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))

    ds_train.prep_minibatches()

    # train
    loss_lab = 0.
    train_err = 0.
    for t in range(ds_train.n_batches):
        batchx, batchy = ds_train.next_batch()
        if t == 0 and epoch==0:
            init_param(batchx) # data based initialization

        ll, te = train_batch_disc(batchx,batchy,lr)
        loss_lab += ll
        train_err += te

        # if t % 100 == 0:
        #     test_err = 0. 
        #     nnn = 10
        #     for i in range(nnn):
        #         test_err += test_batch(testx[i*args.batch_size:(i+1)*args.batch_size],testy[i*args.batch_size:(i+1)*args.batch_size])
        #     test_err /= nnn
        #     print("%.4f" % (test_err,))
        

    loss_lab /= ds_train.n_batches
    train_err /= ds_train.n_batches

    # test
    test_err = 0.
    test_err2 = 0.
    ds_test.prep_minibatches()
    for t in range(ds_test.n_batches):
        batchx, batchy = ds_test.next_batch()
        test_err += test_batch(batchx,batchy)
        test_err2 += test_batch(batchx,batchy)

    test_err /= ds_test.n_batches
    test_err2 /= ds_test.n_batches

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, train err = %.4f, test err = %.4f, test err avg. = %.4f" % (epoch, time.time()-begin, loss_lab, train_err, test_err, test_err2))
    sys.stdout.flush()

    # save params
    np.savez('disc_params_{:}.npz'.format(epoch), *[p.get_value() for p in disc_params])
