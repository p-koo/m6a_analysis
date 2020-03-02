import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import tensorflow as tf
import modisco



def load_data(file_path, struct='PU', reverse_compliment=True, ):

    # load dataset
    dataset = h5py.File(file_path, 'r')
    X_train = np.array(dataset['X_train']).astype(np.float32)
    Y_train = np.array(dataset['Y_train']).astype(np.float32)
    X_valid = np.array(dataset['X_valid']).astype(np.float32)
    Y_valid = np.array(dataset['Y_valid']).astype(np.float32)
    X_test = np.array(dataset['X_test']).astype(np.float32)
    Y_test = np.array(dataset['Y_test']).astype(np.float32)

    X_train_seq = X_train[:,:4,:]
    X_train_struct = X_train[:,4:,:]

    X_valid_seq = X_valid[:,:4,:]
    X_valid_struct = X_valid[:,4:,:]

    X_test_seq = X_test[:,:4,:] 
    X_test_struct = X_test[:,4:,:] 

    if reverse_compliment:
        X_train_rc = X_train_seq[:,::-1,:][:,:,::-1]
        X_valid_rc = X_valid_seq[:,::-1,:][:,:,::-1]
        X_test_rc = X_test_seq[:,::-1,:][:,:,::-1]
        
        X_train_seq = np.vstack([X_train_seq, X_train_rc])
        X_valid_seq = np.vstack([X_valid_seq, X_valid_rc])
        X_test_seq = np.vstack([X_test_seq, X_test_rc])
        
        Y_train = np.vstack([Y_train, Y_train])
        Y_valid = np.vstack([Y_valid, Y_valid])
        Y_test = np.vstack([Y_test, Y_test])

    if struct == 'seq':
        X_train = X_train_seq
        X_valid = X_valid_seq
        X_test = X_test_seq

    elif struct == 'pu':

        p = np.expand_dims(X_train_struct[:,0,:], axis=1)
        u = np.expand_dims(np.sum(X_train_struct[:,1:,:], axis=1), axis=1)
        X_train = np.concatenate([X_train_seq, p, u], axis=1)

        p = np.expand_dims(X_valid_struct[:,0,:], axis=1)
        u = np.expand_dims(np.sum(X_valid_struct[:,1:,:], axis=1), axis=1)
        X_valid = np.concatenate([X_valid_seq, p, u], axis=1)

        p = np.expand_dims(X_test_struct[:,0,:], axis=1)
        u = np.expand_dims(np.sum(X_test_struct[:,1:,:], axis=1), axis=1)
        X_test = np.concatenate([X_test_seq, p, u], axis=1)

    # add another dimension to make it a 4D tensor and transpose dimensions 
    X_train = np.expand_dims(X_train, axis=3).transpose([0, 2, 3, 1])
    X_test = np.expand_dims(X_test, axis=3).transpose([0, 2, 3, 1])
    X_valid = np.expand_dims(X_valid, axis=3).transpose([0, 2, 3, 1])

    train = {'inputs': X_train, 'targets': Y_train}
    valid = {'inputs': X_valid, 'targets': Y_valid}
    test = {'inputs': X_test, 'targets': Y_test}

    return train, valid, test



def import_model(model_name):

    # get model
    if model_name == 'cnn_5':
        from model_zoo import cnn_5 as genome_model
    elif model_name == 'cnn_5_exp':
        from model_zoo import cnn_5_exp as genome_model
    elif model_name == 'cnn_5_residual':
        from model_zoo import cnn_5_residual as genome_model
    elif model_name == 'cnn_5_residual_exp':
        from model_zoo import cnn_5_residual_exp as genome_model
    elif model_name == 'cnn_10':
        from model_zoo import cnn_10 as genome_model
    elif model_name == 'cnn_10_exp':
        from model_zoo import cnn_10_exp as genome_model
    elif model_name == 'cnn_10_residual':
        from model_zoo import cnn_10_residual as genome_model
    elif model_name == 'cnn_10_residual_exp':
        from model_zoo import cnn_10_residual_exp as genome_model
    elif model_name == 'cnn_25':
        from model_zoo import cnn_25 as genome_model
    elif model_name == 'cnn_25_residual':
        from model_zoo import cnn_25_residual as genome_model
    elif model_name == 'cnn_25_residual_exp':
        from model_zoo import cnn_25_residual_exp as genome_model
    elif model_name == 'cnn_deep':
        from model_zoo import cnn_deep as genome_model
    elif model_name == 'cnn_deep_exp':
        from model_zoo import cnn_deep_exp as genome_model
    elif model_name == 'residualbind':
        from model_zoo import residualbind as genome_model
    elif model_name == 'residualbind_exp':
        from model_zoo import residualbind_exp as genome_model

    return genome_model




def meme_generate(W, output_file='meme.txt', prefix='filter', factor=None):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j in range(len(W)):
        if factor:
            pwm = utils.normalize_pwm(W[j], factor=factor)
        else:
            pwm = W[j]
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (pwm.shape[1], pwm.shape[1]))
        for i in range(pwm.shape[1]):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[:,i]))
        f.write('\n')

    f.close()



def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir

    