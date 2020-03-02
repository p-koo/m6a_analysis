from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np

import wrangler

# Dependencies: Biopython, Meme suite, bedtools
#-----------------------------------------------------------------------------------------------

windows = [400]
tf_names = ['wt_b', 'c1c2_b', 'wt_ht', 'c1c2_zt2l', 'ect2_zt14d', 'wt_amb',
            'wt_shade', 'wt_zt14d', 'c2_zt2l', 'c1c2_zt14d']
neg_names = ['wt_B', 'c1c2_B', 'wt_ht', 'c1c2_ZT2L', 'ect2_ZT14D', 'wt_amb',
            'wt_shade', 'wt_ZT14D', 'c2_ZT2L', 'c1c2_ZT14D']

for window in windows:
    print(window)
    max_length = window
    for i, tf_name in enumerate(tf_names):

        print(tf_name)

        # path to bed file
        data_path = '../data'
        pos_path = os.path.join(data_path, 'exomepeak', 'input_bed_peaks', 'individual_samples', tf_names[i]+'.peak.bed')
        neg_path = os.path.join(data_path, 'piranha', 'on_IP_minus_peaks', 'minp_'+ neg_names[i]+'_minus_IP.bed_piranha')

        # set genome path
        genome_path = os.path.join(data_path, 'genome', 'tair10.fa')

        #-----------------------------------------------------------------------------------------------

        # create new bed file with window enforced
        pos_bed_path = os.path.join(data_path, tf_name + '_pos.bed')
        wrangler.bedtools.enforce_constant_size(pos_path, pos_bed_path, window)#, compression='gzip')

        # extract sequences from bed file and save to fasta file
        pos_fasta_path = os.path.join(data_path, tf_name + '_pos.fa')
        wrangler.bedtools.to_fasta(pos_bed_path, pos_fasta_path, genome_path)

        # parse sequence and chromosome from fasta file
        sequences = wrangler.fasta.parse_sequences(pos_fasta_path)

        # filter sequences with absent nucleotides
        pos_sequences, _ = wrangler.munge.filter_nonsense_sequences(sequences)

        # save to fasta file
        wrangler.fasta.generate_fasta(pos_sequences, pos_fasta_path)
        
        # convert filtered sequences to one-hot representation
        pos_one_hot = wrangler.munge.convert_one_hot(pos_sequences, max_length=window)

        # generate secondary structure profiles with rnaplfold
        profile_path = os.path.join(data_path, tf_name+'_rnaplfold_pos_')
        pos_structure = wrangler.structure.RNAplfold_profile(pos_fasta_path, profile_path, window=max_length)

        # merge sequences and structural profiles
        pos_data = np.concatenate([pos_one_hot, pos_structure], axis=1)

        # create new bed file with window enforced
        neg_bed_path = os.path.join(data_path, tf_name + '_neg.bed')
        wrangler.bedtools.enforce_constant_size(neg_path, neg_bed_path, window)#, compression='gzip')

        # extract sequences from bed file and save to fasta file
        neg_fasta_path = os.path.join(data_path, tf_name + '_neg.fa')
        wrangler.bedtools.to_fasta(neg_bed_path, neg_fasta_path, genome_path)

        # parse sequence and chromosome from fasta file
        sequences = wrangler.fasta.parse_sequences(neg_fasta_path)

        # filter sequences with absent nucleotides
        neg_sequences, _ = wrangler.munge.filter_nonsense_sequences(sequences)


        # convert filtered sequences to one-hot representation
        neg_one_hot = wrangler.munge.convert_one_hot(neg_sequences, max_length=window)

        # nucleotide frequency matched background
        seq_pos = np.squeeze(np.argmax(pos_one_hot, axis=1))
        seq_neg = np.squeeze(np.argmax(neg_one_hot, axis=1))

        f_pos = []
        for s in seq_pos:
            f_pos.append([np.sum(s==0), np.sum(s==1), np.sum(s==2), np.sum(s==3)]) 
        f_pos = np.array(f_pos)/200

        f_neg = []
        for s in seq_neg:
            f_neg.append([np.sum(s==0), np.sum(s==1), np.sum(s==2), np.sum(s==3)]) 
        f_neg = np.array(f_neg)/200

        neg_index = list(range(len(f_neg)))

        match_index = []
        for i, f in enumerate(f_pos):

            dist = np.sum((f - f_neg[neg_index])**2, axis=1)
            index = np.argsort(dist)[0]
            match_index.append(neg_index[index])
            neg_index.pop(index)

        neg_one_hot = neg_one_hot[match_index]
        neg_sequences = neg_sequences[match_index]

        # save to fasta file
        wrangler.fasta.generate_fasta(neg_sequences, neg_fasta_path)

        # generate secondary structure profiles with rnaplfold
        profile_path = os.path.join(data_path,tf_name+'_rnaplfold_neg_')
        neg_structure = wrangler.structure.RNAplfold_profile(neg_fasta_path, profile_path, window=max_length)

        # merge sequences and structural profiles
        neg_data = np.concatenate([neg_one_hot, neg_structure], axis=1)

        # merge postive and negative sequences
        one_hot = np.vstack([pos_data, neg_data])
        labels = np.vstack([np.ones((len(pos_data), 1)), np.zeros((len(neg_data), 1))])

        # split dataset into training set, cross-validation set, and test set
        train, valid, test, _ = wrangler.munge.split_dataset(one_hot, labels, valid_frac=0.1, test_frac=0.2)

        # save dataset to hdf5 file
        print('saving dataset')
        with h5py.File(os.path.join(data_path, tf_name+'_'+str(window)+'_struct_match.h5'), 'w') as fout:
            X_train = fout.create_dataset('X_train', data=train[0], dtype='float32', compression="gzip")
            Y_train = fout.create_dataset('Y_train', data=train[1], dtype='int8', compression="gzip")
            X_valid = fout.create_dataset('X_valid', data=valid[0], dtype='float32', compression="gzip")
            Y_valid = fout.create_dataset('Y_valid', data=valid[1], dtype='int8', compression="gzip")
            X_test = fout.create_dataset('X_test', data=test[0], dtype='float32', compression="gzip")
            Y_test = fout.create_dataset('Y_test', data=test[1], dtype='int8', compression="gzip")
