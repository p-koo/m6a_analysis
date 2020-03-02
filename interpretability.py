import numpy as np
import logomaker
import modisco


def generate_mutagenesis(X):

    L,A = X.shape 

    X_mut = []
    for l in range(L):
        for a in range(A):
            X_new = np.copy(X)
            X_new[l,:] = 0
            X_new[l,a] = 1
            X_mut.append(X_new)
            
    return np.array(X_mut)


def mutagenesis(sess, X, nntrainer, wt_correct=True, class_index=0, **kwargs):

    L, A = X.shape

    # generate first-order mutagenesis sequences for sequence X 
    X_mut = np.expand_dims(generate_mutagenesis(X), axis=2)

    if 'layer' in kwargs:
        layer = kwargs['layer']
    else:
        layer = 'output'


    # get predictions of mutagenized sequences
    predictions = nntrainer.get_activations(sess, {'inputs': X_mut}, layer=layer)[:,class_index]

    # reshape mutagenesis predictiosn
    mut_score = np.zeros((L,A))
    k = 0
    for l in range(L):
        for a in range(A):
            mut_score[l,a] = predictions[k]
            k += 1
            
    # correct w.r.t. wild type score
    if wt_correct:
        X = np.expand_dims(np.expand_dims(X, axis=0), axis=2)
        wt_score = nntrainer.get_activations(sess, {'inputs': X}, layer=layer)[:,class_index]
        mut_score -= wt_score[0]    
        
    return mut_score
            

def mutagenesis_analysis(sess, X, nntrainer, wt_correct=True, class_index=0, **kwargs):
    
    N,L,_,A = X.shape
    
    mut_scores = []
    for i in range(N):
        mut_scores.append(mutagenesis(sess, np.squeeze(X[i]), nntrainer, wt_correct, class_index, **kwargs))

    return np.array(mut_scores)
    
    
def saliency_analysis(sess, X, nntrainer, class_index=None, batch_size=512, **kwargs):

    if 'layer' in kwargs:
        layer = kwargs['layer']
    else:
        layer = list(nntrainer.network.keys())[-2]
    
    saliency = nntrainer.get_saliency(sess, 
                                      X, 
                                      nntrainer.network[layer], 
                                      class_index=class_index, 
                                      batch_size=batch_size)
    return saliency
        

    
    
def smoothgrad_analysis(sess, X, nntrainer, class_index=None, batch_size=512, **kwargs):

    if 'layer' in kwargs:
        layer = kwargs['layer']
    else:
        layer = list(nntrainer.network.keys())[-2]

    if 'num_average' in kwargs:
        num_average = kwargs['num_average']
    else:
        num_average = 100

    if 'scale' in kwargs:
        scale = kwargs['scale']
    else:
        scale = 0.2

    shape = list(X.shape)
    shape[0] = num_average

    saliency = np.zeros(X.shape)
    for i, x in enumerate(X):
        x = np.expand_dims(x, axis=0)            
        noisy_saliency = nntrainer.get_saliency(sess, 
                                                x+np.random.normal(scale=scale, size=shape), 
                                                nntrainer.network[layer], 
                                                class_index=class_index, 
                                                batch_size=batch_size)
        saliency[i,:,:,:] = np.mean(noisy_saliency, axis=0)

    return saliency
        


    
def attribution_analysis(sess, X, nntrainer, method='mutagenesis', class_index=0, **kwargs):
    
    N,L,_,A = X.shape
       

    if method == 'mutagenesis':
        attr_score = mutagenesis_analysis(sess, 
                                          X, 
                                          nntrainer, 
                                          wt_correct=True, 
                                          class_index=class_index,
                                          **kwargs)
        
    elif method == 'saliency':        
        attr_score = saliency_analysis(sess, 
                                       X, 
                                       nntrainer, 
                                       class_index=class_index, 
                                       batch_size=512,
                                       **kwargs)
        attr_score = np.squeeze(attr_score)
        
    elif method == 'smoothgrad':
        attr_score = smoothgrad_analysis(sess, 
                                         X, 
                                         nntrainer, 
                                         class_index=class_index, 
                                         batch_size=512,
                                         **kwargs)
        attr_score = np.squeeze(attr_score)

    elif method == 'grad_times_input':        
        attr_score = saliency_analysis(sess, 
                                       X, 
                                       nntrainer, 
                                       class_index=class_index, 
                                       batch_size=512, 
                                       **kwargs)
        attr_score *= X
        attr_score = np.squeeze(attr_score)
        
        
    return attr_score


def l2_norm(attr_score, axis=2):
    attr_score = np.sqrt(np.sum(attr_score**2, axis=axis))
    return attr_score

    

def preprocess_modisco(X, attr_score, task_idx='task0'):
    from collections import OrderedDict

    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()
    
    onehot_data = np.squeeze(X)
    
    task_to_hyp_scores[task_idx] = attr_score
    task_to_scores[task_idx] = onehot_data*np.sum(attr_score*onehot_data, axis=2, keepdims=True)

    task_to_hyp_scores[task_idx] = task_to_hyp_scores[task_idx]-np.mean(task_to_hyp_scores[task_idx],axis=-1)[:,:,None]

    return task_to_scores, task_to_hyp_scores, onehot_data



def modisco_analysis(X, attr_score, task='task0'):

    task_to_scores, task_to_hyp_scores, onehot_data = preprocess_modisco(X, 
                                                                         attr_score, 
                                                                         task_idx=task)

    results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        sliding_window_size=15,
                        flank_size=5,
                        target_seqlet_fdr=0.15,
                        seqlets_to_patterns_factory=
                            modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                                trim_to_window_size=15,
                                initial_flank_to_add=5,
                                kmer_len=5, num_gaps=1,
                                num_mismatches=0,
                                final_min_cluster_size=30)
                        )(
                        task_names=[task],
                        contrib_scores=task_to_scores,
                        hypothetical_contribs=task_to_hyp_scores,
                        one_hot=onehot_data)

    return results, (task_to_scores, task_to_hyp_scores)


    
def generate_logomaker_df(X, mut_score, alphabet='ACGT'):


    # convert one hot to sequence
    X_index = np.argmax(np.squeeze(X), axis=1)
    seq = ''
    for i in range(X.shape[0]):
        seq += alphabet[X_index[i]]

    # create saliency matrix
    saliency_mat_df = logomaker.saliency_to_matrix(seq=seq, 
                                            values=mut_score)
    return saliency_mat_df


def plot_filters(sess, test, layer, window=19, threshold=0.5, num_rows=10):
    
    # get 1st convolution layer filters
    fmap = nntrainer.get_activations(sess, test, layer=layer)
    W = visualize.activation_pwm(fmap, 
                                 X=test['inputs'], 
                                 threshold=threshold, 
                                 window=window)

    # plot 1st convolution layer filters
    fig = visualize.plot_filter_logos(W, 
                                      nt_width=50, 
                                      height=100, 
                                      norm_factor=None, 
                                      num_rows=num_rows)
    return fig
