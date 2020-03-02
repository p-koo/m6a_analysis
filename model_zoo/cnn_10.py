
def model(input_shape, output_shape):
    
    layer1 = {'layer': 'input', #200          
              'input_shape': input_shape
             }
    layer2 = {'layer': 'conv1d',        
              'num_filters': 30,
              'filter_size': 21,
              'norm': 'batch',          
              'activation': 'relu',    
              'dropout': 0.1,           
              'padding': 'SAME',        
              'max_pool': 10,            
             }
    layer3 = {'layer': 'conv1d',        
              'num_filters': 64,
              'filter_size': 5,
              'norm': 'batch',          
              'activation': 'relu',     
              'dropout': 0.2,           
              'padding': 'SAME',        
              'max_pool': 10,            
             }
    layer5 = {'layer': 'dense',
              'num_units': 128,
              'norm': 'batch',
              'activation': 'relu',
              'dropout': 0.5,
             }
    layer6 = {'layer': 'dense',
              'num_units': output_shape[1],
              'activation': 'sigmoid'
             }

    model_layers = [layer1, layer2, layer3, layer5, layer6]

    # optimization parameters
    optimization = {"objective": "binary",     
                    "optimizer": "adam",       
                    "learning_rate": 0.0003,  
                    "l2": 1e-6,
                    #"label_smoothing": 0.1,
                    "l1": 1e-6,
                    }
    return model_layers, optimization
