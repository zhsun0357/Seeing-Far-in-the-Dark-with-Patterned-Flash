

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # -----------------------------------------
    # Patterned flash reconstruction
    # -----------------------------------------
    if dataset_type in ['pf_data']:
        from data.pf_data import DatasetPF as D
        
    # -----------------------------------------
    # Patterned flash/no-flash reconstruction
    # -----------------------------------------
    elif dataset_type in ['fnf_data']:
        from data.fnf_data import DatasetFnF as D
    
    # -----------------------------------------
    # common
    # -----------------------------------------
    elif dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D

    elif dataset_type in ['plainpatch']:
        from data.dataset_plainpatch import DatasetPlainPatch as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
