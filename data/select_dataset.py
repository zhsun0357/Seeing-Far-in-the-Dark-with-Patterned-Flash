

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
    elif dataset_type in ['pattern_fnf_randp_low_poiss_quant_rggb_scale']:
        from data.dataset_pattern_fnf_randp_low_poiss_quant_rggb_scale import DatasetFnF as D
    elif dataset_type in ['pattern_fnf_randp_low_poiss_quant_rggb_scale_1']:
        from data.dataset_pattern_fnf_randp_low_poiss_quant_rggb_scale_1 import DatasetFnF as D
    elif dataset_type in ['pattern_fnf_randp_low_poiss_quant_rggb_recfnf_scale']:
        from data.dataset_pattern_fnf_randp_low_poiss_quant_rggb_recfnf_scale import DatasetFnF as D
    elif dataset_type in ['pattern_fnf_randp_low_poiss_quant_rggb_dloss_scale']:
        from data.dataset_pattern_fnf_randp_low_poiss_quant_rggb_dloss_scale import DatasetFnF as D
    elif dataset_type in ['pattern_fnf_randp_low_poiss_quant_rggb_dloss_scale_1']:
        from data.dataset_pattern_fnf_randp_low_poiss_quant_rggb_dloss_scale_1 import DatasetFnF as D
    elif dataset_type in ['pattern_nof_randp_low_poiss_quant_rggb_scale_ft3d']:
        from data.dataset_pattern_nof_randp_low_poiss_quant_rggb_scale_ft3d import DatasetFnF as D

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
