
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
    if dataset_type in ['sid_dataset']:
        from data.SID_dataset import DatasetSID as D

    # -----------------------------------------
    # Patterned flash/no-flash reconstruction
    # -----------------------------------------
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset