import pandas as pd

def split_to_train_val_test(df, label_column, splits=(0.7, 0.2, 0.1), verbose=False):
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]

        lbl_train_df        = lbl_df.sample(frac=splits[0])
        lbl_val_and_test_df = lbl_df.drop(lbl_train_df.index)
        lbl_test_df         = lbl_val_and_test_df.sample(frac=splits[2]/(splits[1] + splits[2]))
        lbl_val_df          = lbl_val_and_test_df.drop(lbl_test_df.index)

        if verbose:
            print('\n{}:\n---------\ntotal:{}\ntrain_df:{}\nval_df:{}\ntest_df:{}'.format(lbl,
                                                                        len(lbl_df), 
                                                                        len(lbl_train_df), 
                                                                        len(lbl_val_df), 
                                                                        len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        val_df   = val_df.append(lbl_val_df)
        test_df  = test_df.append(lbl_test_df)

    # shuffle them on the way out using .sample(frac=1)
    return train_df.sample(frac=1), val_df.sample(frac=1), test_df.sample(frac=1)

def get_train_val_dataframes(BASE_DIR, classes, images_to_exclude, split_ratios):
    CLASSES_FILE = BASE_DIR + 'classes.txt'
    IMAGE_FILE   = BASE_DIR + 'images.txt'
    LABEL_FILE   = BASE_DIR + 'image_class_labels.txt'

    images_df = pd.read_csv(IMAGE_FILE, sep=' ',
                            names=['image_pretty_name', 'image_file_name'],
                            header=None)
    image_class_labels_df = pd.read_csv(LABEL_FILE, sep=' ',
                                names=['image_pretty_name', 'orig_class_id'], header=None)

    # Merge the metadata into a single flat dataframe for easier processing
    full_df = pd.DataFrame(images_df)
    full_df = full_df[~full_df.image_file_name.isin(images_to_exclude)]

    full_df.reset_index(inplace=True, drop=True)
    full_df = pd.merge(full_df, image_class_labels_df, on='image_pretty_name')

    # grab a small subset of species for testing
    criteria = full_df['orig_class_id'].isin(classes)
    full_df = full_df[criteria]
    print('Using {} images from {} classes'.format(full_df.shape[0], len(classes)))

    unique_classes = full_df['orig_class_id'].drop_duplicates()
    sorted_unique_classes = sorted(unique_classes)
    id_to_one_based = {}
    i = 1
    for c in sorted_unique_classes:
        id_to_one_based[c] = str(i)
        i += 1

    full_df['class_id'] = full_df['orig_class_id'].map(id_to_one_based)
    full_df.reset_index(inplace=True, drop=True)

    def get_class_name(fn):
        return fn.split('/')[0]
    full_df['class_name'] = full_df['image_file_name'].apply(get_class_name)
    full_df = full_df.drop(['image_pretty_name'], axis=1)

    train_df = []
    test_df  = []
    val_df   = []

    # split into training and validation sets
    train_df, val_df, test_df = split_to_train_val_test(full_df, 'class_id', split_ratios)

    print('num images total: ' + str(images_df.shape[0]))
    print('\nnum train: ' + str(train_df.shape[0]))
    print('num val: ' + str(val_df.shape[0]))
    print('num test: ' + str(test_df.shape[0]))
    return train_df, val_df, test_df