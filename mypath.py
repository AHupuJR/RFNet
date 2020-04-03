class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return 'E:\\datasets\\cityscapes\\'      # folder that contains leftImg8bit/
        elif dataset == 'citylostfound':
            return 'E:\\datasets\\cityscapesandlostandfound\\'  # folder that mixes Cityscapes and Lost and Found
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
