class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return 'E:\\datasets\\cityscapes\\'      # foler that contains leftImg8bit/
        elif dataset == 'citylostfound':
            return 'E:\\datasets\\cityscapes\\'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
