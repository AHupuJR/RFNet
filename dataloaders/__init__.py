from dataloaders.datasets import cityscapes, citylostfound
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'citylostfound':
        if args.depth:
            train_set = citylostfound.CitylostfoundSegmentation(args, split='train')
            val_set = citylostfound.CitylostfoundSegmentation(args, split='val')
            test_set = citylostfound.CitylostfoundSegmentation(args, split='test')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        else:
            train_set = citylostfound.CitylostfoundSegmentation_rgb(args, split='train')
            val_set = citylostfound.CitylostfoundSegmentation_rgb(args, split='val')
            test_set = citylostfound.CitylostfoundSegmentation_rgb(args, split='test')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

