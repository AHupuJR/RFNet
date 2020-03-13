import os
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders import custom_transforms_rgb as tr_rgb

class CitylostfoundSegmentation(data.Dataset):
    NUM_CLASSES = 20  # small_obstacle index: 19

    def __init__(self, args, root=Path.db_root_dir('citylostfound'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.disparities = {}
        self.labels = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.disparities_base = os.path.join(self.root,'disparity',self.split)  # 增加了直方图均衡化
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        # 挑选特定图片inference
        self.val_list = self.args.val_image_names
        print('inference 特定images的val_list属性')
        print(type(self.val_list))
        print(self.val_list)

        #0214inference特定图片
        if self.split =='train' or self.split=='test':
            self.images[split] = self.recursive_glob(rootdir=self.images_base, suffix= '.png')
            self.images[split].sort()

            self.disparities[split] = self.recursive_glob(rootdir=self.disparities_base, suffix= '.png')
            self.disparities[split].sort()

            self.labels[split] = self.recursive_glob(rootdir=self.annotations_base,
                                                     suffix='labelTrainIds.png')
            self.labels[split].sort()

        else:

            if self.val_list is not None:

                self.images[split] = self.recursive_glob(rootdir=self.images_base, suffix=self.val_list)#'.png')
                self.images[split].sort()

                self.disparities[split] = self.recursive_glob(rootdir=self.disparities_base, suffix=self.val_list)#'.png')
                self.disparities[split].sort()

                self.labels[split] = self.recursive_glob(rootdir=self.annotations_base, suffix=self.val_list)#'labelTrainIds.png')
                self.labels[split].sort()

            else:
                print('else: val_list is None')
                self.images[split] = self.recursive_glob(rootdir=self.images_base, suffix= '.png')
                self.images[split].sort()

                self.disparities[split] = self.recursive_glob(rootdir=self.disparities_base, suffix= '.png')
                self.disparities[split].sort()

                self.labels[split] = self.recursive_glob(rootdir=self.annotations_base,
                                                         suffix='labelTrainIds.png')
                self.labels[split].sort()

        self.ignore_index = 255

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.images_base))
        if not self.disparities[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.disparities_base))


        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s disparity images" % (len(self.disparities[split]), split))


    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        disp_path = self.disparities[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()

        _img = Image.open(img_path).convert('RGB')

        # 转化成深度图
        # _depth = self.convert_dips_to_depths_cityscapes(disp_path)
        # _depth = Image.fromarray(_depth)
        _depth = Image.open(disp_path)
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)

        # relabel lost and found labels
        # _tmp = tr.Relabel(255, self.ignore_index)(_tmp)
        if self.split == 'train':
            if index < 1036:  # lostandfound
                _tmp = self.relabel_lostandfound(_tmp)
            else:  # cityscapes
                pass
        elif self.split == 'val':
            if index < 1203:  # lostandfound
                _tmp = self.relabel_lostandfound(_tmp)
            else:  # cityscapes
                pass
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'depth': _depth, 'label': _target}

        # data augment
        if self.split == 'train':
            return self.transform_tr(sample)  # Image object
        elif self.split == 'val':
            return self.transform_val(sample), img_path
        elif self.split == 'test':
            return self.transform_ts(sample)

    def convert_dips_to_depths_cityscapes(self, img_disp, baseline = 0.209313, focal_length = 2262.52):
        img_d = cv2.imread(img_disp, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256
        # set baseline and depth
        depth = (baseline * focal_length) / img_d
        # set depth range
        max_depth = 100
        min_depth = 1e-3
        depth[depth > max_depth] = max_depth
        depth[depth < min_depth] = min_depth
        depth = depth.astype(np.uint8)
        depth = Image.fromarray(depth)

        return depth


    def relabel_lostandfound(self, input):
        input = tr.Relabel(0, self.ignore_index)(input)  # background->255 ignore
        input = tr.Relabel(1, 0)(input)  # road 1->0
        # input = Relabel(255, 20)(input)  # unlabel 20
        input = tr.Relabel(2, 19)(input)  # obstacle  19
        return input

    # def recursive_glob(self, rootdir='.', suffix=''):
    #     """Performs recursive glob with given suffix and rootdir
    #         :param rootdir is the root directory
    #         :param suffix is the suffix to be searched
    #     """
    #     return [os.path.join(looproot, filename)
    #             for looproot, _, filenames in os.walk(rootdir)
    #             for filename in filenames if filename.endswith(suffix)]

    def recursive_glob(self, rootdir='.', suffix=None):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        if isinstance(suffix, str):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for filename in filenames if filename.endswith(suffix)]
        elif isinstance(suffix, list):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for x in suffix for filename in filenames if filename.startswith(x)]


    def transform_tr(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),  # 原图大小val
            tr.CropBlackArea(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            # tr.CropBlackArea(),
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


class CitylostfoundSegmentation_rgb(data.Dataset):
    NUM_CLASSES = 19  # small_obstacle index: 19    2.20用完改回20

    def __init__(self, args, root=Path.db_root_dir('citylostfound'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.labels = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        self.files[split].sort()

        self.labels[split] = self.recursive_glob(rootdir=self.annotations_base, suffix='labelTrainIds.png')
        self.labels[split].sort()

        self.ignore_index = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()
        # lbl_path = os.path.join(self.annotations_base,
        #                         img_path.split(os.sep)[-2],
        #                         os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')

        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)

        # relabel lost and found labels
        # _tmp = tr.Relabel(255, self.ignore_index)(_tmp)
        if self.split == 'train':
            if index < 1036:  # lostandfound
                _tmp = self.relabel_lostandfound(_tmp)
            else:  # cityscapes
                pass
        elif self.split == 'val':
            if index < 1203:  # lostandfound
                _tmp = self.relabel_lostandfound(_tmp)
            else:  # cityscapes
                pass
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)  # Image object
        elif self.split == 'val':
            return self.transform_val(sample), img_path
        elif self.split == 'test':
            return self.transform_ts(sample)


    def relabel_lostandfound(self, input):
        input = tr.Relabel(0, self.ignore_index)(input)  # background->255 ignore
        input = tr.Relabel(1, 0)(input)  # road 1->0
        # input = Relabel(255, 20)(input)  # unlabel 20
        input = tr.Relabel(2, 19)(input)  # obstacle  19
        return input

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr_rgb.CropBlackArea(),
            tr_rgb.RandomHorizontalFlip(),
            tr_rgb.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            # tr.RandomGaussianBlur(),
            tr_rgb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_rgb.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr_rgb.CropBlackArea(),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),  # 原图大小val
            tr_rgb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_rgb.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr_rgb.FixedResize(size=self.args.crop_size),
            tr_rgb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_rgb.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CitylostfoundSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

