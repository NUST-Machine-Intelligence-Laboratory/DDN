import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from dataloader import CUB200Bags
from ddnnet import DDNNet
import argparse


parser = argparse.ArgumentParser(description='Deep Denoising Network')
parser.add_argument('--data_base', dest='data_base', type=str, required=True,
                    help='Data Base Folder Name')
args = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed(1)

start = time.time()
print('Init Model')
model = DDNNet(pretrained=False)
# Move the net into cuda
if torch.cuda.device_count() >= 1:
    model = torch.nn.DataParallel(model).cuda()
    use_cuda = True
else:
    use_cuda = False

print('Load Model State Dict')
model.load_state_dict(torch.load('model/vgg_16_epoch_best.pth'))

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

data_base = args.data_base
test_loader = DataLoader(CUB200Bags(train=False), batch_size=1, shuffle=False, **loader_kwargs)


def test():
    model.eval()
    test_instance_accuracy = 0.
    instance_num = 0

    noise_paths = set()
    with torch.no_grad():
        for batch_idx, (data, label, path) in enumerate(test_loader):
            instance_labels = label[1].numpy()[0]
            paths = np.array(path)
            if use_cuda:
                data = data.cuda()
            # calculate loss and metrics
            Y_prob, predicted_label, attention_weights = model(data)

            # instance level accuracy
            instance_prob = np.round(attention_weights.cpu().data.numpy()[0], decimals=3)
            instance_pred_label = (instance_prob > 0.5).astype(int)
            test_instance_accuracy += (instance_labels == instance_pred_label).astype(int).sum()
            instance_num += instance_labels.size

            # save noise instances' file names into txt
            noise = paths[instance_pred_label == 1]
            if noise.size != 0:
                for ind in range(noise.size):
                    noise_paths.add(noise[ind][0])

        test_instance_accuracy /= instance_num
        print('Test Instance Accuracy : {:.4f}'.format(test_instance_accuracy))
    with open('noise-list.txt', 'w') as f:
        for item in noise_paths:
            f.writelines(item.replace(data_base+'/val/0', 'data/train') + '\n')  # item type is numpy.str_


if __name__ == "__main__":
    print('Start Extracting')
    test()
    end = time.time()
    print('------ Total Runtime {} ------'.format(end-start))
