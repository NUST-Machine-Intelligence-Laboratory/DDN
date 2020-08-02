import numpy as np
from torch.utils.data import DataLoader
from dataloader import CUB200Bags

train_loader = DataLoader(CUB200Bags(train=True), batch_size=1, shuffle=True)

len_bag_list = []
cub200_bags_positive = 0
for batch_idx, (bag, label, _) in enumerate(train_loader):
    len_bag_list.append(int(bag.squeeze(0).size()[0]))
    cub200_bags_positive += label[0].item()
    # print('Bag {}: bag size is {}, label is {}'.format(batch_idx+1, int(bag.squeeze(0).size()[0]), label[0].item()))

print('Number of positive train bags : {}/{}\nNumber of instances per bag, mean: {}, max: {}, min: {}\n'.format(
    cub200_bags_positive, len(train_loader), np.mean(len_bag_list), np.max(len_bag_list), np.min(len_bag_list)))

