import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import CUB200Bags
from ddnnet import DDNNet
import argparse
from focalloss import FocalLoss


torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser(description='Deep Denoising Network')
parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                    help='Base learning rate for training')
parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                    help='Epochs for training')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, required=True,
                    help='Weight decay')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.25)
parser.add_argument('--gamma', dest='gamma', type=float, default=2.0)
parser.add_argument('--lambda1', dest='lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', dest='lambda2', type=float, default=3.0)
args = parser.parse_args()

if args.base_lr <= 0:
    raise AttributeError('--base_lr parameter must > 0')
if args.epochs < 0:
    raise AttributeError('--epochs parameter must >= 0')
if args.weight_decay <= 0:
    raise AttributeError('--weight_decay parameter must > 0')

start = time.time()
print('Init Model')
model = DDNNet(pretrained=True)
# Move the net into cuda
if torch.cuda.device_count() >= 1:
    model = torch.nn.DataParallel(model).cuda()
    use_cuda = True
else:
    use_cuda = False
    # raise EnvironmentError('This is a GPU version')

optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                 patience=3, verbose=True, threshold=1e-4)
# criterion = torch.nn.CrossEntropyLoss().cuda()
criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha).cuda()
lambda1 = args.lambda1
lambda2 = args.lambda2

print('alpha = {}, gamma = {}, lambda1 = {}, lambda2 = {}'.format(args.alpha, args.gamma, lambda1, lambda2))

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(CUB200Bags(train=True), batch_size=1, shuffle=True, **loader_kwargs)
test_loader = DataLoader(CUB200Bags(train=False), batch_size=1, shuffle=False, **loader_kwargs)


def train():
    best_f1_score, best_instance_f1_score = 0., 0.
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_start = time.time()

        model.train()
        train_loss = 0.
        train_error = 0.
        bag_true_positive, bag_true_negative, bag_false_positive, bag_false_negative = 0., 0., 0., 0.
        true_positive, true_negative, false_positive, false_negative = 0., 0., 0., 0.
        for batch_idx, (data, label, _) in enumerate(train_loader):
            bag_label = label[0]
            instance_labels = label[1].numpy()[0]
            if use_cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
                target = label[1].cuda()  # (1 * N)

            # reset gradients
            optimizer.zero_grad()
            # net forward
            Y_prob, predicted_label, attention_weights = model(data)
            # calculate loss
            # bag level loss : negative log bernoulli
            bag_label = bag_label.float()
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
            bag_loss = -1. * (bag_label * torch.log(Y_prob) + (1. - bag_label) * torch.log(1. - Y_prob))
            # instance level loss
            pred = torch.transpose(attention_weights, 0, 1)  # (N * 1)
            pred = torch.cat((1 - pred, pred), dim=1)  # (N, 2)
            target = target.squeeze()  # (N)
            instance_loss = criterion(pred, target)
            loss = lambda1 * bag_loss + lambda2 * instance_loss
            # calculate metrics
            error = 1. - predicted_label.eq(bag_label).cpu().float().mean().item()
            train_loss += loss.data[0]
            train_error += error

            bag_true_positive += ((bag_label == 1) & (predicted_label == 1)).sum().item()
            bag_true_negative += ((bag_label == 0) & (predicted_label == 0)).sum().item()
            bag_false_positive += ((bag_label == 0) & (predicted_label == 1)).sum().item()
            bag_false_negative += ((bag_label == 1) & (predicted_label == 0)).sum().item()

            # ---- Print out some attention weights
            # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            #     instance_level = list(zip(instance_labels.tolist(),
            #                               np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
            #     print('\n~~~~~~\nTrue Bag Label, Predicted Bag Label: {}\n'
            #           'True Instance Labels, Attention Weights: {}\n~~~~~~'.format(bag_level,instance_level))
            # ----

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            # instance level accuracy
            instance_prob = np.round(attention_weights.cpu().data.numpy()[0], decimals=3)
            # instance_prob_mean = np.mean(instance_prob)
            # instance_pred_label = (instance_prob > instance_prob_mean).astype(int)
            instance_pred_label = (instance_prob > 0.5).astype(int)
            # Precision and Recall
            true_positive += ((instance_labels == 1) & (instance_pred_label == 1)).sum()
            true_negative += ((instance_labels == 0) & (instance_pred_label == 0)).sum()
            false_positive += ((instance_labels == 0) & (instance_pred_label == 1)).sum()
            false_negative += ((instance_labels == 1) & (instance_pred_label == 0)).sum()

        # calculate loss and error for epoch
        train_loss /= len(train_loader)
        train_error /= len(train_loader)
        # calculate Bag level Precision, Recall, F1-score, Accuracy
        bag_precision = bag_true_positive / (bag_true_positive + bag_false_positive + 1e-9)
        bag_recall = bag_true_positive / (bag_true_positive + bag_false_negative + 1e-9)
        train_bag_f1 = 2 * bag_precision * bag_recall / (bag_precision + bag_recall + 1e-9)
        train_bag_acc = (bag_true_positive + bag_true_negative) / (bag_true_positive + bag_true_negative +
                                                                   bag_false_positive + bag_false_negative)
        # calculate Instance level Precision, Recall, F1-score, Accuracy
        precision = true_positive / (true_positive + false_positive + 1e-9)
        recall = true_positive / (true_positive + false_negative + 1e-9)
        instance_f1_score = 2 * precision * recall / (precision + recall + 1e-9)
        instance_acc = (true_positive + true_negative) / (true_positive + true_negative +
                                                          false_positive + false_negative)

        train_end = time.time()
        print('--------------------------------------------------------------------------------')
        print('Train Set - Epoch: {} - Runtime: {}'.format(epoch, train_end - train_start))
        print('Loss: {:.4f}, Error: {:.4f}'.format(train_loss.cpu().numpy()[0], train_error))
        print('Bag      Level - Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, Acc: {:.4f}'.format(
            bag_precision, bag_recall, train_bag_f1, train_bag_acc))
        print('Instance Level - Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, Acc: {:.4f}'.format(
            precision, recall, instance_f1_score, instance_acc))

        scheduler.step(train_error)

        if (train_bag_f1 > best_f1_score) or ((train_bag_f1 == best_f1_score)
                                              and (instance_f1_score >= best_instance_f1_score)):
            best_f1_score = train_bag_f1
            best_instance_f1_score = instance_f1_score
            best_epoch = epoch
            torch.save(model.state_dict(), 'model/vgg_16_epoch_best.pth')

    print('Best at epoch %d, train set F1 score is %f, instance F1 score is %f' % (best_epoch, best_f1_score,
                                                                                   best_instance_f1_score))


if __name__ == "__main__":
    print('Start Training')
    train()
    end = time.time()
    print('------ Total Runtime {} ------'.format(end-start))
