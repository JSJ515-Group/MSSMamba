import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, test_single_volume
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from datasets.dataset_synapse import Synapse_dataset, RandomGenerator


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)  # pt is the probability of the correct class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(testloader.dataset)
    
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return performance, mean_hd95


def plot_result(dice, h, snapshot_path, args):
    # 创建一个字典并转换为 DataFrame
    data_dict = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(data_dict)

    # 绘制 Mean Dice 图像
    plt.figure(0)
    df['mean_dice'].plot()
    plt.title('Mean Dice')
    plt.xlabel('Epochs')  # 添加 x 轴标签
    plt.ylabel('Mean Dice')  # 添加 y 轴标签
    resolution_value = 1200
    date_and_time = datetime.datetime.now()
    formatted_date_time = date_and_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename_dice = f'{args.model_name}_{formatted_date_time}_dice.png'
    save_mode_path_dice = os.path.join(snapshot_path, filename_dice)
    plt.savefig(save_mode_path_dice, format="png", dpi=resolution_value)
    plt.close()  # 关闭当前图形，释放内存

    # 绘制 Mean HD95 图像
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean HD95')
    plt.xlabel('Epochs')  # 添加 x 轴标签
    plt.ylabel('Mean HD95')  # 添加 y 轴标签
    filename_hd95 = f'{args.model_name}_{formatted_date_time}_hd95.png'
    save_mode_path_hd95 = os.path.join(snapshot_path, filename_hd95)
    plt.savefig(save_mode_path_hd95, format="png", dpi=resolution_value)
    plt.close()  # 关闭当前图形，释放内存

    # 保存结果到 CSV 文件
    filename_results = f'{args.model_name}_{formatted_date_time}_results.csv'
    save_mode_path_results = os.path.join(snapshot_path, filename_results)
    df.to_csv(save_mode_path_results, sep='\t')



def trainer(args, model, snapshot_path):
    # 获取当前日期和时间，并格式化为字符串，确保没有非法字符
    now = datetime.datetime.now()
    date_and_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")  # 使用下划线和破折号代替冒号

    # 创建测试结果的保存路径（如果尚不存在）
    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    # 构造日志文件的完整路径
    log_filename = os.path.join(snapshot_path, f"{args.model_name}_{date_and_time_str}_log.txt")

    # 配置日志记录
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))


    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # focal_loss = FocalLoss(alpha=1, gamma=2)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))


    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_=[]
    hd95_= []



    ##原始代码
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            B, C, H, W = image_batch.shape
            image_batch = image_batch.expand(B, 3, H, W)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))



            try:
                if iter_num % 10 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
            except: pass

        # Test
        if (epoch_num + 1) % args.eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()

            iterator.close()
            break

    plot_result(dice_,hd95_,snapshot_path,args)
    writer.close()
    return "Training Finished!"