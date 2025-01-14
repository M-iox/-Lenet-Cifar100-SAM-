import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from argparse import ArgumentParser
from models import *
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        SAM 优化器初始化
        :param params: 模型参数
        :param base_optimizer: 基础优化器（如 SGD）
        :param rho: 扰动幅度
        :param kwargs: 传递给基础优化器的其他参数
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        第一步：计算扰动并更新权重到 w + e(w)
        :param zero_grad: 是否清空梯度
        """
        grad_norm = self._grad_norm()  # 计算梯度范数
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)  # 计算扰动
                p.add_(e_w)  # 应用扰动
                self.state[p]["e_w"] = e_w  # 保存扰动值
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        第二步：回到 w 并进行实际权重更新
        :param zero_grad: 是否清空梯度
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # 恢复到原始权重 w
        self.base_optimizer.step()  # 更新权重
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        """
        计算梯度的 2 范数
        :return: 梯度范数
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
class CIFARTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.args.gpu)
        self.best_acc = 0
        self.start_epoch = 0
        self.best_epoch = 0
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        os.makedirs(self.args.save_dir, exist_ok=True)
        self._prepare_data()
        self._build_model()
        self._setup_optimizer()

    def _prepare_data(self):
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=False, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=False, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def _build_model(self):
        print('==> Building model..')
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        activation_func = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "gelu": F.gelu,
            "elu": F.elu
        }.get(self.args.func, None)

        if activation_func is None:
            raise ValueError(f"Unsupported activation function: {self.args.func}")

        self.net = LeNet2(activation_func, self.args.norm, self.args.attention)
        self.net = self.net.to(self.device)
        self.net.apply(init_weights)

        if self.device.type == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        if self.args.resume:
            print('==> Resuming from checkpoint..')
            load_path = os.path.join(self.args.save_dir, 'ckpt.pth')
            checkpoint = torch.load(load_path)
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            self.best_epoch = self.start_epoch

    def _setup_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()

        if self.args.opt == "sam":
            self.optimizer = SAM(
                self.net.parameters(),
                base_optimizer=optim.SGD,  # SAM 默认使用 SGD 作为基础优化器
                rho=0.05,  # 扰动大小，可以根据需求调整
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=5e-4
            )
        elif self.args.opt == "sgd":
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=5e-4
            )
        elif self.args.opt == "adam":
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.args.lr,
                weight_decay=5e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.opt}")

        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epoch)

    def _plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        save_path = f"{self.args.opt}_metrics_plot.pdf"
        plt.savefig(save_path, format='pdf')
        print(f"Metrics plot saved as {save_path}")

    def train_one_epoch(self, epoch):
        print(f'\nEpoch: {epoch}')
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        train_loader_iter = tqdm(self.trainloader, total=len(self.trainloader))
        for inputs, targets in train_loader_iter:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 如果是 SAM 优化器，执行两步优化
            if isinstance(self.optimizer, SAM):
                # 第一次前向与反向传播
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # 第二次前向与反向传播
                outputs = self.net(inputs)
                self.criterion(outputs, targets).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                # 普通优化器
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            train_loader_iter.set_postfix({
                'Loss': f"{train_loss / (len(self.trainloader)):.3f}",
                'Acc': f"{100. * correct / total:.3f}% ({correct}/{total})"
            })

        self.train_losses.append(train_loss / len(self.trainloader))
        self.train_accuracies.append(100. * correct / total)

    def test_one_epoch(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0

        test_loader_iter = tqdm(self.testloader, total=len(self.testloader))
        with torch.no_grad():
            for inputs, targets in test_loader_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                test_loader_iter.set_postfix({
                    'Loss': f"{test_loss / (len(self.testloader)):.3f}",
                    'Acc': f"{100. * correct / total:.3f}% ({correct}/{total})"
                })

        self.test_losses.append(test_loss / len(self.testloader))
        self.test_accuracies.append(100. * correct / total)

        acc = 100. * correct / total
        print(f"epoch:{epoch}, test loss: {test_loss}, test accuracy: {acc:.4f}")
        if acc > self.best_acc:
            print('Saving..')
            state = {'net': self.net.state_dict(), 'acc': acc, 'epoch': epoch}
            save_path = os.path.join(self.args.save_dir, 'ckpt.pth')
            torch.save(state, save_path)
            self.best_acc = acc
            self.best_epoch = epoch
        print(f"best_acc:{self.best_acc:.4f}")
        print(f"best_epoch:{self.best_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.args.epoch):
            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)
            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                self._plot_metrics()

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epoch', default=200, type=int, help='总共训练批次')
    parser.add_argument('--batch_size', default=256, type=int, help='批次大小')
    parser.add_argument('--gpu', default=0, type=int, help='gpu序号')
    parser.add_argument('--num_workers', default=0, type=int, help='线程数量')
    parser.add_argument('--save_dir', default='./checkpoint', type=str, help="保存位置")
    parser.add_argument('--func', default='elu', type=str, help="激活函数")
    parser.add_argument('--norm', default='bn', type=str, help="归一化类型（none, bn, ln, gn）")
    parser.add_argument('--attention', default='se', type=str, help="视觉注意力")
    parser.add_argument('--opt', default='sam', type=str, help="优化器")

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, str(args.batch_size))

    trainer = CIFARTrainer(args)
    trainer.train()
