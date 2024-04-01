import torch
import time
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
from termcolor import cprint
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt
from datetime import datetime
from src.lie_algebra import SO3, CPUSO3


class LearningBasedProcessing:
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        self.res_dir = res_dir
        self.tb_dir = tb_dir
        self.net_class = net_class
        self.net_params = net_params
        self._ready = False
        self.train_params = {}
        self.figsize = (20, 12)
        self.dt = dt # (s)
        self.address, self.tb_address = self.find_address(address)
        if address is None:  # create new address
            pdump(self.net_params, self.address, 'net_params.p')
            ydump(self.net_params, self.address, 'net_params.yaml')
        else:  # pick the network parameters
            self.net_params = pload(self.address, 'net_params.p')
            self.train_params = pload(self.address, 'train_params.p')
            self._ready = True
        self.path_weights = os.path.join(self.address, 'weights.pt')
        self.net = self.net_class(**self.net_params)
        if self._ready:  # fill network parameters
            self.load_weights()

    def find_address(self, address):
        """return path where net and training info are saved"""
        if address == 'last':
            addresses = sorted(os.listdir(self.res_dir))
            tb_address = os.path.join(self.tb_dir, str(len(addresses)))
            address = os.path.join(self.res_dir, addresses[-1])
        elif address is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            address = os.path.join(self.res_dir, now)
            mkdir(address)
            tb_address = os.path.join(self.tb_dir, now)
        else:
            tb_address = None
        return address, tb_address

    def load_weights(self):
        weights = torch.load(self.path_weights)
        self.net.load_state_dict(weights)
        self.net.cuda()

    # def train(self, dataset_class, dataset_params, train_params):
    #     # 初始化损失函数
    #     LossClass = train_params['loss_class']
    #     loss_params = train_params['loss_params']
    #     self.criterion = LossClass(**loss_params)  # 正确地使用了'loss_class'和'loss_params'
    #     """train the neural network. GPU is assumed"""
    #     self.train_params = train_params
    #     pdump(self.train_params, self.address, 'train_params.p')
    #     ydump(self.train_params, self.address, 'train_params.yaml')
    #
    #     hparams = self.get_hparams(dataset_class, dataset_params, train_params)
    #     ydump(hparams, self.address, 'hparams.yaml')
    #
    #     # define datasets
    #     dataset_train = dataset_class(**dataset_params, mode='train')
    #     dataset_train.init_train()
    #     dataset_val = dataset_class(**dataset_params, mode='val')
    #     dataset_val.init_val()
    #
    #     # get class
    #     Optimizer = train_params['optimizer_class']
    #     Scheduler = train_params['scheduler_class']
    #     Loss = train_params['loss_class']
    #
    #     # get parameters
    #     dataloader_params = train_params['dataloader']
    #     optimizer_params = train_params['optimizer']
    #     scheduler_params = train_params['scheduler']
    #     loss_params = train_params['loss']
    #
    #     # define optimizer, scheduler and loss
    #     dataloader = DataLoader(dataset_train, **dataloader_params)
    #     optimizer = Optimizer(self.net.parameters(), **optimizer_params)
    #     scheduler = Scheduler(optimizer, **scheduler_params)
    #     criterion = Loss(**loss_params)
    #
    #     # remaining training parameters
    #     freq_val = train_params['freq_val']
    #     n_epochs = train_params['n_epochs']
    #
    #     # init net w.r.t dataset
    #     self.net = self.net.cuda()
    #     mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
    #     self.net.set_normalized_factors(mean_u, std_u)
    #
    #     # start tensorboard writer
    #     writer = SummaryWriter(self.tb_address)
    #     start_time = time.time()
    #     best_loss = torch.Tensor([float('Inf')])
    #
    #     # define some function for seeing evolution of training
    #     def write(epoch, loss_epoch):
    #         writer.add_scalar('loss/train', loss_epoch.item(), epoch)
    #         writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    #         print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(
    #             epoch, loss_epoch.item()))
    #         scheduler.step(epoch)
    #
    #     def write_time(epoch, start_time):
    #         delta_t = time.time() - start_time
    #         print("Amount of time spent for epochs " +
    #             "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
    #         writer.add_scalar('time_spend', delta_t, epoch)
    #
    #     def write_val(loss, best_loss):
    #         if 0.5*loss <= best_loss:
    #             msg = 'validation loss decreases! :) '
    #             msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
    #                 best_loss.item())
    #             cprint(msg, 'green')
    #             best_loss = loss
    #             self.save_net()
    #         else:
    #             msg = 'validation loss increases! :( '
    #             msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
    #                 best_loss.item())
    #             cprint(msg, 'yellow')
    #         writer.add_scalar('loss/val', loss.item(), epoch)
    #         return best_loss
    #
    #     # training loop !
    #     for epoch in range(1, n_epochs + 1):
    #         loss_epoch = self.loop_train(dataloader, optimizer, criterion)
    #         write(epoch, loss_epoch)
    #         scheduler.step(epoch)
    #         if epoch % freq_val == 0:
    #             loss = self.loop_val(dataset_val, criterion)
    #             write_time(epoch, start_time)
    #             best_loss = write_val(loss, best_loss)
    #             start_time = time.time()
    #     # training is over !
    #
    #     # test on new data
    #     dataset_test = dataset_class(**dataset_params, mode='test')
    #     self.load_weights()
    #     test_loss = self.loop_val(dataset_test, criterion)
    #     dict_loss = {
    #         'final_loss/val': best_loss.item(),
    #         'final_loss/test': test_loss.item()
    #         }
    #     writer.add_hparams(hparams, dict_loss)
    #     ydump(dict_loss, self.address, 'final_loss.yaml')
    #     writer.close()

    def train(self, dataset_class, dataset_params, train_params):
        # 初始化损失函数
        LossClass = train_params['loss_class']
        loss_params = train_params['loss_params']
        self.criterion = LossClass(**loss_params)  # 正确地使用了'loss_class'和'loss_params'
        """train the neural network. GPU is assumed"""
        self.train_params = train_params
        pdump(self.train_params, self.address, 'train_params.p')
        ydump(self.train_params, self.address, 'train_params.yaml')

        hparams = self.get_hparams(dataset_class, dataset_params, train_params)
        ydump(hparams, self.address, 'hparams.yaml')

        # define datasets
        dataset_train = dataset_class(**dataset_params, mode='train')
        dataset_train.init_train()
        dataset_val = dataset_class(**dataset_params, mode='val')
        dataset_val.init_val()

        # get class
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']

        # get parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']

        # define optimizer, scheduler
        dataloader = DataLoader(dataset_train, **dataloader_params)
        optimizer = Optimizer(self.net.parameters(), **optimizer_params)
        scheduler = Scheduler(optimizer, **scheduler_params)

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        # init net w.r.t dataset
        self.net = self.net.cuda()
        mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
        self.net.set_normalized_factors(mean_u, std_u)

        # start tensorboard writer
        writer = SummaryWriter(self.tb_address)
        start_time = time.time()
        best_loss = torch.Tensor([float('Inf')])

        # define some function for seeing evolution of training
        def write(epoch, loss_epoch):
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(epoch, loss_epoch.item()))
            scheduler.step(epoch)

        def write_time(epoch, start_time):
            delta_t = time.time() - start_time
            print("Amount of time spent for epochs " + "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
            writer.add_scalar('time_spend', delta_t, epoch)

        def write_val(loss, best_loss):
            if 0.5 * loss <= best_loss:
                msg = 'validation loss decreases! :) '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(), best_loss.item())
                cprint(msg, 'green')
                best_loss = loss
                self.save_net()
            else:
                msg = 'validation loss increases! :( '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(), best_loss.item())
                cprint(msg, 'yellow')
            writer.add_scalar('loss/val', loss.item(), epoch)
            return best_loss

        # training loop !
        for epoch in range(1, n_epochs + 1):
            loss_epoch = self.loop_train(dataloader, optimizer, self.criterion)  # 使用self.criterion计算损失
            write(epoch, loss_epoch)
            scheduler.step(epoch)
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, self.criterion)  # 同上
                write_time(epoch, start_time)
                best_loss = write_val(loss, best_loss)
                start_time = time.time()
        # training is over !

        # test on new data
        dataset_test = dataset_class(**dataset_params, mode='test')
        self.load_weights()
        test_loss = self.loop_val(dataset_test, self.criterion)  # 同上
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
        }
        writer.add_hparams(hparams, dict_loss)
        ydump(dict_loss, self.address, 'final_loss.yaml')
        writer.close()

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data"""
        for us, xs in dataloader:
            print(f"Input tensor shape: {us.shape}")
            print("In loop_train, before calling loss function...")
            # Add noise or any pre-processing if needed
            us, xs = us.cuda(), xs.cuda()

            print(f"Before model input, us shape: {us.shape}, type: {us.dtype}")  # 打印us的形状和类型
            print(f"Before model input, xs shape: {xs.shape}, type: {xs.dtype}")  # 打印xs的形状和类型

            # Print to debug
            print(f"Type of xs in loop_train: {type(xs)}")

            hat_xs = self.net(us)
            predicted = hat_xs
            print(f"Predicted tensor shape: {predicted.shape}")
            print(f"Sample predicted data: {predicted[0, :5]}")  # Adjust indices as needed
            loss = criterion(xs, hat_xs, us) / len(dataloader)
        loss_epoch = 0
        optimizer.zero_grad()
        for us, xs in dataloader:
            us = dataloader.dataset.add_noise(us.cuda())
            hat_xs = self.net(us)
            predicted = hat_xs
            loss = criterion(xs.cuda(), hat_xs)/len(dataloader)
            loss.backward()
            loss_epoch += loss.detach().cpu()
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data"""
        loss_epoch = 0
        self.net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                us, xs = dataset[i]
                hat_xs = self.net(us.cuda().unsqueeze(0))
                loss = criterion(xs.cuda().unsqueeze(0), hat_xs)/len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def save_net(self):
        """save the weights on the net in CPU"""
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), self.path_weights)
        self.net.train().cuda()

    def get_hparams(self, dataset_class, dataset_params, train_params):
        """return all training hyperparameters in a dict"""
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get training class parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss'] if 'loss' in train_params else {}

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        dict_class = {
            'Optimizer': str(Optimizer),
            'Scheduler': str(Scheduler),
            'Loss': str(Loss)
        }

        return {**dict_class, **dataloader_params, **optimizer_params,
                **loss_params, **scheduler_params,
                'n_epochs': n_epochs, 'freq_val': freq_val}

    def test(self, dataset_class, dataset_params, modes):
        """test a network once training is over"""

        # get loss function
        Loss = self.train_params['loss_class']
        loss_params = self.train_params['loss']
        criterion = Loss(**loss_params)

        # test on each type of sequence
        for mode in modes:
            dataset = dataset_class(**dataset_params, mode=mode)
            self.loop_test(dataset, criterion)
            self.display_test(dataset, mode)

    def loop_test(self, dataset, criterion):
        """Forward loop over test data"""
        self.net.eval()
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            us, xs = dataset[i]
            with torch.no_grad():
                hat_xs = self.net(us.cuda().unsqueeze(0))
            loss = criterion(xs.cuda().unsqueeze(0), hat_xs)
            mkdir(self.address, seq)
            mondict = {
                'hat_xs': hat_xs[0].cpu(),
                'loss': loss.cpu().item(),
            }
            pdump(mondict, self.address, seq, 'results.p')

    def display_test(self, dataset, mode):
        raise NotImplementedError


class GyroLearningBasedProcessing(LearningBasedProcessing):
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt,train_params,dataset_class, dataset_params):
        # 根据 train_params 初始化损失函数
        LossClass = train_params['loss_class']
        loss_params = train_params['loss_params']
        criterion = LossClass(**loss_params)  # 使用参数字典来初始化损失函数
        super().__init__(res_dir, tb_dir, net_class, net_params, address, dt)
        self.roe_dist = [7, 14, 21, 28, 35] # m
        self.freq = 100 # subsampling frequency for RTE computation
        self.roes = { # relative trajectory errors
            'Rots': [],
            'yaws': [],
            }

    def display_test(self, dataset, mode):
        self.roes = {
            'Rots': [],
            'yaws': [],
        }
        self.to_open_vins(dataset)
        for i, seq in enumerate(dataset.sequences):
            print('\n', 'Results for sequence ' + seq )
            self.seq = seq
            # get ground truth
            self.gt = dataset.load_gt(i)
            Rots = SO3.from_quaternion(self.gt['qs'].cuda())
            self.gt['Rots'] = Rots.cpu()
            self.gt['rpys'] = SO3.to_rpy(Rots).cpu()
            # get data and estimate
            self.net_us = pload(self.address, seq, 'results.p')['hat_xs']
            self.raw_us, _ = dataset[i]
            N = self.net_us.shape[0]
            self.gyro_corrections =  (self.raw_us[:, :3] - self.net_us[:N, :3])
            self.ts = torch.linspace(0, N*self.dt, N)

            self.convert()
            self.plot_gyro()
            self.plot_gyro_correction()
            plt.show()

    def to_open_vins(self, dataset):
        """
        Export results to Open-VINS format. Use them eval toolbox available 
        at https://github.com/rpng/open_vins/
        """

        for i, seq in enumerate(dataset.sequences):
            self.seq = seq

            # get ground truth
            self.gt = dataset.load_gt(i)
            # get raw us
            raw_us,_ = dataset[i]
            # get net us
            net_us = pload(self.address, seq, 'results.p')['hat_xs']

            N = net_us.shape[0]
            gt_qs = self.gt['qs']
            gt_qs = gt_qs[:N, :]
            covariance_matrix = np.zeros((N, 3, 3))
            net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)
            error = gt_qs - net_qs
            for j in range(N):
                #使用差异计算姿态误差协方差矩阵
                q_error = error[i]
                q_error_matrix = np.zeros((3, 3))
                q_error_matrix[0, 1] = -q_error[2]
                q_error_matrix[0, 2] = q_error[1]
                q_error_matrix[1, 0] = q_error[2]
                q_error_matrix[1, 2] = -q_error[0]
                q_error_matrix[2, 0] = -q_error[1]
                q_error_matrix[2, 1] = q_error[0]
                covariance_matrix[i] = np.dot(q_error_matrix, q_error_matrix.T)
            output_file = os.path.join(self.address, seq + '_covariance_matrix.txt')
            print(covariance_matrix)
            print(covariance_matrix.shape)
            np.savetxt(output_file, covariance_matrix.reshape((N, 9)), delimiter=',')
            print(net_qs)
            print(imu_Rots)
            print(net_Rots)
            path = os.path.join(self.address, seq + '.txt')
            path_net_qs = os.path.join(self.address, seq + '_net_qs.txt')
            path_imu_Rots = os.path.join(self.address, seq + '_imu_Rots.txt')
            path_net_Rots = os.path.join(self.address, seq + '_net_Rots.txt')
            header = "timestamp(s) tx ty tz qx qy qz qw"
            x = np.zeros((net_qs.shape[0], 8))
            x[:, 0] = self.gt['ts'][:net_qs.shape[0]]
            x[:, [7, 4, 5, 6]] = net_qs

            np.savetxt(path, x[::10], header=header, delimiter=" ",
                    fmt='%1.9f')
            np.savetxt(path_net_qs, net_qs, delimiter=" ", fmt='%1.9f')
            np.savetxt(path_imu_Rots, imu_Rots.cpu().reshape((N, 9)), delimiter=" ", fmt='%1.9f')
            np.savetxt(path_net_Rots, net_Rots.cpu().reshape((N, 9)), delimiter=" ", fmt='%1.9f')

    def convert(self):
        # s -> min
        l = 1/60
        self.ts *= l

        # rad -> deg
        l = 180/np.pi
        self.gyro_corrections *= l
        self.gt['rpys'] *= l

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*self.dt))
        Rot0 = SO3.qnorm(self.gt['qs'][:2].cuda().double())
        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]

        N = np.log2(imu_qs.shape[0])
        for i in range(int(N)):
            k = 2**i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = imu_qs[k:].shape[0]
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot_gyro(self):
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.net_us[:, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N,
        raw_us, net_us)
        imu_rpys = 180/np.pi*SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180/np.pi*SO3.to_rpy(net_Rots).cpu()
        self.plot_orientation(imu_rpys, net_rpys, N)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_orientation(self, imu_rpys, net_rpys, N):
        title = "Orientation estimation"
        gt = self.gt['rpys'][:N]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, gt[:, i], color='black', label=r'ground truth')
            axs[i].plot(self.ts, imu_rpys[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_rpys[:, i], color='blue', label=r'net IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N):
        gt = self.gt['Rots'][:N].cuda()
        raw_err = 180/np.pi*SO3.log(bmtm(imu_Rots, gt)).cpu()
        net_err = 180/np.pi*SO3.log(bmtm(net_Rots, gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, raw_err[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_err[:, i], color='blue', label=r'net IMU')
            axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation_error')

    def plot_gyro_correction(self):
        title = "Gyro correction" + self.end_title
        ylabel = 'gyro correction (deg/s)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, self.gyro_corrections, label=r'net IMU')
        ax.set_xlim(self.ts[0], self.ts[-1])
        self.savefig(ax, fig, 'gyro_correction')

    @property
    def end_title(self):
        return " for sequence " + self.seq.replace("_", " ")

    def savefig(self, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.address, self.seq, name + '.png'))

