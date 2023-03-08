import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

from utils.centrality import get_centrality, get_centrality_all
from utils.utils import *
from model.modules import (
    FeatureNet,
    PredNet
)

from visdom import Visdom
import torch.autograd as autograd

# the base model
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)

        self.model_name = opt.model
        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.A = opt.A
        self.dataset = opt.dataset

        self.device = opt.device
        if "DRO" in self.model_name or "TRO" in self.model_name:
            self.groupdro_eta = opt.groupdro_eta
        if "TRO" in self.model_name:
            self.lmbda = opt.lmbda

        # visualization
        self.use_visdom = opt.use_visdom
        if opt.use_visdom:
            self.env = Visdom(port=opt.visdom_port)
            self.test_pane = dict()

        self.src_domain = opt.src_domain
        self.criterion = nn.NLLLoss().cuda()
        self.netE = FeatureNet(opt).to(opt.device) # encoder
        self.netF = PredNet(opt).to(opt.device) # predictor

        self.__init_weight__()

        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())

        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )

        self.lr_schedulers = [self.lr_scheduler_EF]
        self.loss_names = ["loss"]

        self.num_domain = opt.num_domain
        if self.opt.test_on_all_dmn:
            self.test_dmn_num = self.num_domain
        else:
            self.test_dmn_num = self.opt.tgt_dmn_num

        self.outf = opt.outf
        if opt.outf:
            self.train_log = os.path.join(opt.outf, "loss.log")
            self.model_path = os.path.join(opt.outf, "model.pth")

            if not os.path.exists(self.opt.outf):
                os.mkdir(self.opt.outf)
            with open(self.train_log, "w") as f:
                f.write("log start!\n")

        src_domains = [str(i) for i in self.src_domain]
        if self.outf:
            self.__log_write__("src domains: " + " ".join(src_domains))

        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.src_domain] = 1

        self.domain_mask = torch.IntTensor(mask_list).to(opt.device)


    def learn(self, epoch, dataloader):
        self.train()

        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}

        count = 0
        for data in dataloader:
            count += 1
            self.__set_input__(data)
            self.__train_forward__()
            new_loss_values = self.__optimize__()

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        if (self.epoch + 1) % 10 == 0:
            print("epoch {}: {}".format(self.epoch, loss_values))

        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def test(self, epoch, dataloader):
        self.eval() # validation mode

        acc_curve = []
        l_x = []
        l_domain = []
        l_label = []
        l_encode = []

        for data in dataloader:
            self.__set_input__(data)

            # forward
            with torch.no_grad():
                self.__test_forward__()

            acc_curve.append(
                self.g_seq.eq(self.y_seq)
                    .to(torch.float)
                    .mean(-1, keepdim=True)
            )
            l_x.append(to_np(self.x_seq))
            l_domain.append(to_np(self.domain_seq))
            l_encode.append(to_np(self.e_seq))
            l_label.append(to_np(self.g_seq))

        x_all = np.concatenate(l_x, axis=1)
        e_all = np.concatenate(l_encode, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        label_all = np.concatenate(l_label, axis=1)

        d_all = dict()

        d_all["data"] = flat(x_all)
        d_all["domain"] = flat(domain_all)
        d_all["label"] = flat(label_all)
        d_all["encodeing"] = flat(e_all)

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))

        test_acc = (
                (acc.sum() - acc[self.opt.src_domain].sum())
                / (self.opt.num_target)
                * 100
        )

        acc_msg = "[{}] Acc: all avg {:.1f}, test avg {:.2f}".format(epoch, acc.mean() * 100, test_acc)
        each_domain_acc = [str(i) for i in np.around(acc * 100, decimals=1)]
        each_domain_acc_msg = "[" + ",  ".join(each_domain_acc) + "]"

        if self.outf:
            self.__log_write__(acc_msg)
            self.__log_write__(each_domain_acc_msg)

        if self.use_visdom:
            self.__vis_test_error__(test_acc, "test acc")
        d_all["acc_msg"] = acc_msg

    def __vis_test_error__(self, loss, title):
        if self.epoch == self.opt.test_interval - 1:
            # initialize
            self.test_pane[title] = self.env.line(
                X=np.array([self.epoch]),
                Y=np.array([loss]),
                opts=dict(title=title),
            )
        else:
            self.env.line(
                X=np.array([self.epoch]),
                Y=np.array([loss]),
                win=self.test_pane[title],
                update="append",
            )

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        torch.load(name)

    def __set_input__(self, data, train=True):
        """
        :param
            x_seq: Number of domain x Batch size x  Data dim
            y_seq: Number of domain x Batch size x Predict Data dim
            one_hot_seq: Number of domain x Batch size x Number of vertices (domains)
            domain_seq: Number of domain x Batch size x domain dim (1)
        """
        x_seq, y_seq, domain_seq = (
            [d[0][None, :, :] for d in data],
            [d[1][None, :] for d in data],
            [d[2][None, :] for d in data],
        )
        self.x_seq = torch.cat(x_seq, 0).to(self.device)
        self.y_seq = torch.cat(y_seq, 0).to(self.device)

        self.domain_seq = torch.cat(domain_seq, 0).to(self.device)
        self.tmp_batch_size = self.x_seq.shape[1]
        one_hot_seq = [
            torch.nn.functional.one_hot(d[2], self.num_domain)
            for d in data
        ]

        if train:
            self.one_hot_seq = (
                torch.cat(one_hot_seq, 0)
                .reshape(self.num_domain, self.tmp_batch_size, -1)
                .to(self.device)
            )
        else:
            self.one_hot_seq = (
                torch.cat(one_hot_seq, 0)
                .reshape(self.test_dmn_num, self.tmp_batch_size, -1)
                .to(self.device)
            )

    def __train_forward__(self):
        self.e_seq = self.netE(self.x_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)  # prediction

    def __test_forward__(self):
        self.e_seq = self.netE(self.x_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        if "toy" in self.dataset:
            self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)  # class of the prediction

    def test_x_seq(self, x_seq):
        e_seq = self.netE(x_seq)  # encoder of the data
        f_seq = self.netF(e_seq)
        if "toy" in self.dataset:
            g_seq = torch.argmax(f_seq.detach(), dim=2)  # class of the prediction
        return g_seq

    def test_x_seq_feat(self, x_seq):
        e_seq = self.netE(x_seq)  # encoder of the data
        _, x_seq_feat = self.netF(e_seq, return_feature=True)
        return x_seq_feat

    def __optimize__(self):
        loss_value = dict()

        self.loss_E_pred = self.__loss_EF__()

        self.optimizer_EF.zero_grad()
        self.loss_E_pred.backward(retain_graph=True)
        self.optimizer_EF.step()

        loss_value["loss"] = self.loss_E_pred.item()
        return loss_value

    def __loss_EF__(self):
        pass

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, "a") as f:
            f.write(loss_msg + "\n")

    def __vis_loss__(self, loss_values):
        if self.epoch == 0:
            self.panes = {
                loss_name: self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title="loss for {} on epochs".format(loss_name)),
                )
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    win=self.panes[loss_name],
                    update="append",
                )

    def __init_weight__(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

class ERM(BaseModel):
    """
    ERM Model
    """
    def __init__(self, opt):
        super(ERM, self).__init__(opt)

    def __loss_EF__(self):
        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]
        loss_E_pred = self.criterion(flat(f_seq_source), flat(y_seq_source))
        return loss_E_pred

class DRO(BaseModel):
    """
    DRO Model
    """
    def __init__(self, opt):
        super(DRO, self).__init__(opt)

        # q
        self.register_buffer("q", torch.Tensor())

    def __loss_EF__(self):

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        if not len(self.q):
            self.q = torch.ones(len(self.src_domain)).to(self.device)

        losses = torch.zeros(len(self.src_domain)).to(self.device)

        for m in range(len(self.src_domain)):
            losses[m] = self.criterion(f_seq_source[m], y_seq_source[m])
            self.q[m] *= (self.groupdro_eta * losses[m].data).exp()

        self.q /= self.q.sum()

        loss_E_pred = torch.dot(losses, self.q)

        return loss_E_pred


class IRM(BaseModel):
    """
    IRM Model
    """
    def __init__(self, opt):
        super(IRM, self).__init__(opt)

        # Doesn't use it for now (for penalty and lr decay)
        self.register_buffer('update_count', torch.tensor([0]))
        self.irm_penal = opt.irm_penal

    def _irm_penalty(self, logits, y):
        scale = torch.tensor(1.).to("cuda").requires_grad_()
        loss_1 = self.criterion(logits[::2] * scale, y[::2])
        loss_2 = self.criterion(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def __loss_EF__(self):

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        nll = 0.
        penalty = 0.

        for m in range(len(self.src_domain)):
            nll += self.criterion(f_seq_source[m], y_seq_source[m])
            penalty += self._irm_penalty(f_seq_source[m], y_seq_source[m])

        nll /= len(self.src_domain)
        penalty /= len(self.src_domain)
        loss_E_pred = nll + (self.irm_penal * penalty)

        return loss_E_pred

class TRO(BaseModel):
    """
    TRO Model
    """
    def __init__(self, opt):
        super(TRO, self).__init__(opt)
        self.register_buffer("q", torch.Tensor())

        # Graph centrality
        if opt.learn == 0: # physical graph is provided
            if opt.partial == 0: # use both source + target graph
                central = get_centrality_all(self.A, opt.src_domain, opt.num_domain)
                self.prior = central[opt.src_domain]
            else: # only use source
                central = get_centrality(self.A)
                self.prior = central

            self.prior /= self.prior.sum()
            self.prior = np.around(self.prior, 3)
            
        else: # data graph is used
            # DG-15
            if "15" in opt.dataset:
                # replace self.prior with values generated from learn_graph.py
                self.prior = [0.0, 0.0, 0.0, 0.0, 0.0, 0.4]
            elif "60" in opt.dataset:
                self.prior = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]

            self.prior = np.asarray(self.prior)
            self.prior /= self.prior.sum()

        self.prior = torch.from_numpy(self.prior).to(self.device)

    def __loss_EF__(self):

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        if not len(self.q):
            # Learnable q
            self.q = torch.ones(len(self.src_domain)).to(self.device)
            self.q /= self.q.sum() # Key!

        losses = torch.zeros(len(self.src_domain)).to(self.device)

        for m in range(len(self.src_domain)):
            losses[m] = self.criterion(f_seq_source[m], y_seq_source[m])
            self.q[m] += self.groupdro_eta * (losses[m] - self.lmbda * (self.q[m] - self.prior[m]))

        self.q = to_tensor(projection_simplex(to_np(self.q)), self.device)

        loss_E_pred = torch.dot(losses, self.q.to(torch.float))

        return loss_E_pred