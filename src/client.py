import copy
import gc
import io
import logging

import numpy as np

# custom packages
from utils.utils import pretty_list

logger = logging.getLogger(__name__)
import torch


class Client(object):
    def __init__(self,
                 client_id,
                 device,
                 distribution,
                 batch_size,
                 local_epoch):

        # server side configs
        self.id = client_id
        self.device = device
        self.distribution = distribution

        # client training configs
        self.batch_size = batch_size
        self.local_epoch = local_epoch
        self.criterion = 'torch.nn.CrossEntropyLoss'
        self.optimizer = 'torch.optim.SGD'
        self.optim_config = {
            'lr': 0.1,
            'momentum': 0.9,
        }

        # dataset
        self.train = None
        self.test = None

        # models
        self.client_current = None
        self.global_previous = None
        self.client_previous = None
        self.global_current = None
        self.global_sparse_weights = None

        # For scaffold only
        self.c_local = None
        self.c_global = None
        self.c_delta = None
        self.gradient_magnitude = []
        self.sparse_ratio = 0.1

    def get_gradient(self):
        grad = np.subtract(self.global_current.flatten_model(), self.client_current.flatten_model())
        # return grad / (args.num_sample * args.local_ep * lr / args.local_bs)
        # grad = grad / (len(self.train) * self.local_epoch * self..OPTIMIZER_CONFIG['lr'] / self.batch_size)
        return np.array(grad)

    def set_gradient(self, gradient):
        difference = np.subtract(self.get_gradient(), gradient)
        print(np.linalg.norm(difference))
        new_parameter = np.subtract(self.global_current.flatten_model(), gradient)
        new_parameter = self.client_current.unflatten_model(new_parameter)

        self.client_current.load_state_dict(new_parameter)

    def get_gradient_s(self, model1, model2, difference=False):
        grad = np.subtract(model1.flatten_model(), model2.flatten_model())
        if difference:
            return grad
        # return grad / (args.num_sample * args.local_ep * lr / args.local_bs)
        grad = grad / (len(self.train) * self.local_epoch * self.optim_config['lr'] / self.batch_size)
        return grad

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train)

    def update_train(self, new_dataset, replace=False):
        if self.train is None or replace:
            self.train = new_dataset
        else:
            self.train + new_dataset

    def update_test(self, new_dataset, replace=True):
        if self.test is None or replace:
            self.test = new_dataset
        else:
            self.test + new_dataset

    def client_update(self, update_type):
        if update_type == 'fedprox':
            return self.client_update_fedprox()
        elif update_type == 'gradient_compression':
            return self.client_update_gradient_compression_inner5()
        elif update_type == 'scaffold':
            return self.client_update_scaffold()
        elif update_type == 'fedavg':
            return self.client_update_fedavg()
        elif update_type == 'param_freeze':
            return self.client_update_param_freeze()
        elif update_type == 'reversed_l1':
            return self.client_update_adaptive_l1_mask()
        else:
            return self.client_update_fedavg()

    def client_update_fedavg(self):
        """Update local model using local dataset."""
        self.client_current.train()
        self.client_current.to(self.device)

        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")

    def client_update_reversed_l1(self):
        """Update local model using local dataset."""
        self.client_current.train()
        self.client_current.to(self.device)

        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        global_parameters = {}
        for name, param in self.global_current.named_parameters():
            global_parameters[name] = param.clone().detach()  # 复制参数并确保不计算梯度

        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                for name, param in self.client_current.named_parameters():
                    update_magnitude = torch.abs(param - global_parameters[name])
                    reward_threshold = np.percentile(np.abs(param.detach().cpu().numpy().flatten()), 80)

                    reward_mask = update_magnitude >= reward_threshold  # 大于阈值的更新
                    penalty_mask = update_magnitude < reward_threshold  # 小于阈值的更新

                    loss -= 0.1 * torch.sum(reward_mask * update_magnitude)  # 奖励更新幅度大的参数
                    loss += 0.1 * torch.sum(penalty_mask * update_magnitude)  # 惩罚更新幅度小的参数
                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")

    def client_update_adaptive_l1_mask(self):
        """Update local model using local dataset with adaptive L1 regularization using mask."""
        self.client_current.train()
        self.client_current.to(self.device)
        global_weight_collector = list(self.global_current.to(self.device).parameters())
        mu = 0.02
        alpha = 0  # 动态阈值系数
        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        num_params = 0
        num_l1_applied = 0
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)
                param_deltas = []
                abs_param_deltas = []
                for param_index, param in enumerate(self.client_current.parameters()):
                    delta_theta = param - global_weight_collector[param_index]
                    param_deltas.append(delta_theta)
                    abs_param_deltas.append(torch.abs(delta_theta))
                delta_thetas = np.concatenate(
                    [delta_theta.detach().cpu().numpy().ravel() for delta_theta in param_deltas])
                abs_delta_thetas = np.concatenate(
                    [abs_delta_theta.detach().cpu().numpy().ravel() for abs_delta_theta in abs_param_deltas])
                tau = np.mean(abs_delta_thetas) + alpha * np.std(abs_delta_thetas)
                print('tau is ', tau)
                mask = [torch.abs(delta_theta) < tau for delta_theta in param_deltas]
                l1_reg = 0.0
                num_params = 0
                num_l1_applied = 0
                for delta_theta, mask_tensor in zip(param_deltas, mask):
                    sparse_delta = delta_theta * mask_tensor  # 对 delta_theta 进行 mask
                    l1_reg += mu * torch.norm(sparse_delta, p=1)
                    num_params += delta_theta.numel()
                    num_l1_applied += mask_tensor.sum().item()
                loss += l1_reg
                loss.backward()
                optimizer.step()
                if self.device == "cuda": torch.cuda.empty_cache()
        l1_ratio = num_l1_applied / num_params if num_params > 0 else 0
        print(f"Epoch {e + 1}, L1 Regularization Applied to {l1_ratio:.2%} of Parameters")
        self.client_current.to("cpu")

    def client_update_param_mask(self):
        hook_handles = []

        # Define a hook function to zero out gradients based on a bool_idx
        def create_hook_fn(bool_idx):
            def hook_fn(grad):
                grad[bool_idx] = 0
                return grad

            return hook_fn

        self.client_current.train()
        self.client_current.to(self.device)

        for e in range(self.local_epoch):
            total_batches = len(self.train.get_dataloader())
            current_batch = 0
            target_batch = max(int(total_batches * 0.1), 1)
            optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)

            saved_state_dict_buffer = io.BytesIO()
            torch.save(self.client_current.state_dict(), saved_state_dict_buffer)
            saved_state_dict_buffer.seek(0)

            for data, labels in self.train.get_dataloader():
                if current_batch == target_batch:
                    gradient = self.get_gradient()
                    threshold = np.percentile(np.abs(gradient), 80)
                    gradient = self.client_current.unflatten_model(gradient)

                    loaded_state_dict = torch.load(saved_state_dict_buffer)
                    self.client_current.load_state_dict(loaded_state_dict)
                    optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)

                    for layer_name in gradient:
                        bool_idx = torch.tensor(np.abs(gradient[layer_name]) < threshold)
                        layer = getattr(self.client_current, layer_name[:-7])  # remove '.weight' at the end
                        hook_handle = layer.weight.register_hook(create_hook_fn(bool_idx))
                        hook_handles.append(hook_handle)

                current_batch += 1
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        print('Gradient for fc1 layer:', self.client_current.fc1.weight.grad)

        for hook_handle in hook_handles:
            hook_handle.remove()
        self.client_current.to("cpu")

    def client_update_fedprox(self):
        """Update local model using local dataset."""
        self.client_current.train()
        self.client_current.to(self.device)

        global_weight_collector = list(self.global_current.to(self.device).parameters())
        mu = 0.01

        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.client_current.parameters()):
                    fed_prox_reg += (mu * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")

    def client_update_scaffold(self):
        """Update local model using local dataset."""
        if self.c_global is None:
            self.c_global = copy.deepcopy(self.client_current)
        if self.c_local is None:
            self.c_local = copy.deepcopy(self.client_current)

        self.client_current.train()
        self.client_current.to(self.device)
        self.c_global.to(self.device)
        self.c_local.to(self.device)
        self.global_current.to(self.device)

        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()
        count = 0
        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                net_para = self.client_current.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - self.optim_config['lr'] * \
                                    (c_global_para[key] - c_local_para[key])
                self.client_current.load_state_dict(net_para)
                count += 1

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")
        self.c_global.to("cpu")
        self.c_local.to("cpu")
        self.global_current.to("cpu")

        c_new_para = self.c_local.state_dict()
        c_delta_para = copy.deepcopy(self.c_local.state_dict())
        global_current_para = self.global_current.state_dict()
        client_current_para = self.client_current.state_dict()

        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()

        for key in client_current_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (
                    global_current_para[key] - client_current_para[key]) / \
                              (count * self.optim_config['lr'])

            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        self.c_local.load_state_dict(c_new_para)
        self.c_delta_para = c_delta_para
        # print(self.c_delta_para)

    def evaluate(self, model, dataset):
        model.eval()
        model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataset.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        model.to("cpu")

        test_loss = test_loss / len(dataset.get_dataloader())
        test_accuracy = correct / len(dataset)

        return test_accuracy, test_loss

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.client_current.eval()
        self.client_current.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.test.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.client_current(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.client_current.to("cpu")

        test_loss = test_loss / len(self.test.get_dataloader())
        test_accuracy = correct / len(self.test)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\
            \n\t=> Distribution: {pretty_list(self.distribution.tolist())}\n"

        print(message, flush=True)
        logging.info(message)

        del message
        gc.collect()

        return test_loss, test_accuracy

    @staticmethod
    def restore_model_param_freeze(model):
        # 合并参数并恢复模型
        for name, layer in model.named_modules():
            params = list(layer.named_parameters(recurse=False))
            for param_name, param in params:
                # 检查是否存在拆分的参数
                if hasattr(layer, f'param_requires_grad_{param_name}'):
                    param_requires_grad = getattr(layer, f'param_requires_grad_{param_name}')
                    param_no_grad = getattr(layer, f'param_no_grad_{param_name}')
                    indices_requires_grad = getattr(layer, f'indices_requires_grad_{param_name}')
                    indices_no_grad = getattr(layer, f'indices_no_grad_{param_name}')
                    original_shape = getattr(layer, f'original_shape_{param_name}')

                    # 重构完整的参数张量
                    param_flat = torch.zeros(param_requires_grad.numel() + param_no_grad.numel(), device=param.device)
                    param_flat[indices_requires_grad] = param_requires_grad.data
                    param_flat[indices_no_grad] = param_no_grad.data
                    param_full = param_flat.view(original_shape)

                    # 更新原始参数
                    param.data.copy_(param_full)

                    # 删除临时的参数属性
                    delattr(layer, f'param_requires_grad_{param_name}')
                    delattr(layer, f'param_no_grad_{param_name}')
                    delattr(layer, f'indices_requires_grad_{param_name}')
                    delattr(layer, f'indices_no_grad_{param_name}')
                    delattr(layer, f'original_shape_{param_name}')

            # 恢复原始的 forward 方法
            if hasattr(layer, '_original_forward'):
                layer.forward = layer._original_forward
                del layer._original_forward

            # 删除绑定标记
            if hasattr(layer, 'modified_forward_bound'):
                del layer.modified_forward_bound

            # 删除其他临时属性
            attrs_to_remove = [attr for attr in dir(layer) if attr.startswith('param_requires_grad_') or
                               attr.startswith('param_no_grad_') or
                               attr.startswith('indices_requires_grad_') or
                               attr.startswith('indices_no_grad_') or
                               attr.startswith('original_shape_')]
            for attr in attrs_to_remove:
                delattr(layer, attr)

    def get_performance_gap(self):
        _, global_accuracy = self.client_evaluate(current_model=False, log=False)
        _, current_accuracy = self.client_evaluate(current_model=True, log=False)

        self.idx_t1 = current_accuracy - global_accuracy

        # print(id(self.model))
        # print(id(self.global_model))

        if self.idx_t0 is None:
            self.idx_t0 = self.idx_t1
        else:
            res = self.idx_t1 - self.idx_t0

            message = f"\t[Client {str(self.id).zfill(4)}]:!\
                \n\t=> Global Accuracy: {100. * global_accuracy:.2f}%\
                \n\t=> Current Accuracy: {100. * current_accuracy:.2f}%\
                \n\t=> idx_t0: {100. * self.idx_t0: .2f}%\
                \n\t=> idx_t1: {100. * self.idx_t1: .2f}%"

            print(message, flush=True);
            logging.info(message)

            return max(res, 0.000001)

    @staticmethod
    def view_training_samples(data, labels):
        import matplotlib.pyplot as plt
        for i in range(min(10, len(data))):
            img = data[i].permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C) for displaying
            plt.subplot(2, 5, i + 1)
            plt.imshow(img)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis('off')

        plt.show()
