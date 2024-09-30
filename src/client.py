import copy
import gc
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import io
# custom packages
from utils.utils import pretty_list
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
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
                        layer = getattr(self.client_current, layer_name[:-7]) # remove '.weight' at the end
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

    def client_update_param_freeze(self):
        def modified_forward(self, input):
            # Create a dictionary to store original parameters
            original_params = {}

            # Reconstruct and replace parameters
            for original_param_name in getattr(self, 'original_param_names'):
                if hasattr(self, f'param_requires_grad_{original_param_name}'):
                    # Retrieve the split parameters and indices
                    param_requires_grad = getattr(self, f'param_requires_grad_{original_param_name}')
                    param_no_grad = getattr(self, f'param_no_grad_{original_param_name}')

                    # First, calculate the forward pass with param_requires_grad
                    output_grad = self._forward_with_params(input, param_requires_grad, original_param_name)

                    # Then, calculate the forward pass with param_no_grad
                    output_no_grad = self._forward_with_params(input, param_no_grad, original_param_name)

                    # Add the two outputs together to form the final output for this parameter
                    if final_output is None:
                        final_output = output_grad + output_no_grad
                    else:
                        final_output += output_grad + output_no_grad

            # Call the original forward method
            output = self._original_forward(input)

            return output

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
                    threshold = np.percentile(np.abs(gradient), 30)
                    gradient_dict = self.client_current.unflatten_model(gradient)

                    loaded_state_dict = torch.load(saved_state_dict_buffer)
                    self.client_current.load_state_dict(loaded_state_dict)

                    for name, layer in self.client_current.named_modules():
                        params = list(layer.named_parameters(recurse=False))
                        param_names = []
                        for param_name, param in params:
                            full_param_name = f'{name}.{param_name}' if name else param_name
                            if full_param_name in gradient_dict:
                                mask = torch.tensor(np.abs(gradient_dict[full_param_name]) < threshold)
                                indices_requires_grad = torch.nonzero(~mask)
                                indices_no_grad = torch.nonzero(mask)

                                param_requires_grad = nn.Parameter(param * mask, requires_grad=True)
                                param_no_grad = (param * ~mask).clone().detach()

                                # 删除原始参数
                                if hasattr(layer, param_name):
                                    delattr(layer, param_name)
                                    param_names.append(param_name)

                                # 存储拆分的参数和索引
                                setattr(layer, f'param_requires_grad_{param_name}', param_requires_grad)
                                setattr(layer, f'param_no_grad_{param_name}', param_no_grad)
                                setattr(layer, f'original_shape_{param_name}', param.shape)

                                # 保存原始的 forward 方法
                                if not hasattr(layer, '_original_forward'):
                                    layer._original_forward = layer.forward

                                # 如果还没有绑定 modified_forward 方法，则绑定
                                if not hasattr(layer, 'modified_forward_bound'):
                                    layer.forward = types.MethodType(modified_forward, layer)
                                    layer.modified_forward_bound = True
                            setattr(layer, f'original_param_names', param_names)

                    trainable_parameters = list(
                        filter(lambda p: p.requires_grad, self.client_current.parameters())
                    )
                    if trainable_parameters:
                        optimizer = eval(self.optimizer)(trainable_parameters, **self.optim_config)
                    else:
                        print("No parameters require gradients after freezing.")
                        optimizer = None

                current_batch += 1
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

            print(self.client_current.fc1.param_requires_grad_weight.grad)

            self.restore_model_param_freeze(self.client_current)
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
