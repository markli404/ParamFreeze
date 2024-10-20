import numpy as np
import logging
import copy
import gc
import operator
# custom packages

from utils.Printer import Printer
import torch
from collections import OrderedDict
import time
from torch.nn import CosineSimilarity
from scipy import spatial
from sklearn import metrics
from utils.utils import pretty_list


class CommunicationController:
    def __init__(self, clients, upload_chance=0.3):
        self.weight = None
        self.improvement = None
        self.num_clients = len(clients)
        self.clients = clients
        self.upload_chance = upload_chance

        self.cos = CosineSimilarity(dim=0, eps=1e-6)
        self.sampled_clients_indices = None
        # FedPNS
        self.test_count = np.zeros(len(clients))

    @staticmethod
    def calculate_similarity(g1, g2):
        return spatial.distance.cosine(g1, g2)

    @staticmethod
    def l2_norm(gradient):
        res = gradient * gradient
        res = np.sum(res)
        return np.sqrt(res)

    @staticmethod
    def l1_norm(gradient):
        res = [abs(num) for num in gradient]
        res = np.sum(res)
        return res

    def sample_all_clients(self):
        sampled_client_indices = list(range(self.num_clients))
        self.sampled_clients_indices = sampled_client_indices
        message = "All clients are selected"

        return message, sampled_client_indices

    def sample_clients_random(self):
        num_sampled_clients = max(int(self.upload_chance * self.num_clients), 1)
        sampled_client_indices = np.random.choice(self.num_clients, num_sampled_clients, replace=False).tolist()

        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update."

        return message, sampled_client_indices

    def sample_clients_fed_pns(self):
        def average(grad_all):

            value_list = list(grad_all.values())

            w_avg = copy.deepcopy(value_list[0])
            # print(type(w_avg))
            for i in range(1, len(value_list)):
                w_avg += value_list[i]
            return w_avg / len(value_list)

        def client_deleting(expect_list, expect_value, selected_clients, local_gradients):
            for i in range(len(selected_clients)):
                worker_ind_del = [n for n in selected_clients if n != selected_clients[i]]
                grad_del = local_gradients.copy()
                grad_del.pop(selected_clients[i])
                avg_grad_del = average(grad_del)
                grad_del['avg_grad'] = avg_grad_del
                expect_value_del = get_relation(grad_del, worker_ind_del)
                expect_list[selected_clients[i]] = expect_value_del
            expect_list['all'] = expect_value

            return expect_list

        def get_relation(avg_grad, idxs_users):
            def dot_sum(K, L):
                return round(sum(i[0] * i[1] for i in zip(K.numpy(), L.numpy())), 2)

            innnr_value = {}

            for i in range(len(idxs_users)):
                innnr_value[idxs_users[i]] = dot_sum(avg_grad[idxs_users[i]], avg_grad['avg_grad'])

            return round(sum(list(innnr_value.values())), 3)

        def test_part(clients, selected_clients, test_set, key):
            model = FedAvg(clients, selected_clients)
            _, loss_all = model_evaluation_simple(model, test_set)
            selected_clients.remove(key)
            model = FedAvg(clients, selected_clients)
            _, loss_part = model_evaluation_simple(model, test_set)
            return loss_all, loss_part

        def model_evaluation_simple(model, test_set):
            # calculate the sample distribution of all clients
            device = 'cuda'
            model.eval()
            model.to(device)

            test_loss, correct = 0, 0
            with torch.no_grad():
                for data, labels in test_set.get_dataloader():
                    data, labels = data.float().to(device), labels.long().to(device)
                    outputs = model(data)
                    test_loss += eval(config.CRITERION)()(outputs, labels).item()

                    predicted = outputs.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(labels.view_as(predicted)).sum().item()

                    if device == "cuda": torch.cuda.empty_cache()
            model.to("cpu")

            # calculate the metrics
            test_loss = test_loss / len(test_set.get_dataloader())
            test_accuracy = correct / len(test_set)
            return test_accuracy, test_loss

        def FedAvg(clients, selected_clients):
            fedavg_coeff = [len(clients[idx]) for idx in selected_clients]
            fedavg_coeff = np.array(fedavg_coeff) / sum(fedavg_coeff)

            new_model = copy.deepcopy(clients[0].client_current)
            averaged_weights = OrderedDict()

            for it, idx in enumerate(selected_clients):
                local_weights = clients[idx].client_current.state_dict()
                for key in new_model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = fedavg_coeff[it] * local_weights[key]
                    else:
                        averaged_weights[key] += fedavg_coeff[it] * local_weights[key]

            new_model.load_state_dict(averaged_weights)
            return new_model

        test_set = None
        for client in self.clients:
            if test_set is None:
                test_set = copy.deepcopy(client.test)
            else:
                test_set + client.test

        selected_clients = list(range(self.num_clients))

        st = time.time()

        local_gradients = {}
        for client in self.clients:
            local_gradients[client.id] = client.get_gradient()
        local_gradients['avg_grad'] = average(local_gradients)
        max_now = get_relation(local_gradients, selected_clients)
        local_gradients.pop('avg_grad')

        et = time.time()
        elapsed_time = et - st
        print('Get gradients:', elapsed_time, 'seconds')

        expect_list = {}
        labeled = []

        num_sampled_clients = max(int(self.upload_chance * self.num_clients), 1)
        while len(selected_clients) > num_sampled_clients:
            st = time.time()
            expect_list = client_deleting(expect_list, max_now, selected_clients, local_gradients)
            # print(len(w_locals), expect_list)
            copy_expect_list = copy.deepcopy(expect_list)
            copy_expect_list.pop('all')
            key = max(copy_expect_list.items(), key=operator.itemgetter(1))[0]
            et = time.time()
            elapsed_time = et - st
            print('Expect_list:', elapsed_time, 'seconds')

            # if expect_list[key] <= expect_list["all"]:
            #     break
            # else:
            #     labeled.append(key)
            #     expect_list.pop("all")
            #     loss_all, loss_pop = test_part(self.clients, selected_clients, test_set, key)
            #
            #     if loss_all < loss_pop:
            #         selected_clients.append(key)
            #         break
            #     else:
            #         local_gradients.pop(key)
            #         max_now = expect_list[key]
            #         expect_list.pop(key)


            labeled.append(key)
            expect_list.pop("all")
            loss_all, loss_pop = test_part(self.clients, selected_clients, test_set, key)

            local_gradients.pop(key)
            max_now = expect_list[key]
            expect_list.pop(key)

        self.sampled_clients_indices = selected_clients
        message = f"{selected_clients} clients are selected for the next update."

        return message, selected_clients

    def sample_clients(self):
        if self.weight is None:
            self.weight = np.ones(len(self.clients)) / len(self.clients)

        p = np.array(self.weight) / sum(self.weight)
        num_sampled_clients = max(int(self.upload_chance * self.num_clients), 1)
        client_indices = [i for i in range(self.num_clients)]
        sampled_client_indices = sorted(
            np.random.choice(a=client_indices, size=num_sampled_clients, replace=False, p=p).tolist())

        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update with possibility {self.weight[sampled_client_indices]}."

        return message, sampled_client_indices

    def update_selected_clients(self, update_type, all_client=False):
        """Call "client_update" function of each selected client."""
        if all_client:
            self.sampled_clients_indices = np.arange(0, self.num_clients)

        for idx in self.sampled_clients_indices:
            self.clients[idx].client_update(update_type)

        message = f"...{len(self.sampled_clients_indices)} clients are selected and updated"

        return message

    def evaluate_selected_models(self):
        """Call "client_evaluate" function of each selected client."""
        for idx in self.sampled_clients_indices:
            self.clients[idx].client_evaluate()

        message = f"...finished evaluation of {str(self.sampled_clients_indices)} selected clients!"

        return message

    def evaluate_all_models(self):
        """Call "client_evaluate" function of each selected client."""
        for client in self.clients:
            client.client_evaluate()

        message = f"...finished evaluation of {str(self.sampled_clients_indices)} selected clients!"

        return message

    def transmit_model(self, model, to_all_clients=True):
        if to_all_clients:
            target_clients = self.clients
            message = f"...successfully transmitted models to all {str(self.num_clients)} clients!"
        else:
            target_clients = []
            for index in self.sampled_clients_indices:
                target_clients.append(self.clients[index])
            message = f"...successfully transmitted models to {str(len(self.sampled_clients_indices))} selected clients!"

        for target_client in target_clients:
            target_client.global_previous = copy.deepcopy(target_client.global_current)
            target_client.client_previous = copy.deepcopy(target_client.client_current)
            target_client.client_current = copy.deepcopy(model)
            target_client.global_current = copy.deepcopy(model)
            # print('transmit sparse_weights',sparse_weights)
            # target_client.global_sparse_weights = copy.deepcopy(sparse_weights)

        return message



