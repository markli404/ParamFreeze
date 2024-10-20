from src.server import Server
from utils.Printer import Printer
from datetime import datetime
import numpy as np
import pandas
import argparse
from src.models import *
from utils.utils import *




def main():
    parser = argparse.ArgumentParser(description='federated learning')

    # 客户端相关参数
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--number_of_training_samples', type=int, default=512)
    parser.add_argument('--number_of_testing_samples', type=int, default=128)

    # 数据集相关
    parser.add_argument('--dataset', type=str, default='Fashionmnist')

    # 实验设定相关
    parser.add_argument('--model_name', type=str, default='CNN2')
    parser.add_argument('--run_type', type=str, default='fedavg')
    parser.add_argument('--update_type', type=str, default='fedavg') #param_freeze
    # parser.add_argument('--mu', type=str, default='0.01')
    parser.add_argument('--number_of_clients', type=int, default=20)
    parser.add_argument('--number_of_selected_classes', type=int, default=10)
    parser.add_argument('--round', type=int, default=50)

    # 程序相关
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--name', type=str, default='test')

    # 解析参数
    args = parser.parse_args()
    printer = Printer(args.name)

    res = {}
    accuracy = []
    uploads = []
    for i in range(args.repeat):
        # setup tensorboard and logging printer
        printer.print("\n[WELCOME] ")
        tensorboard_writer = printer.get_tensorboard_writer()

        # initialize federated learning
        central_server = Server(tensorboard_writer)
        central_server.setup(model_name=args.model_name,
                             number_of_clients=args.number_of_clients,
                             number_of_selected_classes=args.number_of_selected_classes,
                             dataset=args.dataset,
                             number_of_training_samples=args.number_of_training_samples,
                             number_of_testing_samples=args.number_of_testing_samples,
                             batch_size=args.batch_size,
                             local_epoch=args.epoch,
                             total_rounds=args.round,
                             run_type=args.run_type,
                             update_type=args.update_type,
                             save=args.save)

        # do federate learning
        accu, round_uploads = central_server.fit()
        accuracy.append(accu)
        uploads.append(round_uploads)

    accuracy = np.array(accuracy)
    accuracy = np.mean(accuracy, axis=0)
    uploads = np.array(uploads)
    uploads = np.mean(uploads, axis=0)
    res['f={}'.format(np.mean(uploads))] = accuracy
    res['uploads_f={}'.format(np.mean(uploads))] = uploads
    printer.print(accuracy)
    printer.print(uploads)

    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if args.save:
        df = pandas.DataFrame(res)
        df.to_csv('{}_{}_class={}_with_pruning_le={}_time={}_runtime={}.csv'.format(
          args.dataset, args.run_type, args.number_of_selected_clients,
          args.epoch,now_time,args.repeat), index=False)
    # bye!
    printer.print("...done all learning process!\n...exit program!")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    main()