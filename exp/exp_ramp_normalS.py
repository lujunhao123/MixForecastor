from data_provider.ramp_data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,cal_class_metric
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.multi_task import pcgrad_fn,pareto_fn
warnings.filterwarnings('ignore')


class Exp_Ramp_normal_S(Exp_Basic):
    def __init__(self, args):
        super(Exp_Ramp_normal_S, self).__init__(args)
        self.args = args

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        task_dis_criterion = nn.CrossEntropyLoss()
        task_con_criterion = nn.MSELoss()
      
        return task_dis_criterion, task_con_criterion

    def vali(self, vali_data, vali_loader, task_dis_criterion,task_con_criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_dis, batch_x_con, batch_y_dis, batch_y_con, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x_dis = batch_x_dis.float().to(self.device).transpose(-1, -2)
                batch_x_con = batch_x_con.float().to(self.device).transpose(-1, -2)
                batch_y_dis = batch_y_dis.long().to(self.device).transpose(-1, -2)
                batch_y_con = batch_y_con.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)
                batch_y_mark = batch_y_mark.float().to(self.device).transpose(-1, -2)

                # decoder input
                dec_inp_con = torch.zeros_like(batch_y_con[:, :, -self.args.pred_len:]).float()
                dec_inp_con = torch.cat([batch_y_con[:, :, :self.args.label_len], dec_inp_con], dim=2).float().to(self.device)

                dec_inp_dis = torch.zeros_like(batch_y_dis[:, :, -self.args.pred_len:]).float()
                dec_inp_dis = torch.cat([batch_y_dis[:, :, :self.args.label_len], dec_inp_dis], dim=2).float().to(self.device)

                batch_y_dis = batch_y_dis[:, :, -self.args.pred_len:].to(self.device)
                batch_y_con = batch_y_con[:, :, -self.args.pred_len:].to(self.device)

                x_dis_pe = None


                output = self.model(batch_x_dis, batch_x_con, batch_x_mark, dec_inp_dis, dec_inp_con, batch_y_mark)
                # x_dis_forecast==【B,D,T,3】 x_con_forecast==【B,D,T】
                #dis_fore_loss = task_dis_criterion(x_dis_forecast.reshape(-1, 3),
                #                                   batch_y_dis.reshape(-1, 1).squeeze(-1).long())
                #con_fore_loss = task_con_criterion(x_con_forecast, batch_y_con)

                if self.args.Singel_task == "Con":
                    loss = task_con_criterion(output, batch_y_con)
                elif self.args.Singel_task == "Dis":
                    loss = task_dis_criterion(output.reshape(-1, 3),batch_y_dis.reshape(-1, 1).squeeze(-1).long())

                total_loss.append(loss.detach().cpu().numpy())

        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss


    def train(self, setting):


        smooth_arr = torch.zeros((self.args.seq_len - 1, self.args.seq_len))
        for i in range(self.args.seq_len - 1):
            smooth_arr[i, i] = -1
            smooth_arr[i, i + 1] = 1

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        task_dis_criterion, task_con_criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_loss_list = []

        dis_forecast_loss_list = []
        con_forecast_loss_list = []
        dis_forecast_acc_list = []




        for epoch in range(self.args.train_epochs):
            iter_count = 0

            train_loss_l = []
            dis_forecast_loss_l = []
            con_forecast_loss_l = []
            dis_forecast_acc_l = []


            self.model.train()
            epoch_time = time.time()
            for i, (batch_x_dis, batch_x_con, batch_y_dis, batch_y_con, batch_x_mark, batch_y_mark) in enumerate(
                    train_loader):
                batch_x_dis = batch_x_dis.float().to(self.device).transpose(-1, -2)
                batch_x_con = batch_x_con.float().to(self.device).transpose(-1, -2)
                batch_y_dis = batch_y_dis.float().to(self.device).transpose(-1, -2)
                batch_y_con = batch_y_con.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)
                batch_y_mark = batch_y_mark.float().to(self.device).transpose(-1, -2)

                # decoder input
                dec_inp_con = torch.zeros_like(batch_y_con[:, :, -self.args.pred_len:]).float()
                dec_inp_con = torch.cat([batch_y_con[:, :, :self.args.label_len], dec_inp_con], dim=2).float().to(self.device)

                dec_inp_dis = torch.zeros_like(batch_y_dis[:, :, -self.args.pred_len:]).float()
                dec_inp_dis = torch.cat([batch_y_dis[:, :, :self.args.label_len], dec_inp_dis], dim=2).float().to(self.device)

                batch_y_dis = batch_y_dis[:, :, -self.args.pred_len:].to(self.device)
                batch_y_con = batch_y_con[:, :, -self.args.pred_len:].to(self.device)



                iter_count += 1
                
                output = self.model(batch_x_dis, batch_x_con, batch_x_mark, dec_inp_dis, dec_inp_con, batch_y_mark)
                if self.args.Singel_task == "Con":
                    loss = task_con_criterion(output, batch_y_con)
                elif self.args.Singel_task == "Dis":
                    loss = task_dis_criterion(output.reshape(-1, 3),batch_y_dis.reshape(-1, 1).squeeze(-1).long())


                #dis_fore_acc = (batch_y_dis == x_dis_forecast.argmax(dim=-1)).float().mean().detach() * 100
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                # if self.args.optype != "ConFIG":
                #     output = self.model(batch_x_dis, batch_x_con, batch_x_mark, dec_inp_dis, dec_inp_con, batch_y_mark)
                #     if self.args.Singel_task == "Con":
                #         loss = task_con_criterion(output, batch_y_con)
                #     elif self.args.Singel_task == "Dis":
                #         loss = task_dis_criterion(output.reshape(-1, 3),batch_y_dis.reshape(-1, 1).squeeze(-1).long())


                #     #dis_fore_acc = (batch_y_dis == x_dis_forecast.argmax(dim=-1)).float().mean().detach() * 100
                #     model_optim.zero_grad()
                #     loss.backward()
                #     model_optim.step()
                # x_dis_forecast==【B,D,T,3】 x_con_forecast==【B,D,T】
                #dis_fore_loss = task_dis_criterion(x_dis_forecast.reshape(-1, 3),
                #                                   batch_y_dis.reshape(-1, 1).squeeze(-1).long())
                #con_fore_loss = task_con_criterion(x_con_forecast, batch_y_con)


                    # if self.args.optype == "Human":
                    #     loss =  0.2 * dis_fore_loss+  0.8 * con_fore_loss


                    #     model_optim.zero_grad()
                    #     loss.backward()
                    #     model_optim.step()

                    # elif self.args.optype == "Pareto":
                    #     number_task = 2
                    #     loss_list = [dis_fore_loss, con_fore_loss]
                    #     w_list = [0.5, 0.5]
                    #     c_list = [0.2, 0.2]
                    #     new_w_list = pareto_fn(w_list, c_list, self.model, loss_list, number_task)
                    #     loss = sum(new_w_list[i] * loss_list[i] for i in range(len(w_list)))

                    #     model_optim.zero_grad()
                    #     loss.backward()
                    #     model_optim.step()

                    # elif self.args.optype == "Pcgrad":
                    #     number_task = 2
                    #     loss_list = [dis_fore_loss, con_fore_loss]
                    #     loss = pcgrad_fn(self.model, loss_list, model_optim)
                    #     model_optim.step()

                    # else:
                    #     raise NotImplementedError
                # elif self.args.optype == "ConFIG":
                #     from conflictfree.grad_operator import ConFIGOperator
                #     from conflictfree.utils import get_gradient_vector,apply_gradient_vector
                #     operator=ConFIGOperator() # initialize operator
                #     loss_type = ['continue', 'discrete']
                #     grads=[]
                #     loss_all = []
                #     max_len = 0
                #     for loss_fn in loss_type:
                #         model_optim.zero_grad()
                #         loss_i, dis_fore_acc =self.ConFIG_optimizer(loss_fn, input_i=(batch_x_dis, batch_x_con, batch_x_mark, dec_inp_dis, dec_inp_con, batch_y_mark), true_i=(batch_y_con, batch_y_dis))
                #         loss_i.backward()
                #         loss_all.append(loss_i)
                #         grad_vec = get_gradient_vector(self.model)
                #         grads.append(grad_vec)
                #         # 第二次遍历，将所有梯度向量补齐到相同长度
                #         max_len = max(max_len, grad_vec.numel())
                #         grads = [torch.cat([g, g.new_zeros(max_len - g.numel())]) for g in grads]
                #         g_config=operator.calculate_gradient(grads) # calculate the conflict-free direction
                #         apply_gradient_vector(self.model,g_config) # or simply use `operator.update_gradient(network,grads)` to calculate and set the conflict-free direction to the network
                #         model_optim.step()
                #         if loss_fn == 'continue':
                #             con_fore_loss = loss_all[-1]
                #         elif loss_fn == 'discrete':
                #             dis_fore_loss = loss_all[-1]
                #     loss =  0.5 * loss_all[1]+  0.5 * loss_all[0]
                    

                train_loss_l.append(loss.item())
                # dis_forecast_loss_l.append(dis_fore_loss.item())
                # con_forecast_loss_l.append(con_fore_loss.item())
                # dis_forecast_acc_l.append(dis_fore_acc.item())


                if (i + 1) % 10 == 0:
                    # print("\titers: {0}, epoch: {1} | loss_all: {2:.7f} |dis_rec_loss: {2:.7f} |con_rec_loss: {2:.7f} ".format(i + 1,epoch + 1, loss.item(),dis_rec_loss.item(),con_rec_loss.item()))
                    print(
                        "iters: {0}, epoch: {1} | loss_all: {2:.7f} ".format(
                            i + 1, epoch + 1, loss.item()))
                    # print('Con_forecasting_loss = %f' % con_fore_loss.item())
                    # print('Dis_forecasting_loss = %f' % dis_fore_loss.item())
                    # print('Dis_forecasting_acc = %f' % dis_fore_acc)




                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss_l = np.average(train_loss_l)
            # dis_forecast_loss_l = np.average(dis_forecast_loss_l)
            # con_forecast_loss_l = np.average(con_forecast_loss_l)
            # dis_forecast_acc_l = np.average( dis_forecast_acc_l)


            vali_loss = self.vali(vali_data, vali_loader, task_dis_criterion,task_con_criterion)
            test_loss = self.vali(test_data, test_loader, task_dis_criterion,task_con_criterion)



            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f} Vali Loss: {3:.5f} Test Loss: {4:.5f} "
                    .format(epoch + 1, train_steps, train_loss_l, vali_loss,test_loss))

            early_stopping(vali_loss, self.model, path)

            # early_stopping(999, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # train_loss_list.append(train_loss_l)
            # dis_forecast_loss_list.append(dis_forecast_loss_l)
            # con_forecast_loss_list.append(con_forecast_loss_l)
            # dis_forecast_acc_list.append(dis_forecast_acc_l)



#            if self.args.lradj_flag:

#                      learning rate update
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 将所有列表转换为 NumPy 数组
        # np_train_loss = np.array(train_loss_list)
        # np_dis_forecast_loss = np.array(dis_forecast_loss_list)
        # np_con_forecast_loss = np.array(con_forecast_loss_list)
        # np_dis_forecast_acc = np.array(dis_forecast_acc_list)

        # 保存到 .npz 文件
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # filename = "training_metrics.npz"  # 文件名
        # np.savez(
        #     folder_path + filename,
        #     # train_loss=np_train_loss,
        #     # dis_forecast_loss=np_dis_forecast_loss,
        #     # con_forecast_loss=np_con_forecast_loss,
        #     # dis_forecast_acc=np_dis_forecast_acc
        # )


        return self.model
    
    def test_con(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds_dis = []
        trues_dis = []
        preds_con = []
        trues_con = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_dis, batch_x_con, batch_y_dis, batch_y_con, batch_x_mark, batch_y_mark) in enumerate(
                    test_loader):
                batch_x_dis = batch_x_dis.float().to(self.device).transpose(-1, -2)
                batch_x_con = batch_x_con.float().to(self.device).transpose(-1, -2)
                batch_y_dis = batch_y_dis.float().to(self.device).transpose(-1, -2)
                batch_y_con = batch_y_con.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)
                batch_y_mark = batch_y_mark.float().to(self.device).transpose(-1, -2)

                # decoder input
                dec_inp_con = torch.zeros_like(batch_y_con[:, :, -self.args.pred_len:]).float()
                dec_inp_con = torch.cat([batch_y_con[:, :, :self.args.label_len], dec_inp_con], dim=2).float().to(self.device)

                dec_inp_dis = torch.zeros_like(batch_y_dis[:, :, -self.args.pred_len:]).float()
                dec_inp_dis = torch.cat([batch_y_dis[:, :, :self.args.label_len], dec_inp_dis], dim=2).float().to(self.device)

                batch_y_dis = batch_y_dis[:, :, -self.args.pred_len:].to(self.device)
                batch_y_con = batch_y_con[:, :, -self.args.pred_len:].to(self.device)

                x_dis_pe = None


                output = self.model(batch_x_dis, batch_x_con, batch_x_mark, dec_inp_dis, dec_inp_con, batch_y_mark)

                pred_con = output.detach().cpu().numpy()
                true_con = batch_y_con.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    pred_con = test_data.inverse_transform(pred_con)
                    true_con = test_data.inverse_transform(true_con)

                preds_con.append(pred_con)
                trues_con.append(true_con)

                #if i % 20 == 0 and self.args.forecast_visual_flag:
                #    input = batch_x_con.detach().cpu().numpy()
                #    gt = np.concatenate((input[0,  -3,:], true_con[0,  -3,:]), axis=0)
                #    pd = np.concatenate((input[0,  -3,:], pred_con[0,  -3,:]), axis=0)
                #    #visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds_con = np.concatenate(preds_con,axis=0)
        trues_con = np.concatenate(trues_con,axis=0)
        preds_con = preds_con.reshape(-1, preds_con.shape[-2], preds_con.shape[-1])
        trues_con = trues_con.reshape(-1, trues_con.shape[-2], trues_con.shape[-1])
        print('test con shape:', preds_con.shape, trues_con.shape)

        mae, mse, rmse, mape, mspe = metric(preds_con, trues_con)
        print('mse:{}, mae:{}'.format(mse, mae))

        #mae, mse, rmse, mape, mspe,accuracy
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

  
        np.save(folder_path + 'pred_con.npy', preds_con)
        np.save(folder_path + 'true_con.npy', trues_con)

        return mae, mse, None
    
    def test_dis(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds_dis = []
        trues_dis = []
        preds_con = []
        trues_con = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_dis, batch_x_con, batch_y_dis, batch_y_con, batch_x_mark, batch_y_mark) in enumerate(
                    test_loader):
                batch_x_dis = batch_x_dis.float().to(self.device).transpose(-1, -2)
                batch_x_con = batch_x_con.float().to(self.device).transpose(-1, -2)
                batch_y_dis = batch_y_dis.float().to(self.device).transpose(-1, -2)
                batch_y_con = batch_y_con.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)
                batch_y_mark = batch_y_mark.float().to(self.device).transpose(-1, -2)

                # decoder input
                dec_inp_con = torch.zeros_like(batch_y_con[:, :, -self.args.pred_len:]).float()
                dec_inp_con = torch.cat([batch_y_con[:, :, :self.args.label_len], dec_inp_con], dim=2).float().to(self.device)

                dec_inp_dis = torch.zeros_like(batch_y_dis[:, :, -self.args.pred_len:]).float()
                dec_inp_dis = torch.cat([batch_y_dis[:, :, :self.args.label_len], dec_inp_dis], dim=2).float().to(self.device)

                batch_y_dis = batch_y_dis[:, :, -self.args.pred_len:].to(self.device)
                batch_y_con = batch_y_con[:, :, -self.args.pred_len:].to(self.device)

                x_dis_pe = None


                output = self.model(batch_x_dis, batch_x_con, batch_x_mark, dec_inp_dis, dec_inp_con, batch_y_mark)


                #if i % 20 == 0 and self.args.forecast_visual_flag:
                #    input = batch_x_con.detach().cpu().numpy()
                #    gt = np.concatenate((input[0,  -3,:], true_con[0,  -3,:]), axis=0)
                #    pd = np.concatenate((input[0,  -3,:], pred_con[0,  -3,:]), axis=0)
                #    #visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


                pred_dis = output.detach().cpu()
                true_dis = batch_y_dis.detach().cpu()
                preds_dis.append(pred_dis)
                trues_dis.append(true_dis)


        preds_dis = torch.cat(preds_dis, 0)
        trues_dis = torch.cat(trues_dis, 0)
        probs = torch.nn.functional.softmax(preds_dis.reshape(-1,3))  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        accuracy, precision, recall, f1 = cal_class_metric(predictions, trues_dis.flatten().cpu().numpy())
        #mae, mse, rmse, mape, mspe,accuracy
        #print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
        print('accuracy:{}, precision:{}, recall:{}, f1:{}'.format(accuracy, precision, recall, f1))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        #f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
        f.write('accuracy:{}, precision:{}, recall:{}, f1:{}'.format(accuracy, precision, recall, f1))
        f.write('\n')
        f.write('\n')
        f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #print(preds_con.shape,trues_con.shape,preds_dis.data.cpu().numpy().shape,trues_dis.data.cpu().numpy().shape)
        #np.save(folder_path + 'pred_con.npy', preds_con)
        #np.save(folder_path + 'true_con.npy', trues_con)
        np.save(folder_path + 'pred_dis.npy', preds_dis.data.cpu().numpy())
        np.save(folder_path + 'true_dis.npy', trues_dis.data.cpu().numpy())
        return None,None, accuracy


    def test(self, setting, test=0):
        if self.args.Singel_task == "Con":
            mae, mse, accuracy = self.test_con(setting=setting,test=test)
        elif self.args.Singel_task == "Dis":
            mae, mse, accuracy = self.test_dis(setting=setting,test=test)
        return mae, mse, accuracy
    

