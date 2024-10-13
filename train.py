import torch
from torch import optim
from tqdm import tqdm
import random
import fitlog
from transformers.optimization import get_linear_schedule_with_warmup

from metrics import get_metrics, get_four_metrics, get_four_metrics_macro_f1

class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()



class DAIE_Trainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, args=None, logger=None,  writer=None) -> None:
        
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.args = args
        self.logger = logger
        self.writer = writer

        self.refresh_step = 2
        self.best_dev_f1 = 0
        self.best_dev_acc = 0
        self.best_dev_precision = 0
        self.best_dev_recall = 0

        self.best_test_f1 = 0
        self.best_test_acc = 0
        self.best_test_precision = 0
        self.best_test_recall = 0
        self.best_test_m_f1 = 0
        self.best_test_m_precision = 0
        self.best_test_m_recall = 0

        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        
        fitlog.set_log_dir("logger_DAIE/")
        fitlog.add_hyper(self.args)
        fitlog.add_hyper_in_file(__file__)

        self.before_train()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  CLIP Learning rate = {}".format(self.args.clip_lr))
        self.logger.info("  MODEL Learning rate = {}".format(self.args.model_lr))
        self.logger.info("  LINEAR Learning rate = {}".format(self.args.linear_lr))
        self.logger.info("  beta = {}".format(self.args.beta))
        self.logger.info("  theta = {}".format(self.args.theta))
        self.logger.info("  gamml = {}".format(self.args.gamml))
        self.logger.info("  imgpri = {}".format(self.args.imgpri))
        self.logger.info("  nsw = {}".format(self.args.nsw))

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            predict, real_label = [], []
            # best_test_f1 = 0
            for epoch in range(1, self.args.num_epochs+1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (ce_loss, logits), labels = self._step(batch, mode="train")
                    
                    loss = ce_loss
                    avg_loss += ce_loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    predict = predict + get_metrics(logits.cpu())
                    real_label = real_label + labels.cpu().numpy().tolist()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                # if epoch >= self.args.eval_begin_epoch:
                #     self.evaluate(epoch)   # generator to dev.

                acc, recall, precision, f1 = get_four_metrics(real_label, predict)
                result = {'train_acc': acc, 'train_precision': precision, 'train_recall': recall, 'train_f1': f1}
                self.logger.info('Train result: {}.'.format(result))

                # dev.
                self.evaluate(epoch=epoch)

                # begin to test.
                self.test(epoch=epoch)
            
            pbar.close()
            self.pbar = None
            # best dev performance
            self.logger.info("Get best dev performance at epoch {}, best dev acc is {:.4f}".format(self.best_dev_epoch, self.best_dev_acc))
            self.logger.info("Get best dev precision:{:.4f}, best dev redall:{:.4f}, best dev f1 score:{:.4f}".format( \
                                        self.best_dev_precision, self.best_dev_recall, self.best_dev_f1))
            # best test performance
            self.logger.info("Get best test performance at epoch {}, best test acc is {:.4f}".format(self.best_test_epoch, self.best_test_acc))
            self.logger.info("Get best test precision:{:.4f}, best test redall:{:.4f}, best test f1 score:{:.4f},".format( \
                                        self.best_test_precision, self.best_test_recall, self.best_test_f1))
            # Macro-Average
            self.logger.info("Get best test m_precision:{:.4f}, best test m_redall:{:.4f}, best test m_f1 score:{:.4f},".format( \
                                        self.best_test_m_precision, self.best_test_m_recall, self.best_test_m_f1))
        self.fitlog()

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")

        step = 0
        predict, real_label = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    predict = predict + get_metrics(logits.cpu())
                    real_label = real_label + labels.cpu().numpy().tolist()

                    pbar.update()
                # evaluate done
                pbar.close()

                acc, recall, precision, f1 = get_four_metrics(real_label, predict)
                result = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
                self.logger.info('Dev result: {}.'.format(result))

                if self.writer:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    # tensorbordx
                self.logger.info("Dve f1 score: {:.4f}, acc: {:.4f}, dve loss: {:.4f}.".format(f1, acc, total_loss/len(self.test_data)))

                # self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}."\
                #             .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                dev_f1 = f1
                if dev_f1 >= self.best_dev_f1:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_f1 = dev_f1 # update best metric(f1 score)
                    self.best_dev_acc = acc
                    self.best_dev_precision = precision
                    self.best_dev_recall = recall
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch=None):
        self.model.eval()
        self.logger.info("***** Running testing *****")
        
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        predict, real_label = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    (loss, logits), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    predict = predict + get_metrics(logits.cpu())
                    real_label = real_label + labels.cpu().numpy().tolist()
                    
                    pbar.update()
                # evaluate done
                pbar.close()

                # Binary-Average
                acc, recall, precision, f1 = get_four_metrics(real_label, predict)
                result = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
                self.logger.info('Test result: {}.'.format(result))
                # Macro-Average
                m_precision, m_recall, m_f1 = get_four_metrics_macro_f1(real_label, predict)
                m_result = {'m_precision': m_precision, 'm_recall': m_recall, 'm_f1': m_f1}
                self.logger.info('Test m_result: {}.'.format(m_result))

                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)    # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1)    # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data))    # tensorbordx
                self.logger.info("Test f1 score: {:.4f}, acc: {:.4f}, test loss: {:.4f}.".format(f1, acc, total_loss/len(self.test_data)))

                test_f1 = f1
                if test_f1 >= self.best_test_f1:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_f1 = test_f1 # update best metric(f1 score)
                    self.best_test_acc = acc
                    self.best_test_precision = precision
                    self.best_test_recall = recall
                    self.best_test_m_precision = m_precision
                    self.best_test_m_recall = m_recall
                    self.best_test_m_f1 = m_f1
                    # if self.args.save_path is not None:
                    #     torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                    #     self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()
        
    def _step(self, batch, mode="train"):
        if mode != "predict":
            imgs_batch, imgs_pri_batch, texts_token, labels, \
                word_spans, word_len, word_mask, dep_se, gnn_mask, np_mask, \
                    img_mask, img_edge_index = batch
            texts_token = {k: v.to(self.args.device) for k, v in texts_token.items()}
            outputs = self.model(texts=texts_token, imgs=imgs_batch, imgs_pri=imgs_pri_batch, labels=labels,\
                                    word_spans=word_spans, word_len=word_len, word_mask=word_mask, \
                                    dep_se=dep_se, gnn_mask=gnn_mask, np_mask=np_mask, \
                                    img_mask=img_mask, img_edge_index=img_edge_index)
            return outputs, labels

    def before_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay) and (
                            'clip_model' in n or
                            'clip_img_nag_model' in n 
                    )], 
                    'weight_decay': 1e-2,
                    'lr': self.args.clip_lr},

                {'params': [p for n, p in self.model.named_parameters() 
                    if any(nd in n for nd in no_decay) and (
                            'clip_model' in n or
                            'clip_img_nag_model' in n
                    )], 
                    'weight_decay': 0.0,
                    'lr': self.args.clip_lr},
                
                
                {'params': [p for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay) and (
                            'text_mlp' in n or
                            'img_mlp' in n or
                            'img_mlp_pri' in n or
                            'cro_model_att' in n or
                            'fusion' in n or
                            'logit_scale1' in n or
                            'logit_scale_nag1' in n
                    )], 
                    'weight_decay': 1e-2,
                    'lr': self.args.model_lr},

                {'params': [p for n, p in self.model.named_parameters() 
                    if any(nd in n for nd in no_decay) and (
                            'text_mlp' in n or
                            'img_mlp' in n or
                            'img_mlp_pri' in n or
                            'cro_model_att' in n or
                            'fusion' in n or
                            'logit_scale1' in n or
                            'logit_scale_nag1' in n
                    )], 
                    'weight_decay': 0.0,
                    'lr': self.args.model_lr},


                {'params': [p for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay) and (
                            'linear_att' in n or
                            'linear_att_pri' in n or
                            'linear_classifier1' in n or
                            'linear_classifier2' in n
                    )], 
                    'weight_decay': 1e-2,
                    'lr': self.args.linear_lr},

                {'params': [p for n, p in self.model.named_parameters() 
                    if any(nd in n for nd in no_decay) and (
                            'linear_att' in n or
                            'linear_att_pri' in n or
                            'linear_classifier1' in n or
                            'linear_classifier2' in n
                    )], 
                    'weight_decay': 0.0,
                    'lr': self.args.linear_lr}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                                num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def fitlog(self):
        fitlog.add_best_metric({"model": self.args.model_name})
        fitlog.add_best_metric({"time_new": self.args.time_new})
        fitlog.add_best_metric({"batch_size": self.args.batch_size})
        fitlog.add_best_metric({"test": {"acc_max": self.best_test_acc}})
        fitlog.add_best_metric({"test": {"precision_max": self.best_test_precision}})
        fitlog.add_best_metric({"test": {"recall_max": self.best_test_recall}})
        fitlog.add_best_metric({"test": {"f1_max": self.best_test_f1}})
        fitlog.add_best_metric({"test": {"m_precision_max": self.best_test_m_precision}})
        fitlog.add_best_metric({"test": {"m_recall_max": self.best_test_m_recall}})
        fitlog.add_best_metric({"test": {"m_f1_max": self.best_test_m_f1}})
        fitlog.finish()

