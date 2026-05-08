import os
from copy import deepcopy
from shutil import move

import torch
import numpy as np

from tqdm import tqdm
from itertools import accumulate

phar_num_list = [1, 1, 1, 4, 6, 5, 2, 7]


def l1_regularization(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    return l1_lambda * l1_norm


class Trainer_pharmaQA():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, phar_loss_fn, align_loss_fn, evaluator, phar_evaluator,
                 result_tracker, summary_writer, device_id,  label_mean=None, label_std=None, ddp=False,
                 local_rank=0, scaler=None):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.phar_loss_fn = phar_loss_fn
        self.align_loss_fn = align_loss_fn
        self.evaluator = evaluator
        self.phar_evaluator = phar_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device_id
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.save_path = args.save_path
        self.train_kpgt = args.train_kpgt
        self.scaler = scaler
        
        if args.use_amp and (not ddp or local_rank == 0):
            print("Using Automatic Mixed Precision Training")

    def _forward_epoch(self, model, batched_data,text_dict=None):
        (smiles, g, ecfp, md, labels, phar_targets, phar_target_mx, atom_phar_target_map, smiles_embed, smiles_mask) = batched_data

        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        phar_targets = phar_targets.to(self.device)
        atom_phar_target_map = atom_phar_target_map.to(self.device)
        phar_target_mx = phar_target_mx.to(self.device)
        smiles_embed = smiles_embed.to(self.device) 
        smiles_mask = smiles_mask.to(self.device)
        # if not 'question_text_embeddings' in list(text_dict.keys()):
        #     question_tokens = text_dict['question_tokens'].to(self.device)
        #     question_masks = text_dict['question_masks'].to(self.device)
        #     question_text_embeddings = model.text_model(input_ids=question_tokens, attention_mask=question_masks,if_eval=False)
        #     text_dict['question_text_embeddings'] = question_text_embeddings
        # if not 'discription_text_embeddings' in list(text_dict.keys()):
        #     discription_tokens = text_dict['know_tokens'].to(self.device)
        #     discription_masks = text_dict['discription_mask'].to(self.device)
        #     discription_text_embeddings = model.text_model(input_ids=discription_tokens, attention_mask=discription_masks,if_eval=False)
        #     text_dict['discription_text_embeddings'] = discription_text_embeddings
        if not self.args.ablation_mode_flag:
            predictions, pred_phar_num, atten = model.forward_pharmaPrompt(g, ecfp, md, text=text_dict, smiles_embed=smiles_embed, smiles_mask=smiles_mask)
        else:
            predictions, pred_phar_num, atten = model.forward_pharmaPrompt_ablation(g, ecfp, md, text=text_dict, smiles_embed=smiles_embed, smiles_mask=smiles_mask,\
                phar_targets=phar_targets,ablation_mode=self.args.ablation_mode)
        return predictions, labels, pred_phar_num,phar_targets, atten,phar_target_mx


    def train_epoch(self, model, train_loader, epoch_idx, text_dict=None):
        model.train()
        epoch_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            if batch_idx % self.args.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
                
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                if not self.train_kpgt:
                    predictions, labels, pred_phar_num, phar_targets, atten, phar_target_mx = self._forward_epoch(model, batched_data, text_dict)
                else:
                    predictions, labels = self._forward_epoch(model, batched_data)
                
                is_labeled = (~torch.isnan(labels)).to(torch.float32)
                labels = torch.nan_to_num(labels)
                if (self.label_mean is not None) and (self.label_std is not None):
                    labels = (labels - self.label_mean) / self.label_std
                pre_loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
                
                if self.args.ablation_mode_flag:
                    phar_loss = 0
                    align_loss = 0
                else:
                    if self.args.use_phar_loss:
                        phar_loss = self.phar_loss_fn(pred_phar_num, phar_targets).mean()
                    else:
                        phar_loss = 0
                    if self.args.use_align_loss:
                        align_loss = self.align_loss_fn(atten, phar_target_mx)
                    else:
                        align_loss = 0
                loss = pre_loss + self.args.alpha * phar_loss + self.args.beta * align_loss

                    
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    self.optimizer.step()
            
            self.lr_scheduler.step()
            
            epoch_loss += loss.item()

            if self.summary_writer is not None and batch_idx % 100 == 0:
                global_step = epoch_idx * len(train_loader) + batch_idx
                self.summary_writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.summary_writer.add_scalar('Train/PreLoss', pre_loss.item(), global_step)
                self.summary_writer.add_scalar('Train/LearningRate', 
                    self.optimizer.param_groups[0]['lr'], global_step)
                
                if not self.train_kpgt and not self.args.ablation_mode_flag:
                    self.summary_writer.add_scalar('Train/PharLoss', phar_loss.item(), global_step)
                    self.summary_writer.add_scalar('Train/AlignLoss', align_loss.item(), global_step)

        return epoch_loss / len(train_loader)

    def fit(self, model, train_loader, val_loader, test_loader, text_dict=None,text_model=None):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0

        for epoch in tqdm(range(1, self.args.n_epochs + 1)):

            self.train_epoch(model, train_loader, epoch, text_dict)
            if self.local_rank == 0:

                if val_loader == None:
                    train_result,_,_ = self.eval(model, train_loader, epoch, text_dict)
                    val_result = 0
                    ref_result = train_result
                    best_ref_result = best_train_result
                else:
                    val_result,_,_ = self.eval(model, val_loader, epoch, text_dict)
                    train_result = 0
                    ref_result = val_result
                    best_ref_result = best_val_result

                if test_loader == None:
                    test_result = 0
                else:
                    test_result,predictions_all,labels_all = self.eval(model, test_loader, epoch, text_dict)
                if self.result_tracker.update(best_ref_result, ref_result):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result

                    best_epoch = epoch

                    self.save_model(model)
                    if (self.label_mean is not None) and (self.label_std is not None):
                        predictions_all = predictions_all * self.label_std.detach().cpu() + self.label_mean.detach().cpu()
                    np.savetxt(f'{self.args.save_path}/Results_epoch_best.txt',
                                np.concatenate([predictions_all, labels_all], axis=1),
                                fmt='%.4f',
                                header='Predictions,Labels')

                if self.summary_writer is not None:
                    if ',' in self.args.metric:
                        for index, metric in enumerate(self.args.metric.split(',')):
                            # self.summary_writer.add_scalar(f'Results/train_{metric}', train_result[index], epoch)
                            self.summary_writer.add_scalar(f'Results/val_{metric}', val_result[index], epoch)
                            self.summary_writer.add_scalar(f'Results/test_{metric}', test_result[index], epoch)

                    else:
                        # self.summary_writer.add_scalar(f'Results/train_{self.args.metric}', train_result, epoch)
                        self.summary_writer.add_scalar(f'Results/val_{self.args.metric}', val_result, epoch)
                        self.summary_writer.add_scalar(f'Results/test_{self.args.metric}', test_result, epoch)

                    self.summary_writer.add_scalar('Results/best_epoch', best_epoch, epoch)

                if epoch - best_epoch >= 20:
                    if (self.label_mean is not None) and (self.label_std is not None):
                        predictions_all = predictions_all * self.label_std.detach().cpu() + self.label_mean.detach().cpu()
                    np.savetxt(f'{self.args.save_path}/Results_epoch_{epoch}.txt',
                               np.concatenate([predictions_all, labels_all], axis=1),
                               fmt='%.4f',
                               header='Predictions,Labels')
                    self.save_model(model)
                    break


        return best_train_result, best_val_result, best_test_result

    def eval(self, model, dataloader, epoch, text_dict=None):
        model.eval()
        with torch.no_grad():
            predictions_all = []
            labels_all = []
            for batch_index, batched_data in enumerate(dataloader):
                if not self.train_kpgt: 
                    predictions, labels, pred_phar_num, phar_targets, atten, phar_target_mx = self._forward_epoch(model, batched_data, text_dict)
                else:
                    predictions, labels = self._forward_epoch(model, batched_data)
                predictions_all.append(predictions.detach().cpu())
                labels_all.append(labels.detach().cpu())

            if isinstance(self.evaluator, list):
                result = []
                for evaluator in self.evaluator:
                    result.append(evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all)))
            else:
                result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))

        return result,torch.cat(predictions_all), torch.cat(labels_all)

    def save_model(self, model):
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
            
        save_path = self.args.save_path + "/best_model.pth"
        torch.save(model_to_save.state_dict(), save_path)
        
    def load_best_model(self, model):
        best_model_path = self.args.save_path + "/best_model.pth"
        if os.path.exists(best_model_path):
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(best_model_path))
            else:
                model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for final evaluation")
