import torch, torchmetrics, tqdm, copy, time
from utils import LinearLR, unlearn_func, ssd_tuning, distill_kl_loss, compute_accuracy
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools


class Naive():
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.max_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes).cuda()
        self.scaler = GradScaler()


    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        self.model.cuda()


    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss


    def train_one_epoch(self, loader):
        self.model.train()
        self.top1.reset()

        for (images, target, infgt) in tqdm.tqdm(loader):
            images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return


    def eval(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        if save_preds:
            preds, targets = [], []

        with torch.no_grad():
            for (images, target) in tqdm.tqdm(loader):
                with autocast():
                    images, target = images.cuda(), target.cuda()
                    output = self.model(images) if self.prenet is None else self.model(self.prenet(images))
                self.top1(output, target)
                if save_preds:
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = self.top1.compute().item()
        self.top1.reset()
        if not save_preds: print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')
        
        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return


    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.train_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+self.opt.unlearn_method+'_'+self.opt.exp_name
        if self.opt.unlearn_method != 'Naive': 
            self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        return


    def compute_and_save_results(self, train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader, wandb_run=None, method_handle=None, save_model=False):
        self.get_save_prefix()
        print(self.unlearn_file_prefix)
        if not exists(self.unlearn_file_prefix):
            makedirs(self.unlearn_file_prefix)

        if save_model:
            torch.save(self.best_model.state_dict(), self.unlearn_file_prefix+'/model.pth')
        if self.best_model is not None:
            self.model = self.best_model.cuda()

        if wandb_run is not None:
            wandb_run.log({f"{method_handle}_train_top1": self.save_files['train_top1'][-1]})
            wandb_run.log({f"{method_handle}_val_top1s": self.save_files['val_top1'][-1]})
            wandb_run.log({f"{method_handle}_unlearn_time": self.save_files['train_time_taken']})
            wandb_run.log({f"{method_handle}_model_path": self.unlearn_file_prefix+'/model.pth'})
            wandb_run.log({f"{method_handle}_top1_sum": self.save_files['train_top1'][-1] + self.save_files['val_top1'][-1]})
        else:
            np.save(self.unlearn_file_prefix+'/train_top1.npy', self.save_files['train_top1'])
            np.save(self.unlearn_file_prefix+'/val_top1.npy', self.save_files['val_top1'])
            np.save(self.unlearn_file_prefix+'/unlearn_time.npy', self.save_files['train_time_taken'])

        print('==> Completed! Unlearning Time: [{0:.3f}]\t'.format(self.save_files['train_time_taken']))
        
        for loader, name in [(train_test_loader, 'train'), (test_loader, 'test'), (adversarial_train_loader, 'adv_train'), (adversarial_test_loader, 'adv_test')]:
            if loader is not None:
                preds, targets = self.eval(loader=loader, save_preds=True)
                accuracy = compute_accuracy(preds, targets)
                if wandb_run is not None:
                    wandb_run.log({f"{method_handle}_{name}_acc": accuracy})
                else:
                    np.save(self.unlearn_file_prefix+'/preds_'+name+'.npy', preds)
                    np.save(self.unlearn_file_prefix+'/targets'+name+'.npy', targets)
        return self.save_files['val_top1'][-1]


class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
    

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(model, k=self.opt.k, model_name=self.opt.model)
        model = unlearn_func(model=model, method=self.opt.unlearn_method, factor=self.opt.factor, device=self.opt.device) 
        self.model = model
        self.prenet = prenet
        self.model.cuda()
        if self.prenet is not None: self.prenet.cuda().eval()


    def divide_model(self, model, k, model_name):
        if k == -1: # -1 means retrain all layers
            net = model
            prenet = None
            return prenet, net

        if model_name == 'resnet9':
            assert(k in [1,2,4,5,7,8])
            mapping = {1:6, 2:5, 4:4, 5:3, 7:2, 8:1}
            dividing_part = mapping[k]
            all_mods = [model.conv1, model.conv2, model.res1, model.conv3, model.res2, model.conv4, model.fc] 
            prenet = torch.nn.Sequential(*all_mods[:dividing_part])
            net = torch.nn.Sequential(*all_mods[dividing_part:])

        elif model_name == 'resnetwide28x10':
            assert(k in [1,3,5,7,9,11,13,15,17,19,21,23,25])
            all_mods = [model.conv1, model.layer1, model.layer2, model.layer3, model.norm, model.fc]
            mapping = {1:5, 9:3, 17:2, 25:1}
    
            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                vals = list(mapping.keys())
                for val in vals:
                    if val > k:
                        sel_idx = val
                        break
                layer = mapping[sel_idx]
                prenet_list = all_mods[:layer]
                prenet_additions = list(all_mods[layer][:int(4-(((k-1)//2)%4))])
                prenet = torch.nn.Sequential(*(prenet_list+prenet_additions))
                net_list = list(all_mods[layer][int(4-(((k-1)//2)%4)):])
                net_additions = all_mods[layer+1:]
                net = torch.nn.Sequential(*(net_list+net_additions))

        elif model_name == 'vitb16':
            assert(k in [1,2,3,4,5,6,7,8,9,10,11,12,13])
            all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
            mapping = {1:3, 13:1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                prenet = [model.patch_embed]
                k = 13-k
                prenet += [model.blocks[:k]]
                prenet = torch.nn.Sequential(*prenet)
                net = [model.blocks[k:], model.norm, model.head]
                net = torch.nn.Sequential(*net)
                
        prenet.to(self.opt.device)
        net.to(self.opt.device)
        return prenet, net


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)

        return 


class Scrub(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.cuda().eval()
    

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)

        with torch.no_grad():
            logit_t = self.og_model(images)

        loss = F.cross_entropy(output, target)
        loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)
        
        if self.maximize:
            loss = -loss

        self.top1(output, target)
        return loss


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.maximize=False
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                time_start = time.process_time()
                self.train_one_epoch(loader=forget_loader)
                self.save_files['train_time_taken'] += time.process_time() - time_start
                self.eval(loader=test_loader)

            self.maximize=False
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            self.eval(loader=test_loader)
        return
    

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return


class BadT(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.to(self.opt.device)
        self.og_model.eval()
        self.random_model = unlearn_func(model, 'EU')
        self.random_model.eval()
        self.kltemp = 1
    

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        
        full_teacher_logits = self.og_model(images)
        unlearn_teacher_logits = self.random_model(images)
        f_teacher_out = torch.nn.functional.softmax(full_teacher_logits / self.kltemp, dim=1)
        u_teacher_out = torch.nn.functional.softmax(unlearn_teacher_logits / self.kltemp, dim=1)
        labels = torch.unsqueeze(infgt, 1)
        overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
        student_out = F.log_softmax(output /self.kltemp, dim=1)
        loss = F.kl_div(student_out, overall_teacher_out)
        
        self.top1(output, target)
        return loss


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        return 


class SSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        actual_iters = self.opt.train_iters
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = ssd_tuning(self.model, forget_loader, self.opt.SSDdampening, self.opt.SSDselectwt, train_loader, self.opt.device)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        self.opt.train_iters = actual_iters
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.SSDdampening)+'_'+str(self.opt.SSDselectwt)
        return

class Reload(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.threshold = opt.threshold
        self.replacement_strategy = opt.replacement_strategy
        self.alr = opt.alr

    def calculate_gradients(self, data_loader):
        self.model.train()

        wrapped_dataloader = tqdm.tqdm(data_loader, desc="Calculating gradients", unit="batch")
        for images, target, infgt in wrapped_dataloader:
            images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            output = self.model(images)
            loss = F.cross_entropy(output, target)
            loss.backward()

        gradients = {}

        wrapped_params = tqdm.tqdm(self.model.named_parameters(), desc="Collecting gradients", unit="batch")
        for n, p in wrapped_params:
            gradients[n] = p.grad

        return gradients

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None, wandb_run=None):

        time_start = time.process_time()

        forget_gradients = self.calculate_gradients(forget_loader)
        train_gradients = self.calculate_gradients(train_loader)
        kv = {}

        wrapped_params = tqdm.tqdm(self.model.named_parameters(), desc="Computing knowledge values", unit="batch")
        for n, p in wrapped_params:
            kv[n] = (torch.abs_(forget_gradients[n]) + 1e-10) / (torch.abs_(train_gradients[n]) + 1e-10)

        # Single ascent step
        wrapped_params = tqdm.tqdm(self.model.named_parameters(), desc="Performing ascent step", unit="batch")
        for n, p in wrapped_params:
            p.data.copy_(p.data + self.alr * forget_gradients[n])

        # Rank the knowledge_values
        flattened = torch.cat([tensor.flatten() for tensor in kv.values()])
        threshold = torch.max(torch.topk(flattened, int(self.threshold * len(flattened)), largest=False).values)

        if self.replacement_strategy == "mean":
            fn = torch.mean
        elif self.replacement_strategy == "zero":
            fn = torch.zeros_like
        elif self.replacement_strategy == "random":
            fn = torch.randn_like
        elif self.replacement_strategy == "uniform":
            fn = lambda x: torch.empty_like(x).uniform_(-1, 1)
        elif self.replacement_strategy == "normal":
            fn = lambda x: torch.empty_like(x).normal_()
        elif self.replacement_strategy == "xavier_uniform":
            backup_fn = lambda x: torch.empty_like(x).uniform_(-1, 1)
            fn = torch.nn.init.xavier_uniform_
        elif self.replacement_strategy == "xavier_normal":
            backup_fn = lambda x: torch.empty_like(x).normal_()
            fn = torch.nn.init.xavier_normal_
        elif self.replacement_strategy == "kaiming_uniform":
            backup_fn = lambda x: torch.empty_like(x).uniform_(-1, 1)
            fn = torch.nn.init.kaiming_uniform_
        elif self.replacement_strategy == "kaiming_normal":
            backup_fn = lambda x: torch.empty_like(x).normal_()
            fn = torch.nn.init.kaiming_normal_
        else:
            raise ValueError(f"Unknown replacement strategy: {self.replacement_strategy}")

        wrapped_named_params = tqdm.tqdm(self.model.named_parameters(), desc="Resetting parameters", unit="batch")
        for name, param in wrapped_named_params:
            if len(param.shape) < 2 and name in kv and self.replacement_strategy in ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]:
                param.data = torch.where(kv[name] < threshold, backup_fn(param.data), param.data)
            elif name in kv:
                # print(f"Resetting {forget_gradients[name][forget_gradients[name] < threshold].numel()} parameters in layer {name}")
                param.data = torch.where(kv[name] < threshold, fn(param.data), param.data)

        # Fine-tune the model
        self.model.train()

        wrapped_epoch_range = tqdm.tqdm(range(self.opt.repair_epochs), desc="Repair Fine-tuning", unit="epoch")
        for _ in wrapped_epoch_range:
            self.train_one_epoch(train_loader)
            self.eval(loader=test_loader)

        self.save_files['train_time_taken'] += time.process_time() - time_start
        ret_val = None
        if wandb_run is not None and eval_loaders is not None:
            for key in eval_loaders.keys():
                preds, targets = self.eval(loader=eval_loaders[key], save_preds=True, save_model=False)
                accuracy = compute_accuracy(preds, targets)
                wandb_run.log({f"Reload_{key}_acc": accuracy})            
                if key == "correct":
                    ret_val = accuracy        

        return ret_val

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.repair_epochs)+'_'+str(self.opt.replacement_strategy)
        self.unlearn_file_prefix += '_'+str(self.opt.threshold)+'_'+str(self.opt.rlr)+'_'+str(self.opt.alr)
        return

class GRDA(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)

    def train_one_epoch(self, new_loader, forget_loader):
        self.model.train()
        self.top1.reset()

        for (images, target, infgt) in tqdm.tqdm(new_loader):

            retain_gradients = {}

            images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            f_images, f_target, f_infgt = next(forget_loader)
            f_images, f_target, f_infgt = f_images.cuda(), f_target.cuda(), f_infgt.cuda()
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                self.scaler.scale(loss).backward()

                for n, p in self.model.named_parameters():
                    retain_gradients[n] = p.grad

                self.optimizer.zero_grad()
                loss = self.forward_pass(f_images, f_target, f_infgt)
                self.scaler.scale(loss).backward()

                for n, p in self.model.named_parameters():
                    p.grad = p.grad - retain_gradients[n]

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None, wandb_run=None):

        time_start = time.process_time()

        # Fine-tune the model
        self.model.train()

        wrapped_epoch_range = tqdm.tqdm(range(self.opt.repair_epochs), desc="Repair Fine-tuning", unit="epoch")
        for _ in wrapped_epoch_range:
            self.train_one_epoch(train_loader, forget_loader)

        self.save_files['train_time_taken'] += time.process_time() - time_start
        ret_val = None
        self.eval(test_loader)
        if wandb_run is not None and eval_loaders is not None:
            for key in eval_loaders.keys():
                preds, targets = self.eval(loader=eval_loaders[key], save_preds=True)
                accuracy = compute_accuracy(preds, targets)
                wandb_run.log({f"GRDA_{key}_acc": accuracy})            
                if key == "correct":
                    ret_val = accuracy        

        return ret_val

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.repair_epochs)+'_'+str(self.opt.replacement_strategy)
        self.unlearn_file_prefix += '_'+str(self.opt.threshold)+'_'+str(self.opt.rlr)+'_'+str(self.opt.alr)
        return



