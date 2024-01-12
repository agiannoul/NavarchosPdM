import numpy as np
import sklearn.preprocessing
import torch
from math import sqrt

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import TranAD
import TranAD.models
from TranAD.utils import color
from utils.structure import Datapoint
from utils import structure
from thresholding import thresholding
import os




class TranADPdm():
    def __init__(self,source,thresholdtype="selftunne",thresholdfactor=3,window_size=10,num_epochs=15,Auto_transform=False):
        self.tranadmodel=tranadcore(window_size,source,num_epochs,Auto_transform)
        self.window_size=window_size
        self.thresholdtype=thresholdtype
        self.source=source
        self.thresholdfactor=thresholdfactor
        self.threshold=None
        self.thmean=-1
        self.thstd=-1
    def initilize(self,point :Datapoint):
        #self.tranadmodel.fit(point.reference)
        self.createtranadCore(point)
    def createtranadCore(self,point :Datapoint):
        if self.thresholdtype=="selftunne":
            limit=int(0.5*len(point.reference.index))
            profile=point.reference.iloc[:limit]
            dataForNormal=point.reference.iloc[limit:]
            self.tranadmodel.fit(profile)
            dataForNormal= self.tranadmodel.convert_to_windows_numpy(dataForNormal.values)
            nomralscores=[]
            for ddd in dataForNormal:
                nomralscores.append(self.tranadmodel.predict(ddd))
            self.threshold,self.thmean,self.thstd=thresholding.selfTuning(self.thresholdfactor,nomralscores,returnmean=True)
        else:
            self.tranadmodel.fit(point.reference)

    def get_data(self,datapointlist):
        score = self.calculatescore(datapointlist)
        temp_datapoint=datapointlist[-1]
        if score == None:
            prediction = structure.PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,temp_datapoint.source, description="model not initialized")
        else:
            if self.thresholdtype=="selftunne":
                alarm=score > self.threshold
                thfinal=self.threshold
            else:
                alarm=True
                thfinal=-1
            prediction = structure.PredictionPoint(score, thfinal,alarm,  self.thresholdtype,
                                                   temp_datapoint.timestamp, temp_datapoint.source,
                                                   description="tranadpdm",notes=f"{self.thmean},{self.thstd},")
        return prediction

    def calculatescore(self,datapointlist):
        tempwindow=datapointlist[-self.window_size:]
        if len(tempwindow)<self.window_size:
            constructwidnow=[]
            for i in range(self.window_size-len(datapointlist)):
                constructwidnow.append(datapointlist[0].current)
            constructwidnow.extend([d.current for d in datapointlist])
            tempwindow=constructwidnow
        score=self.tranadmodel.predict(np.array([d.current for d in tempwindow]))
        return score

    def reset(self):
        self.score_buffer = []
        self.tranadmodel.model=None
        sfoldername=self.tranadmodel.source
        if os.path.exists(f'checkpoints/{sfoldername}/model.ckpt'):
            try:
                os.remove(f'checkpoints/{sfoldername}/model.ckpt')
            except:
                ok='ok'
class tranadcore():
    def __init__(self,window_size,source,num_epochs=15,Auto_transform=False):
        self.window_size=window_size
        self.source=source
        self.num_epochs = num_epochs
        self.model=None
        self.Auto_transform=Auto_transform
        self.scaler=sklearn.preprocessing.MinMaxScaler()
    def convert_to_windows_numpy(self,data):
        windows = [];
        w_size = self.window_size
        for i, g in enumerate(data):
            if i >= w_size:
                w = data[i - w_size:i]
            else:
                w=[]
                w.extend([data[0] for q in range(w_size - i)])
                w.extend(data[0:i])
            windows.append(np.array(w))
        windows = np.array(windows)
        return windows
    def convert_to_windows(self,data):
        windows = [];
        w_size = self.window_size
        for i, g in enumerate(data):
            if i >= w_size:
                w = data[i - w_size:i]
            else:
                w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
            windows.append(w)
        return torch.stack(windows)

    def load_dataset(self,train, test):
        loader = [train, test]
        # loader[0] is ndarray
        # loader = [i[:, debug:debug+1] for i in loader]
        train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
        test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
        return train_loader, test_loader

    def load_dataset_one(self,train):
        loader = [train]
        train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
        return train_loader

    def backprop(self,epoch, model, data, dataO, optimizer, scheduler, training=True, DoublePass=False):
        l = nn.MSELoss(reduction='mean' if training else 'none')
        feats = dataO.shape[1]
        if 'TranAD' in model.name:
            l = nn.MSELoss(reduction='none')
            # data=data.to(device)
            # data_x = torch.DoubleTensor(data);
            data_x = torch.tensor(data.clone().detach().requires_grad_(True));
            dataset = TensorDataset(data_x, data_x)
            bs = model.batch if training else len(data)
            dataloader = DataLoader(dataset, batch_size=bs)

            n = epoch + 1;
            w_size = model.n_window
            l1s, l2s = [], []
            if training:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                for d, _ in dataloader:
                    # d=d.to(device)
                    local_bs = d.shape[0]
                    window = d.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, feats)

                    # window provide the window and elem the last one (But supposed it provided C and W)
                    z = model(window, elem)
                    l1 = l(z, elem) if not isinstance(z, tuple) else (0.95 ** n) * l(z[0], elem) + (1 - 0.95 ** n) * l(
                        z[1], elem)
                    l1 += 0 if not isinstance(z, tuple) or len(z) < 3 else (0.95 ** n) * l(z[2], elem) - (
                                1 - 0.95 ** n) * l(z[1], elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                scheduler.step()
                #tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
                return np.mean(l1s), optimizer.param_groups[0]['lr']
            else:
                cccc = 0
                for d, _ in dataloader:
                    cccc += 1
                    window = d.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, bs, feats)
                    z = model(window, elem)
                    if DoublePass:
                        return z[0].cpu().detach().numpy()[0], z[1].cpu().detach().numpy()[0]
                    if isinstance(z, tuple): z = z[1]
                loss = l(z, elem)[0]
                return loss.cpu().detach().numpy(), z.cpu().detach().numpy()[0]
        else:
            y_pred = model(data)
            loss = l(y_pred, data)
            if training:
                #tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                return loss.item(), optimizer.param_groups[0]['lr']
            else:
                return loss.detach().numpy(), y_pred.detach().numpy()

    def save_model(self,model, optimizer, scheduler, epoch, accuracy_list, namefolder):
        folder = f'checkpoints/{namefolder}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list}, file_path)

    def load_model_if_existed(self,dims,namefolder, pretrainied=False,force_train=False):
        fname = f'checkpoints/{namefolder}/model.ckpt'
        existed = False
        if pretrainied and os.path.exists(fname) and force_train == False:
            model_class = getattr(TranAD.models, "TranAD")
            model = model_class(dims).double()
            optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
            #print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            existed = True
            return model, existed
        return None,existed
    def load_model(self, dims,namefolder, pretrainied=False,force_train=False):
        model_class = getattr(TranAD.models, "TranAD")
        model = model_class(dims).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        fname = f'checkpoints/{namefolder}/model.ckpt'
        existed = False
        if pretrainied and os.path.exists(fname) and force_train==False:
            #print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            accuracy_list = checkpoint['accuracy_list']
            #print(f" epoch: {epoch}, fname=: {fname}")
            existed = True
        else:
            #print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
            epoch = -1;
            accuracy_list = []
        return model, optimizer, scheduler, epoch, accuracy_list, existed

    # predictOneValue
    def reconstruct(windowpulse, pulse, model):
        windowpulse = np.expand_dims(windowpulse, axis=1)
        pulse = np.expand_dims(pulse, axis=0)
        pulse = np.expand_dims(pulse, axis=0)
        windowtensor = torch.from_numpy(windowpulse)
        pulsetensor = torch.from_numpy(pulse)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        windowtensor = windowtensor.to(device)
        pulsetensor = pulsetensor.to(device)
        z = model(windowtensor, pulsetensor)
        # if isinstance(z, tuple): z = z[1]
        if isinstance(z, tuple): z = z[1]
        return z.cpu().detach().numpy()[0]

    def reconstructDoublePass(windowpulse, pulse, model):
        windowpulse = np.expand_dims(windowpulse, axis=1)
        pulse = np.expand_dims(pulse, axis=0)
        pulse = np.expand_dims(pulse, axis=0)
        windowtensor = torch.from_numpy(windowpulse)
        pulsetensor = torch.from_numpy(pulse)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        windowtensor = windowtensor.to(device)
        pulsetensor = pulsetensor.to(device)
        z = model(windowtensor, pulsetensor)
        # if isinstance(z, tuple): z = z[1]
        return z[0].cpu().detach().numpy()[0], z[1].cpu().detach().numpy()[0]

    def trainmodelinDF(self,dfprofile,name, num_epochs=15, pretrainied=False):
        df_train = dfprofile
        try:
            traindata = df_train.values
        except:
            traindata = df_train

        TrainX = traindata
        train_loader = self.load_dataset_one(TrainX)


        model, optimizer, scheduler, epoch, accuracy_list, existed = self.load_model(len(TrainX[0]), pretrainied,name,force_train=True)

        model.n_window = self.window_size
        ## Prepare data
        trainD = next(iter(train_loader))
        trainO = trainD

        trainD = self.convert_to_windows(trainD)

        ################################    TRAINING

        e = epoch + 1;

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        trainD = trainD.to(device)
        trainO = trainO.to(device)

        for e in list(range(epoch + 1, epoch + num_epochs + 1)):
            lossT, lr = self.backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        self.save_model(model, optimizer, scheduler, e, accuracy_list, name)
        return model
    def reconstructDoublePass(self,windowpulse, pulse, model):
        windowpulse = np.expand_dims(windowpulse, axis=1)
        pulse = np.expand_dims(pulse, axis=0)
        pulse = np.expand_dims(pulse, axis=0)

        windowtensor = torch.from_numpy(windowpulse)

        pulsetensor = torch.from_numpy(pulse)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        windowtensor = windowtensor.to(device)
        pulsetensor = pulsetensor.to(device)
        z = model(windowtensor, pulsetensor)
        # if isinstance(z, tuple): z = z[1]
        return z[0].cpu().detach().numpy()[0], z[1].cpu().detach().numpy()[0]

    def predict(self,window):
        if self.Auto_transform:
            newwindow=[]
            newwindow=self.scaler.transform(window)
            window=np.array(newwindow)
        pulse= window[-1]
        if self.model==None:
            model,existed=self.load_model_if_existed(namefolder=self.source,dims=len(pulse))
            if existed==False:
                return None
        rec_pulse1, rec_pulse2 = self.reconstructDoublePass(window,pulse, self.model)
        pred = rec_pulse1[0]
        sum1 = self.l2predpulse(pred, pulse)
        pred = rec_pulse2[0]
        sum2 = self.l2predpulse(pred, pulse)
        loss1 = sqrt(float(sum1)) / sqrt(len(pred))
        loss2 = sqrt(float(sum2)) / sqrt(len(pred))
        return 0.5 * loss1 + 0.5 * loss2

    def fit(self,dfprofile):
        if self.Auto_transform:
            self.scaler.fit(dfprofile)
            self.model=self.trainmodelinDF(self.scaler.transform(dfprofile),self.source,num_epochs=self.num_epochs)
        else:
            self.model = self.trainmodelinDF(dfprofile, self.source, num_epochs=self.num_epochs)

    def l2predpulse(self,pred, pulse):
        sum1 = 0
        for i in range(len(pred)):
            dist = (pred[i] - pulse[i]) ** 2
            sum1 += dist
        return sum1