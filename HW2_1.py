#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
torch.manual_seed(1)

# %%
'''Load data'''
# 94 行的Korea, South 手動更改為Korea South
data = np.loadtxt("covid_19.csv",dtype=np.str,delimiter=',')
data = np.delete(data, [0,1,2], 0)
data = np.delete(data, [1,2], 1)
countries = data[:,0]
data = np.delete(data, [0], 1)
print(data.shape) #185 countries,  82days
print(data[0])

# %%
diff_data = data.astype(int)
# 計算差值 共有81個差值
for i in range(data.shape[0]):
    for j in range(1,data.shape[1]):
        diff_data[i][j] = data[i][j].astype(int) - data[i][j-1].astype(int)
print(diff_data.shape)

# %%
diff_data = diff_data[:,1:] # 只有81個插值
print(diff_data.shape)

# %%
'''Correlation coefficient matrix'''
corrcoef_matrix = np.zeros((len(countries),len(countries)))

for i in range(len(countries)):
    for j in range(len(countries)):
        c1 = diff_data[i].astype(int)
        c2 = diff_data[j].astype(int)
        
        corrcoef_matrix[j][i] = np.corrcoef(c1,c2)[0][1]

# %%
mask = np.zeros_like(corrcoef_matrix[:15,:15])
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(30,18))
    ax = sns.heatmap(corrcoef_matrix[:15,:15], mask=mask, square=True, xticklabels=countries[:15], yticklabels=countries[:15],)
    ax.figure.savefig("heatmap_mask.png")

# %%
threshold = .4
length = len(corrcoef_matrix)
country_list = []
for i in range(length):
    for j in range(length):
        if abs(corrcoef_matrix[i][j]) > threshold and i != j:
            country_list.append(i)
            country_list.append(j)
C = np.unique(country_list)
len(C), C # 符合threshold的國家

# %%
'''Generate sequence data'''

L = 5 # interval 3,5,7
print("差值 data shape:", diff_data[C].shape)

C_data = diff_data[C]
seqData, seqLabel = [], []

for i in range(C_data.shape[0]): # 84個國家
    for j in range(C_data.shape[1]-(L)): # 81-5=76 
        x = C_data[i] # i th country
        seqData.append(x[j:j+L].astype(int)) # 會少了最後一個 因為沒有index=81
        seqLabel.append(1 if x[j+L]>x[j+L-1] else 0)

len(seqData), len(seqLabel) # 

# %%
'''取得不重複的資料：1)確保不重複 2) 確保同資料不會有不同label'''

# 先觀察不重複的資料有多長
uni_list = np.unique(seqData, axis=0)
print("Unique values:", len(uni_list)) # length

# 取得不重複資料以及其label(以第一個的label為主)
seen = set()
unique_seqData = []
unique_seqLabel = []
for i in range(len(seqData)):
    t = tuple(seqData[i])
    if t not in seen:
        unique_seqData.append(seqData[i])
        unique_seqLabel.append(seqLabel[i])
        seen.add(t)
print("Ensure data length:", len(unique_seqData)," is equal to ",len(unique_seqLabel))
print("1 counts:", unique_seqLabel.count(1), ", 0 counts:", unique_seqLabel.count(0))

# %%
data = torch.Tensor(unique_seqData)
label = torch.Tensor(unique_seqLabel)

print("Shape of data:",data.shape,", shape of label:",label.shape)

# %%
'''train test split'''
split_ratio = 0.6
trainData = data[:int(len(data)*split_ratio)]
testData = data[int(len(data)*split_ratio):]
trainLabel = label[:int(len(label)*split_ratio)]
testLabel = label[int(len(label)*split_ratio):]
print("train length:",len(trainData),", test length:",len(testData))

# %%
input_size = 1 # input 每次的特徵數
seq_length = L # 看幾天 即interval
hidden_size = 64 # 自定義RNN hidden nodes
number_of_layers = 1 # 幾層RNN
batchsize = 8 # 看過幾個資料後更新
epochs = 1000 # 訓練整個資料幾次
learning_rate = 0.001 # learning rate

'''dataset'''
train_dataset = Data.TensorDataset(trainData,trainLabel) # 轉成torch能夠識別的dataset
test_dataset = Data.TensorDataset(testData,testLabel)

'''data loader'''
# 把train dataset放到dataloader裡
loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=batchsize,      # mini batch size
    shuffle=True
)

# %%
def init_model(state):
    if state == 'RNN':
        return RNN()
    elif state == 'LSTM':
        return LSTM()
    elif state == 'GRU':
        return GRU()
    else:
        return "Failed to initialize model."

# %%
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        r_out, h_n = self.rnn(x, None)
        out = self.out(r_out[:, -1, :]) # 選取看完第五個以後的output (只取最後一個hidden output)
        final = self.sigmoid(out)
        return final

# %%
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        l_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(l_out[:, -1, :]) # 選取看完第五個以後的output (只取最後一個hidden output)
        final = self.sigmoid(out)
        return final

# %%
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        g_out, h_n = self.gru(x, None)
        out = self.out(g_out[:, -1, :])
        final = self.sigmoid(out)
        return final

# %%
state = 'LSTM'
model = init_model(state)
optimizer = torch.optim.Adam(model.parameters(), lr=0.009)   # optimize all cnn parameters
loss_func = nn.BCELoss() 
print(model)

# %%
def calAcc(output, label='train'):
    t = output.detach().numpy()
    pred_y = np.where(t > 0.5, 1, 0).reshape(-1)
    if label == 'train':
        true_y = trainLabel.detach().numpy().astype(int)
    else:
        true_y = testLabel.detach().numpy().astype(int)
    acc = np.sum(true_y == pred_y) / len(pred_y)
    return acc

# %%
'''train'''
loss_record = []
train_acc_record = []
test_acc_record = []

for epoch in range(epochs):
    train_loss = 0
    train_acc = 0
    test_acc = 0

    print("----------------------Starting no.",epoch,"epoch----------------------")
    
    cal_acc_num = 0
    cal_loss_num = 0

    for step, (x, y) in enumerate(loader):  # mini-batch learning
        cal_loss_num += 1
        numofdata = int(x.shape[0]*x.shape[1] / seq_length)
        batch_x = Variable(x.view(numofdata, seq_length, input_size))
        batch_y = Variable(y)

        output = model(batch_x)
        loss = loss_func(output, batch_y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            
            cal_acc_num += 1

#             print('[step %d at epoch %d]' % (step, epoch))
#             print('train loss: %.3f' % loss)
            
            # training
            train_output = model(Variable(trainData.view(trainData.shape[0], seq_length, input_size)))
            acc = calAcc(train_output,'train')
#             print('train acc: %.3f' % acc)
            train_acc += acc

            # testing
            test_output = model(Variable(testData.view(testData.shape[0], seq_length, input_size))) # (samples, time_step, input_size)
            acc = calAcc(test_output,'test')
#             print('test acc: %.3f' % acc)
            test_acc += acc
    
    # when an epoch finished... 
    loss_record.append((train_loss/cal_loss_num))
    train_acc_record.append((train_acc/cal_acc_num))
    test_acc_record.append((test_acc/cal_acc_num))

    print('Epoch: ', epoch, '| train loss: %.3f | train acc: %.3f, test acc: %.3f' % (train_loss/cal_loss_num,train_acc/cal_acc_num,test_acc/cal_acc_num))
    


# %%
'''觀察結果'''
len(loss_record), len(train_acc_record)

# %%
# 建立資料夾
import os
if not os.path.exists('rnn_results'):
   os.makedirs('rnn_results')

# %%
# plot the record
plt.style.use('default')
plt.plot(loss_record)
plt.title("training loss")
plt.savefig('./rnn_results/'+str(state)+'_loss.png')
plt.show()

# %%
plt.style.use('default')
plt.title("training and testinng accuracy")
plt.plot(train_acc_record, label='train acc')
plt.plot(test_acc_record, label='test acc')
plt.legend(loc='upper right')
plt.savefig('./rnn_results/'+str(state)+'_acc.png')
plt.show()

# %%
'''World Map'''
import pygal
from pygal_maps_world.maps import World,COUNTRIES

wmData = C_data[:,-L:]
wmData.shape,type(wmData)

# %%
wmData = torch.Tensor(wmData)
wm_pred = model(Variable(wmData.view(wmData.shape[0], seq_length, input_size)))
print(type(wm_pred.detach().numpy()))
wm_pred_np = wm_pred.detach().numpy()
wm_pred_np.shape, len(wm_pred_np)

# %%
# 取得country name 共181個
original_data = np.loadtxt("covid_19.csv",dtype=np.str,delimiter=',')
original_data = np.delete(original_data, [0,1,2], 0)
original_data = np.delete(original_data, [1,2], 1)
countries = original_data[:,0]

# %%
def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code
    return country_name

# %%
for i in range(len(countries)):
    countries[i] = get_country_code(countries[i])

# %%
countries[np.where(countries == "Taiwan*")] = "tw"
countries[np.where(countries =="Bolivia")] = "bo"
countries[np.where(countries =="Brunei")] = "bn"
countries[np.where(countries =="Congo (Brazzaville)")] = "cg"
countries[np.where(countries =="Congo (Kinshasa)")] = "cd"
countries[np.where(countries =="Cote d'Ivoire")] = "ci"
countries[np.where(countries =="Dominica")] = "do"
countries[np.where(countries =="Holy See")] = "va"
countries[np.where(countries =="Iran")] = "ir"
countries[np.where(countries =="Korea, South")] = "kp"
countries[np.where(countries =="Libya")] = "ly"
countries[np.where(countries =="Moldova")] = "md"
countries[np.where(countries =="North Macedonia")] = "mk"
countries[np.where(countries =="Russia")] = "ru"
countries[np.where(countries =="South Sudan")] = "sd"
countries[np.where(countries =="Syria")] = "sy"
countries[np.where(countries =="Tanzania")] = "tz"
countries[np.where(countries =="US")] = "us"
countries[np.where(countries =="Venezuela")] = "ve"
countries[np.where(countries =="Vietnam")] = "vn"

# %%
a_id= []
d_id = []
for i in range(len(wm_pred_np)):
    if wm_pred_np[i] < 0.5:
        d_id.append(C[i]) # find country id and add it into d
    else:
        a_id.append(C[i]) # find country id and add it into a
len(a_id), len(d_id)

# %%
dict_asc = {}
dict_des = {}
for i in range(len(a_id)):
    dict_asc[countries[a_id[i]]] = float(wm_pred_np[np.where( C == a_id[i])])
for i in range(len(d_id)):
    dict_des[countries[d_id[i]]] = float(wm_pred_np[np.where( C == d_id[i])])

# %%
worldmap_chart = World()
worldmap_chart.add('Ascending', dict_asc)
worldmap_chart.add('Descending', dict_des)

worldmap_chart.render()
worldmap_chart.render_to_file('world_map.svg')


# %%
