import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# 主要用于储存单词与id的映射
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: "<SOS>", 1: "<EOS>", -1: "<unk>"}
        self.idx = 2 # Count SOS and EOS

    # 记录word和id之间的映射
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    # 将句子进行分词，添加每个单词与id的映射
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    # 得到某个单词的id
    def __call__(self, word):
        if not word in self.word2idx:
            return -1
        return self.word2idx[word]
    
    # vaocabulary的容量
    def __len__(self):
        return self.idx

class EncoderRNN(nn.Module):
    # 在构造函数内定义了一个Embedding层和一GRU层，
    def __init__(self, input_size, hidden_size):
        # to do

    # 前向传播
    def forward(self, input, hidden):
        # to do
    
    # 最终执行函数
    def sample(self,seq_list):
        # to do

    # 初始化第一层的h0，随机生成一个
    def initHidden(self):
        # to do

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        # to do

    def forward(self, seq_input, hidden):
        # to do

    # pre_hidden即公式中所谓的固定C向量
    def sample(self, pre_hidden):
        # to do
        
# 处理句子，将句子转换成Tensor
def sentence2tensor(lang, sentence):
    indexes = [lang(word) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# 将(input, target)的pair都转换成Tensor
def pair2tensor(pair):
    input_tensor = sentence2tensor(lan1, pair[0])
    target_tensor = sentence2tensor(lan2, pair[1])
    return (input_tensor, target_tensor)

# 定义句子和Vocabulary类
lan1 = Vocabulary()
lan2 = Vocabulary()

data = [['Hi .', '嗨 。'],
        ['Hi .', '你 好 。'],
        ['Run .', '跑'],
        ['Wait !', '等等 ！'],
        ['Hello !', '你好 。'],
        ['I try .', '让 我 来 。'],
        ['I won !', '我 赢 了 。'],
        ['I am OK .', '我 沒事 。']]

for i,j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)
print(len(lan1))
print(len(lan2))

# 定义Encoder和Decoder以及训练的一些参数
import random
learning_rate = 0.001
hidden_size = 256

# 将Encoder, Decoder放到GPU
encoder = EncoderRNN(len(lan1), hidden_size).to(device)
decoder = DecoderRNN(hidden_size, len(lan2)).to(device)
# 网络参数 = Encoder参数 + Decoder参数
params = list(encoder.parameters()) + list(decoder.parameters())
# 定义优化器
optimizer = optim.Adam(params, lr=learning_rate)
loss = 0
# NLLLoss = Negative Log Likelihood Loss
criterion = nn.NLLLoss()
# 一共训练多次轮
turns = 200
print_every = 20
print_loss_total = 0
# 将数据random choice，然后转换成 Tensor
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

# 训练过程
for turn in range(turns):
    optimizer.zero_grad()
    loss = 0
    
    x, y = training_pairs[turn]
    input_length = x.size(0)
    target_length = y.size(0)
    # 初始化Encoder中的h0
    h = encoder.initHidden()
    # 对input进行Encoder
    for i in range(input_length):
        h = encoder(x[i],h)
    # Decoder的一个input <sos>
    decoder_input = torch.LongTensor([SOS_token]).to(device)
    
    for i in range(target_length):
        decoder_output, h = decoder(decoder_input, h)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, y[i])
        if decoder_input.item() == EOS_token:break
                
    print_loss_total += loss.item()/target_length
    if (turn+1) % print_every == 0 :
        print("loss:{loss:,.4f}".format(loss=print_loss_total/print_every))
        print_loss_total = 0
        
    loss.backward()
    optimizer.step()

# 测试函数
def translate(s):
    t = [lan1(i) for i in s.split()]
    t.append(EOS_token)
    f = encoder.sample(t)   # 编码
    s = decoder.sample(f)   # 解码
    r = [lan2.idx2word[i] for i in s]    # 根据id得到单词
    return ' '.join(r) # 生成句子
print(translate('I try .'))
#print(translate('我们 一起 打 游戏 。'))
