import math
import random
import codecs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()
SOS_token = 1
EOS_token = 2


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers = 1, dropout = 0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,n_layers,dropout=dropout, bidirectional=True)
        # self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        def forward(self, src, hidden=None):
            embedded = self.embed(src)
            output, hidden = self.lstm(embedded, hidden)
            # 对双向lstm进行求和
            output = (output[:,:,:self.hidden_size] + output[:,:,:self.hidden_size])

            return output, hidden
        
        def init_hidden(self):
            hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
            if USE_CUDA:
                hidden = hidden.cuda()
            return hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # 16,15,256
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        # self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
        #                   n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        # output, hidden = self.gru(rnn_input, last_hidden)
        output, hidden = self.lstm(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1)) # 维度为1进行拼接
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.max_epoches = 100
        self.batch_index = 0
        self.GO_token = 1
        self.EOS_token = 2

        self.input_size = 14
        self.output_size = 15
        self.hidden_size = 100
        self.max_length = 15
        self.show_epoch = 100
        self.use_cuda = USE_CUDA
        self.model_path = "./model/"
        self.n_layers = 1
        self.beam_search = True
        self.top_k = 5
        self.alpha = 0.5

        self.q = []
        self.a = []


        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def loadData(self):
        with codecs.open("./data/enc.vec", 'r', encoding="utf-8") as q:
            line = q.readline()
            while line:
                q.append(line.strip().split)
                line = q.readline()
        with codecs.open("./data/dec.vec", 'r', encoding="utf-8") as a:
            line = a.readline()
            while line:
                a.append(line.strip().split)
                line = a.readline()
    def next(self, batch_size, shuffle = False):
        inputs = []
        targets = []

        if shuffle:
            ind = random.choice(range(len(self.q)))
            q = [self.q[ind]]
            a = [self.a[ind]]
        else:
            if self.batch_index+batch_size >= len(self.q):
                q = self.q[self.batch_index:]
                a = self.a[self.batch_index:]
                self.batch_index = 0
            else:
                q = self.q[self.batch_index:self.batch_index+batch_size] 
                a = self.a[self.batch_index:self.batch_index+batch_size]
                self.batch_index += batch_size
        for index in range(len(q)):
            q = q[0][:self.max_len] if len(q[0])>self.max_len else q[0]
            a = a[0][:self.max_len] if len(a[0])>self.max_len else a[0]

            q = [int(i) for i in q]
            a = [int(i) for i in a]

            a.append(EOS_token)

            inputs.append(q)
            targets.append(a)
        
        inputs = Variable(torch.LongTensor(inputs)).transpose(1,0).contiguous()
        targets = Variable(torch.LongTensor(targets)).transpose(1,0).contiguous()

        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        return inputs, targets

    def train(self):
        self.loadData()
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        for epoch in range(self.max_epoches):
            start = time.time()
            inputs, targets = self.next(1, shuffle=False)
            loss, logits = self.step(inputs, targets, self.max_length)
            loss_track.append(loss)
            _,v = torch.topk(logits, 1)
            pre = v.cpu().data.numpy().T.tolist()[0][0]
            tar = targets.cpu().data.numpy().T.tolist()[0]
            stop = time.time()
            if epoch % self.show_epoch == 0:
                print("-"*50)
                print("epoch:", epoch)
                print("    loss:", loss)
                print("    target:%s\n    output:%s" % (tar, pre))
                print("    per-time:", (stop-start))
                torch.save(self.state_dict(), self.model_path+'params.pkl')



    def step(self, input_Variable, targets_Variable, max_len):
        teacher_forcing_ratio = 0.1
        clip = 5.0
        loss = 0 

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_lenght = input_Variable.shape[0]
        target_length = targets_Variable.shape[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context =  Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden

        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
        decoder_outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        use_teacher_forcing = True

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                if USE_CUDA: decoder_input = decoder_input.cuda()
                if ni == EOS_token: break
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        decoder_outputs = torch.cat(decoder_outputs, 0)
        return loss.data[0] / target_length, decoder_outputs


    def make_infer_fd(self, input_vec):
        inputs = []
        enc = input_vec[:self.max_length] if len(input_vec) > self.max_length else input_vec
        inputs.append(enc)
        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()
        if USE_CUDA:
            inputs = inputs.cuda()
        return inputs

    def predict(self):
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        # 加载字典
        str_to_vec = {}
        with open("./data/enc.vocab") as enc_vocab:
            for index,word in enumerate(enc_vocab.readlines()):
                str_to_vec[word.strip()] = index

        vec_to_str = {}
        with open("./data/dec.vocab") as dec_vocab:
            for index,word in enumerate(dec_vocab.readlines()):
                vec_to_str[index] = word.strip()

        while True:
            input_strs = input("me > ")
            # 字符串转向量
            segement = jieba.lcut(input_strs)
            input_vec = [str_to_vec.get(i, 3) for i in segement]
            input_vec = self.make_infer_fd(input_vec)

            # inference
            if self.beam_search:
                samples = self.beamSearchDecoder(input_vec)
                for sample in samples:
                    outstrs = []
                    for i in sample[0]:
                        if i == 1:
                            break
                        outstrs.append(vec_to_str.get(i, "Un"))
                    print("ai > ", "".join(outstrs), sample[3])
            else:
                logits = self.infer(input_vec)
                _,v = torch.topk(logits, 1)
                pre = v.cpu().data.numpy().T.tolist()[0][0]
                outstrs = []
                for i in pre:
                    if i == 1:
                        break
                    outstrs.append(vec_to_str.get(i, "Un"))
                print("ai > ", "".join(outstrs))

    def infer(self, input_variable):
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output.unsqueeze(0))
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()
            if ni == EOS_token: break

        decoder_outputs = torch.cat(decoder_outputs, 0)
        return decoder_outputs

    def tensorToList(self, tensor):
        return tensor.cpu().data.numpy().tolist()[0]

    def beamSearchDecoder(self, input_variable):
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        topk = decoder_output.data.topk(self.top_k)
        samples = [[] for i in range(self.top_k)]
        dead_k = 0
        final_samples = []
        for index in range(self.top_k):
            topk_prob = topk[0][0][index]
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index], topk_prob, 0, 0, decoder_context, decoder_hidden, decoder_attention, encoder_outputs]

        for _ in range(self.max_length):
            tmp = []
            for index in range(len(samples)):
                tmp.extend(self.beamSearchInfer(samples[index], index))
            samples = []

            # 筛选出topk
            df = pd.DataFrame(tmp)
            df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores", "decoder_context", "decoder_hidden", "decoder_attention", "encoder_outputs"]
            sequence_len = df.sequence.apply(lambda x:len(x))
            df['ave_scores'] = df['fin_scores'] / sequence_len
            df = df.sort_values('ave_scores', ascending=False).reset_index().drop(['index'], axis=1)
            df = df[:(self.top_k-dead_k)]
            for index in range(len(df)):
                group = df.ix[index]
                if group.tolist()[0][-1] == 1:
                    final_samples.append(group.tolist())
                    df = df.drop([index], axis=0)
                    dead_k += 1
                    print("drop {}, {}".format(group.tolist()[0], dead_k))
            samples = df.values.tolist()
            if len(samples) == 0:
                break

        if len(final_samples) < self.top_k:
            final_samples.extend(samples[:(self.top_k-dead_k)])
        return final_samples

    def beamSearchInfer(self, sample, k):
        samples = []
        decoder_input = Variable(torch.LongTensor([[sample[0][-1]]]))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        sequence, pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs = sample
        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

        # choose topk
        topk = decoder_output.data.topk(self.top_k)
        for k in range(self.top_k):
            topk_prob = topk[0][0][k]
            topk_index = int(topk[1][0][k])
            pre_scores += topk_prob
            fin_scores = pre_scores - (k - 1 ) * self.alpha
            samples.append([sequence+[topk_index], pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs])
        return samples

    def retrain(self):
        try:
            os.remove(self.model_path)
        except Exception as e:
            pass
        self.train()

if __name__ == '__main__':
    seq = seq2seq()
    if sys.argv[1] == 'train':
        seq.train()
    elif sys.argv[1] == 'predict':
        seq.predict()
    elif sys.argv[1] == 'retrain':
        seq.retrain()


    # def forward(self, src, trg, teacher_forcing_ratio=0.5):
    #     batch_size = src.size(1)
    #     max_len = trg.size(0)
    #     vocab_size = self.decoder.output_size
    #     outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

    #     encoder_output, hidden = self.encoder(src)
    #     hidden = hidden[:self.decoder.n_layers]
    #     output = Variable(trg.data[0, :])  # sos
    #     for t in range(1, max_len):
    #         output, hidden, attn_weights = self.decoder(
    #                 output, hidden, encoder_output)
    #         outputs[t] = output
    #         is_teacher = random.random() < teacher_forcing_ratio
    #         top1 = output.data.max(1)[1]
    #         output = Variable(trg.data[t] if is_teacher else top1).cuda()
    #     return outputs