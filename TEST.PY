from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
import torch.functional as F


#hyperparameters
block_size=64  #maximum context length
n_head=6
n_layer=  12
dimension=384
head_size=dimension//n_head
learning_rate=1e-5
eval_interval=500
dropout=0.3
v_size=12005
batch_size=64
epochs=5000
eval_iters=200
device='cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(1337)

with open('shakespeare.txt','r',encoding='utf-8') as f:
    text=f.read()

tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                         unk_token="[UNK]", 
                                         cls_token="[CLS]",
                                         sep_token="[SEP]", 
                                         pad_token="[PAD]",
                                         mask_token="[MASK]")


data=torch.tensor(fast_tokenizer.encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]



def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ["train","val"]:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

loss_fn=nn.CrossEntropyLoss()
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        C=dimension
        self.value=nn.Linear(C,head_size ,bias=False)
        self.combined=nn.Linear(C,C,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        f=self.combined(x)
        v=self.value(x)

        wei=f @ x.transpose(-2,-1)*head_size**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        out=wei @ v
        return out


class MultiHeadAttention(nn.Module):
    #multiple heads of self sttention in parallel

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(dimension,dimension)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=torch.cat([h(x) for h in self.heads],dim=-1)
        x=self.dropout(self.proj(x))
        return x
    

class FeedForward(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dimension,4*dimension),nn.ReLU(),nn.Linear(4*dimension,dimension),nn.Dropout(dropout))

    def forward(self,x):
        x=self.net(x)
        return x

class Block(nn.Module):
    #Transfromer block :communication followed by computation 
    def __init__(self,dimension,n_head):
        super().__init__()
        head_size=dimension//n_head
        self.sa_head=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(dimension)
        self.ln1=nn.LayerNorm(dimension)
        self.ln2=nn.LayerNorm(dimension)
        
    def forward(self,x):
        x=x+self.sa_head(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emebdding=nn.Embedding(v_size,dimension)
        self.pos_embedding=nn.Embedding(block_size,dimension)
        self.blocks=nn.Sequential(*[Block(dimension,n_head) for _ in range(n_layer)])
        self.lm_head=nn.Linear(dimension,v_size)

    def forward(self,idx,targets=None):
       B,T=idx.shape
       tok_emb=self.token_emebdding(idx)
       pos_emb=self.pos_embedding(torch.arange(T,device=device)) 
       x=tok_emb+pos_emb
       x=self.blocks(x)
       logits=self.lm_head(x)

       if targets is None:
           loss=None
       else:
           B,T,C=logits.shape
           logits=logits.view(B*T,C)
           targets=targets.view(B*T)
           loss=loss_fn(logits,targets)
       return logits,loss
           
    def generate(self,idx,max_new_tokens):
        #idx is(B,T) shape ids
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:] #focus only on last token of each batch becomes (B,c)
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)#genarting samples from the distribution shape (B,1)
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx
    

model=BigramLanguageModel()
m=model.to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    
    if epoch%eval_interval==0:
        losses=estimate_loss()#model validation
        print(f"epoch{epoch}:train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb=get_batch('train')#get a batch of data
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    #generate from the model
context=torch.zeros([1,1],dtype=torch.long,device=device)
print(fast_tokenizer.decode(m.generate(context,max_new_tokens=500)[0].tolist()))

