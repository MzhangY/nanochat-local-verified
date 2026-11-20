"""Tiny offline CPU training smoke test using nanochat's real GPT implementation.
Not the upstream end-to-end training pipeline or a language-quality benchmark.
"""
import argparse,json,math,tempfile,time
from pathlib import Path
import torch
from nanochat.gpt import GPT,GPTConfig
from nanochat.tokenizer import RustBPETokenizer

def run():
    torch.set_num_threads(2);torch.manual_seed(20261120)
    config=GPTConfig(sequence_len=32,vocab_size=32,n_layer=2,n_head=2,n_kv_head=2,n_embd=64,window_pattern='L')
    model=GPT(config);model.init_weights();model.train()
    # Deterministic synthetic token pattern: no external data, pretrained weights or tokenizer.
    sequence=torch.arange(33,dtype=torch.long)%8
    x=sequence[:-1].unsqueeze(0).repeat(4,1)
    y=sequence[1:].unsqueeze(0).repeat(4,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.01,weight_decay=0.0)
    with torch.no_grad():initial=float(model(x,targets=y))
    history=[];start=time.monotonic()
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,targets=y)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step in [0,9,19,39,79]:history.append({'step':step+1,'loss':float(loss.detach())})
    model.eval()
    with torch.no_grad():
        final=float(model(x,targets=y));logits=model(x);predictions=logits.argmax(-1)
        accuracy=float((predictions==y).float().mean())
    assert math.isfinite(final) and final<initial*0.25,(initial,final)
    assert accuracy>=0.95,accuracy
    with tempfile.TemporaryDirectory() as directory:
        checkpoint=Path(directory)/'synthetic-smoke.pt'
        torch.save(model.state_dict(),checkpoint)
        clone=GPT(config);clone.init_weights();clone.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));clone.eval()
        with torch.no_grad():torch.testing.assert_close(clone(x),logits,rtol=0,atol=0)
    corpus=['Hello AI! 中文分词与小模型训练 🤖\n','hello world and neural networks 123456\n']*12
    tokenizer=RustBPETokenizer.train_from_iterator(iter(corpus),320)
    samples=['你好，AI 🤖','unseen English 12345','']
    assert all(tokenizer.decode(tokenizer.encode(s))==s for s in samples)
    return {'task':'nanochat synthetic-token CPU training + tokenizer + checkpoint roundtrip','torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads(),'seed':20261120,'parameters':sum(p.numel() for p in model.parameters()),'config':config.__dict__,'steps':80,'initial_loss':initial,'final_loss':final,'training_token_accuracy':accuracy,'history':history,'checkpoint_roundtrip_exact':True,'multilingual_tokenizer_roundtrip':True,'tokenizer_vocab_size':tokenizer.get_vocab_size(),'elapsed_seconds':round(time.monotonic()-start,3),'network_data_or_models_downloaded':False,'limitations':['Overfits a tiny synthetic repeating sequence, not a real language dataset or LLM quality benchmark.','Uses torch.optim.AdamW in eager CPU mode; does not validate upstream Muon/distributed/torch.compile optimizers.','Does not run full pretraining/SFT/evaluation pipeline or GPU performance tests.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    result=run();a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
