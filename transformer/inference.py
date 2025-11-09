import pickle
import torch
from model import Transformer
from torchtext.data.utils import get_tokenizer

## 模型加载后进行推理，可视化翻译效果

#交互式翻译函数
def interactive_translate(model, max_len=50):
    model.eval()
    print("🌍 进入交互式翻译模式 (输入 'quit' 退出)")
    while True:
        sentence = input("\n请输入英文句子: ").strip()
        if sentence.lower() in ["quit", "exit"]:
            print("👋 已退出翻译模式。")
            break

        if not sentence:
            print("⚠️ 输入为空，请重新输入。")
            continue

        # --- 翻译逻辑 ---
        model.eval()
        tokens = ["<bos>"] + tokenizer_en(sentence) + ["<eos>"]
        src_indices = [en_vocab[t] for t in tokens]
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)
        # build encoder output
        src_mask = (src_tensor != en_vocab["<pad>"]).unsqueeze(1).unsqueeze(2)  # [1,1,1,src_len]
        enc_out = model.encoder(src_tensor, src_mask)
        trg_indices = [zh_vocab["<bos>"]]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output = model.decoder(trg_tensor, enc_out, src_mask, trg_mask)  # [1, len, vocab]
            next_token = output.argmax(-1).item() if output.size(1) == 1 else output.argmax(-1)[:, -1].item()
            trg_indices.append(next_token)
            if next_token == zh_vocab["<eos>"]:
                break
        toks = [zh_vocab.lookup_token(i) for i in trg_indices]
        translation = "".join(toks[1:-1])
        print("🗣️ 翻译结果:", translation)



if __name__ == "__main__":
    MODEL_PATH = "transformer_scratch.pth"
    VOCAB_PATH = "vocab.pkl"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_MODEL = 256
    NUM_HEADS = 8
    D_FF = 512
    NUM_LAYERS = 3
    DROPOUT = 0.1

    with open(VOCAB_PATH, "rb") as f:
        data = pickle.load(f)
        en_vocab = data["en_vocab"]
        zh_vocab = data["zh_vocab"]
        tokenizer_en = get_tokenizer(data["tokenizer_en"])
        tokenizer_zh = lambda x: list(x)

    # =========================
    # 初始化并加载模型
    # =========================
    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(zh_vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 模型加载成功，进入交互模式")
    interactive_translate(model)
