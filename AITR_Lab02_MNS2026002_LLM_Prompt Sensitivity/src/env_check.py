"""Environment + data + model sanity check. Run first; share the full console output.
Expected output: system info, library versions, SST-2 split sizes and 3 examples,
label-token check, one forward pass with two label logits, timing."""
import argparse, platform, random, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import numpy as np, torch
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    print("python", sys.version.split()[0], "|", platform.platform())
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU:", p.name, f"{p.total_memory/2**30:.1f} GiB")
    import transformers, datasets
    print("transformers", transformers.__version__, "| datasets", datasets.__version__)

    ds = datasets.load_dataset("stanfordnlp/sst2")
    print({k: len(v) for k, v in ds.items()})
    for ex in ds["validation"].select(range(3)):
        print("EX:", ex["label"], repr(ex["sentence"][:80]))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()

    ids = {}
    for lab in [" negative", " positive"]:
        t = tok.encode(lab, add_special_tokens=False)
        print(f"label {lab!r} -> token ids {t}", "OK single token" if len(t) == 1 else "WARNING: multi-token")
        ids[lab] = t[0]

    sent = ds["validation"][0]["sentence"]
    enc = tok(f"Sentence: {sent}\nLabel:", return_tensors="pt").to(dev)
    t0 = time.time()
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()
    if dev == "cuda": torch.cuda.synchronize()
    print(f"forward pass {time.time()-t0:.3f}s | prompt tokens {enc.input_ids.shape[1]}")
    print({k: round(logits[v].item(), 3) for k, v in ids.items()}, "| gold label:", ds["validation"][0]["label"])
    if dev == "cuda": print(f"peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("ENV CHECK COMPLETE")

if __name__ == "__main__":
    main()
