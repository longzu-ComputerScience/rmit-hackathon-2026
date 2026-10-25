"""CPU-friendly, single-backbone prompt classifier with cached frozen layers.

Frozen activations depend only on the public pretrained model and each input text.
They contain no labels or fitted dataset statistics and can safely be reused across folds.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
SEED = 20261022
PRETRAINED = ROOT / "artifacts/checkpoints/pretrained"
CACHE = ROOT / "artifacts/cache"
FROZEN = 16
TRAIN_FROM = 20


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def frames():
    train = pd.read_csv(ROOT / "dataset/train.csv", dtype=str)
    test = pd.read_csv(ROOT / "dataset/test.csv", dtype=str)
    folds = pd.read_csv(ROOT / "artifacts/audit/folds.csv")
    assert train.id.equals(folds.id)
    return train, test, folds


def config():
    cfg = AutoConfig.from_pretrained(PRETRAINED, local_files_only=True)
    cfg.reference_compile = False
    cfg._attn_implementation = "sdpa"
    return cfg


def tokenize(max_length):
    CACHE.mkdir(parents=True, exist_ok=True)
    train, test, _ = frames()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, local_files_only=True)
    texts = pd.concat([train.text, test.text], ignore_index=True)
    raw = tokenizer(texts.tolist(), add_special_tokens=False, truncation=False)["input_ids"]
    lengths = np.array([len(t) + 2 for t in raw])
    stats = {"max_length": max_length, "frozen_layers": FROZEN,
             "token_quantiles": {str(q): float(np.quantile(lengths, q)) for q in [0, .5, .9, .95, .99, 1]},
             "truncated_fraction": float(np.mean(lengths > max_length)),
             "truncated_by_class": {label: float(np.mean(lengths[:len(train)][train.label == label] > max_length)) for label in LABELS},
             "backbone": json.loads((ROOT / "artifacts/provenance/backbone.json").read_text()),
             "train_sha256": hashlib.sha256((ROOT / "dataset/train.csv").read_bytes()).hexdigest(),
             "test_sha256": hashlib.sha256((ROOT / "dataset/test.csv").read_bytes()).hexdigest()}
    if (CACHE / "metadata.json").exists():
        prior = json.loads((CACHE / "metadata.json").read_text())
        for field in ("max_length", "frozen_layers", "train_sha256", "test_sha256", "backbone"):
            if prior.get(field) != stats[field]:
                raise ValueError(f"Cache provenance changed for {field}; use a separate cache directory")
    ids = []
    for tokens in raw:
        if len(tokens) > max_length - 2:
            # Preserve the request at the start AND instructions often hidden at the end.
            budget = max_length - 2
            tail = budget // 4
            tokens = tokens[:budget-tail] + tokens[-tail:]
        # Gemma's fast tokenizer applies these in its Rust postprocessor; its
        # generic build_inputs_with_special_tokens method does not add them.
        ids.append([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id])
    lens = np.array([len(t) for t in ids], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(lens)].astype(np.int64)
    flat = np.concatenate([np.array(t, np.int32) for t in ids])
    np.savez(CACHE / "tokens.npz", tokens=flat, offsets=offsets, lengths=lens)
    dump(CACHE / "metadata.json", stats)
    print(json.dumps(stats, indent=2), flush=True)
    return flat, offsets, lens


def batches(indices, lengths, batch_size=16, shuffle=False, seed=SEED, token_budget=2048):
    rng = np.random.default_rng(seed)
    order = np.array(indices)
    if shuffle:
        rng.shuffle(order)
        chunks = [order[i:i+256] for i in range(0, len(order), 256)]
        order = np.concatenate([chunk[np.argsort(lengths[chunk], kind="stable")] for chunk in chunks])
    else:
        order = order[np.argsort(lengths[order], kind="stable")]
    output, bucket = [], []
    for i in order:
        if bucket and (len(bucket) >= batch_size or (len(bucket)+1)*int(lengths[i]) > token_budget):
            output.append(np.array(bucket))
            bucket = []
        bucket.append(int(i))
    if bucket:
        output.append(np.array(bucket))
    if shuffle:
        rng.shuffle(output)
    return output


def masks(mask, cfg, dtype=torch.float32):
    n = mask.shape[1]
    full = torch.zeros((len(mask), 1, n, n), dtype=dtype)
    full.masked_fill_(~mask[:, None, None, :], torch.finfo(dtype).min)
    pos = torch.arange(n)
    local = full.masked_fill((pos[:, None]-pos[None, :]).abs()[None, None] > cfg.local_attention//2,
                             torch.finfo(dtype).min)
    return full, local, pos[None]


def pool(hidden, mask):
    mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    return torch.cat([mean, hidden[:, 0]], dim=-1)


def cache_activations(max_length=320):
    flat, offsets, lens = tokenize(max_length)
    cfg = config()
    print("Loading pinned approved backbone", flush=True)
    backbone = AutoModel.from_pretrained(PRETRAINED, config=cfg, local_files_only=True,
                                          attn_implementation="sdpa", weights_only=True)
    backbone.eval().requires_grad_(False)
    print("Loaded backbone, parameters", sum(p.numel() for p in backbone.parameters()), flush=True)
    n, h = len(lens), cfg.hidden_size
    shape = (int(offsets[-1]), h)
    path = CACHE / "hidden.f16"
    if path.exists() and path.stat().st_size != shape[0]*h*2:
        if (CACHE / "done.npy").exists():
            raise ValueError("Existing activation cache shape differs; choose a separate cache before changing length")
        # A process interrupted during initial allocation has no completed rows.
        with path.open("r+b") as handle:
            handle.truncate(shape[0]*h*2)
    if not path.exists():
        with path.open("wb") as handle:
            handle.truncate(shape[0]*h*2)
    done_path = CACHE / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(n, bool)
    pooled_path = CACHE / "pooled.npy"
    pooled = np.load(pooled_path) if pooled_path.exists() else np.zeros((n, h*2), np.float32)
    batch_list = batches(np.where(~done)[0], lens)
    started = time.monotonic()
    count = 0
    for step, ix in enumerate(batch_list):
        size = int(lens[ix].max())
        token_ids = torch.zeros((len(ix), size), dtype=torch.long)
        mask = torch.arange(size)[None] < torch.from_numpy(lens[ix])[:, None]
        for row, i in enumerate(ix):
            token_ids[row, :lens[i]] = torch.from_numpy(flat[offsets[i]:offsets[i+1]].astype(np.int64))
        full, local, pos = masks(mask, cfg)
        with torch.inference_mode():
            hidden = backbone.embeddings(token_ids)
            for layer in backbone.layers[:FROZEN]:
                hidden = layer(hidden, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
            # Round at the boundary consistently for cached and direct inference.
            rounded = hidden.to(torch.float16).float()
            mm = np.memmap(path, dtype=np.float16, mode="r+", shape=shape)
            for row, i in enumerate(ix):
                mm[offsets[i]:offsets[i+1]] = rounded[row, :lens[i]].numpy().astype(np.float16)
            mm.flush()
            del mm
            for layer in backbone.layers[FROZEN:]:
                rounded = layer(rounded, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
            pooled[ix] = pool(backbone.final_norm(rounded), mask).numpy()
        done[ix] = True
        count += len(ix)
        if step % 25 == 0 or step == len(batch_list)-1:
            np.save(done_path, done)
            np.save(pooled_path, pooled)
            elapsed = time.monotonic()-started
            print(f"Cache {int(done.sum())}/{n}, {count/max(elapsed,1):.1f} rows/s, elapsed {elapsed/60:.1f}m", flush=True)
    metadata = json.loads((CACHE / "metadata.json").read_text())
    metadata.update({"shape": list(shape), "complete": bool(done.all()), "cache_seconds": time.monotonic()-started})
    dump(CACHE / "metadata.json", metadata)


class FixedStandardization(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean",torch.zeros(size))
        self.register_buffer("scale",torch.ones(size))

    def forward(self, inputs):
        return (inputs-self.mean)/self.scale


def fit_linear(features, y, c, weights=None):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    model = LogisticRegression(C=c,max_iter=1500,solver="lbfgs",random_state=SEED,
                               class_weight=None if weights is None else {i:float(w) for i,w in enumerate(weights)})
    model.fit(standardized,y)
    if int(model.n_iter_.max()) >= model.max_iter:
        raise RuntimeError("Standardized linear fit did not converge")
    return model,scaler


class TailClassifier(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = torch.nn.ModuleList([ModernBertEncoderLayer(cfg, i) for i in range(TRAIN_FROM, cfg.num_hidden_layers)])
        self.final_norm = torch.nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.classifier = torch.nn.Sequential(FixedStandardization(cfg.hidden_size*2),torch.nn.Dropout(.1),torch.nn.Linear(cfg.hidden_size*2,3))

    def forward(self, hidden, mask):
        full, local, pos = masks(mask, self.cfg, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
        return self.classifier(pool(self.final_norm(hidden), mask))


def initialize_tail():
    cfg = config()
    model = TailClassifier(cfg)
    # Read only the last layers from the official torch weights using mmap and weights_only.
    state = torch.load(PRETRAINED / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    translated = {}
    for name in model.state_dict():
        if name.startswith("classifier."):
            continue
        original = name
        if name.startswith("layers."):
            pieces = name.split(".")
            pieces[1] = str(int(pieces[1])+TRAIN_FROM)
            original = ".".join(pieces)
        translated[name] = state["model." + original]
    missing, extra = model.load_state_dict(translated, strict=False)
    assert all(k.startswith("classifier.") for k in missing) and not extra
    return model


class CachedInputs:
    def __init__(self):
        token_data = np.load(CACHE / "tokens.npz")
        self.lengths = token_data["lengths"]
        self.offsets = token_data["offsets"]
        self.metadata = json.loads((CACHE / "metadata.json").read_text())
        assert self.metadata["complete"], "Activation caching must finish first"
        self.shape = tuple(self.metadata["shape"])

    def batch(self, ix):
        length = int(self.lengths[ix].max())
        hidden = np.zeros((len(ix), length, self.shape[1]), np.float32)
        mm = np.memmap(CACHE / self.metadata.get("active_file", "hidden.f16"), dtype=np.dtype(self.metadata.get("active_dtype", "float16")), mode="r", shape=self.shape)
        for row, i in enumerate(ix):
            hidden[row, :self.lengths[i]] = mm[self.offsets[i]:self.offsets[i+1]]
        del mm
        mask = torch.arange(length)[None] < torch.from_numpy(self.lengths[ix])[:, None]
        return torch.from_numpy(hidden), mask


def refine_cache():
    """Cache four additional frozen layers after measured CPU budget analysis."""
    cached = CachedInputs()
    if cached.metadata.get("train_from") == TRAIN_FROM:
        print("Refined cache already complete", flush=True)
        return
    cfg = config()
    layers = torch.nn.ModuleList([ModernBertEncoderLayer(cfg, i) for i in range(FROZEN, TRAIN_FROM)])
    state = torch.load(PRETRAINED / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    for j, layer in enumerate(layers):
        prefix = f"model.layers.{FROZEN+j}."
        layer.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    layers.eval().requires_grad_(False)
    destination = CACHE / "hidden20.f32"
    if not destination.exists():
        with destination.open("wb") as handle:
            handle.truncate(int(np.prod(cached.shape))*4)
    done_path = CACHE / "refine_done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(len(cached.lengths), bool)
    started = time.monotonic()
    for step,ix in enumerate(batches(np.where(~done)[0], cached.lengths)):
        hidden, mask = cached.batch(ix)
        full, local, pos = masks(mask, cfg)
        with torch.inference_mode():
            for layer in layers:
                hidden = layer(hidden, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
        mm = np.memmap(destination, dtype=np.float32, mode="r+", shape=cached.shape)
        for row,i in enumerate(ix):
            mm[cached.offsets[i]:cached.offsets[i+1]] = hidden[row,:cached.lengths[i]].numpy()
        mm.flush()
        del mm
        done[ix] = True
        if step % 50 == 0:
            np.save(done_path, done)
            print(f"Additional frozen-layer cache {int(done.sum())}/{len(done)}, elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
    assert done.all()
    np.save(done_path, done)
    metadata = cached.metadata
    metadata.update({"train_from":TRAIN_FROM,"active_file":destination.name,"active_dtype":"float32"})
    dump(CACHE / "metadata.json", metadata)
    print("Additional frozen-layer cache completed", flush=True)


def evaluate(y, probabilities, data, indices):
    pred = probabilities.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=[0,1,2])),
            "report": classification_report(y, pred, labels=[0,1,2], target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
            "by_language": {lang: float(f1_score(y[data.language.iloc[indices].to_numpy()==lang],
                                                pred[data.language.iloc[indices].to_numpy()==lang],
                                                labels=[0,1,2], average="macro", zero_division=0))
                            for lang in sorted(data.language.unique())}}


def predict(model, cached, indices, batch_size=16):
    result = np.zeros((len(indices), 3), np.float32)
    locations = {int(i): n for n,i in enumerate(indices)}
    model.eval()
    with torch.inference_mode():
        for ix in batches(indices, cached.lengths, batch_size=batch_size):
            x, mask = cached.batch(ix)
            p = model(x, mask).softmax(-1).numpy()
            result[[locations[int(i)] for i in ix]] = p
    return result


def decision_correct(probabilities, weights, power):
    adjusted = probabilities / np.asarray(weights)[None,:]**power
    return (adjusted/adjusted.sum(1,keepdims=True)).astype(np.float32)


def baseline():
    train, _, folds = frames()
    features = np.load(CACHE / "pooled.npy", mmap_mode="r")[:len(train)]
    y = np.array([LABELS.index(v) for v in train.label])
    ix = np.where(~np.isin(folds.fold.to_numpy(),[0,1]))[0]
    valid = np.where(folds.fold.to_numpy() == 0)[0]
    results = []
    for c in (.01, .1, 1):
        model,scaler = fit_linear(features[ix],y[ix],c)
        p = model.predict_proba(scaler.transform(features[valid]))
        metrics = evaluate(y[valid], p, train, valid)
        metrics["C"] = c
        metrics["iterations"] = int(model.n_iter_.max())
        metrics["standardized"] = True
        results.append(metrics)
        print("Frozen backbone baseline", c, metrics["macro_f1"], flush=True)
    dump(ROOT / "artifacts/experiments/frozen_baseline.json", results)


def train_model(name, fold=0, epochs=4, lr=8e-5, weight_power=.35, final=False, batch_size=12, init_c=.1, resume=False, decision_power=0.):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train, _, folds = frames()
    out = ROOT / "artifacts/experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    y = np.array([LABELS.index(v) for v in train.label])
    if final:
        ix = np.arange(len(train))
    elif fold == 0:
        # Fold 1 stays completely outside development fitting and selection.
        ix = np.where(~np.isin(folds.fold.to_numpy(),[0,1]))[0]
    else:
        ix = np.where(folds.fold.to_numpy() != fold)[0]
    valid = np.array([], int) if final else np.where(folds.fold.to_numpy() == fold)[0]
    if not final:
        assert set(folds.group.iloc[ix]).isdisjoint(folds.group.iloc[valid])
    cached = CachedInputs()
    assert cached.metadata.get("train_from") == TRAIN_FROM, "Run refine-cache before training"
    model = initialize_tail()
    counts = np.bincount(y[ix], minlength=3)
    weights = (len(ix)/(3*counts))**weight_power
    weights /= weights.mean()
    if init_c > 0:
        features = np.load(CACHE / "pooled.npy", mmap_mode="r")
        warm_start,scaler = fit_linear(features[ix],y[ix],init_c,weights)
        with torch.no_grad():
            model.classifier[0].mean.copy_(torch.from_numpy(scaler.mean_).float())
            model.classifier[0].scale.copy_(torch.from_numpy(scaler.scale_).float())
            model.classifier[2].weight.copy_(torch.from_numpy(warm_start.coef_).float())
            model.classifier[2].bias.copy_(torch.from_numpy(warm_start.intercept_).float())
        del warm_start, features
    backbone_params = [p for n,p in model.named_parameters() if not n.startswith("classifier.")]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([{"params":backbone_params,"lr":lr}, {"params":head_params,"lr":lr*5}], weight_decay=.01)
    total_steps = 0
    for epoch in range(1, epochs+1):
        planned = batches(ix,cached.lengths,batch_size=batch_size,shuffle=True,seed=SEED+epoch,token_budget=1536)
        accumulated = 0
        for step,batch in enumerate(planned):
            accumulated += len(batch)
            if accumulated >= 32 or step == len(planned)-1:
                total_steps += 1
                accumulated = 0
    total_steps = max(1,total_steps)
    warmup = max(1, int(total_steps*.06))
    def schedule(step):
        if step < warmup:
            return max(.01, step/warmup)
        return max(.05, 1-(step-warmup)/(total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), reduction="none")
    settings = {"name":name,"seed":SEED,"fold":None if final else fold,"epochs":epochs,"lr":lr,
                "head_lr_multiplier":5,"weight_power":weight_power,"weights":weights.tolist(),
                "micro_batch_size":batch_size,"target_effective_batch_size":32,"frozen_layers":TRAIN_FROM,
                "train_rows":len(ix),"valid_rows":len(valid),"trainable_parameters":sum(p.numel() for p in model.parameters()),
                "cache":cached.metadata,"final":final,"head_initialization_C":init_c,
                "reserved_confirmation_fold":1 if not final and fold==0 else None,
                "head_standardization":"training-partition mean/std stored as fixed head buffers"}
    settings["decision_weight_power"] = decision_power
    dump(out / "config.json", settings)
    best, best_epoch, history = -1, 0, []
    first_epoch = 1
    if resume and (out / "recovery.pt").exists():
        recovery = torch.load(out / "recovery.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        torch.set_rng_state(recovery["torch_rng"])
        np.random.set_state(recovery["numpy_rng"])
        history = json.loads((out / "history.json").read_text())
        first_epoch = recovery["epoch"] + 1
        if not final:
            selected = max(history, key=lambda row:row["macro_f1"])
            best, best_epoch = selected["macro_f1"], selected["epoch"]
    started = time.monotonic()
    global_step = 0
    for epoch in range(first_epoch, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro = batches(ix, cached.lengths, batch_size=batch_size, shuffle=True, seed=SEED+epoch, token_budget=1536)
        accumulated, accumulated_weight, loss_sum, seen = 0, 0., 0., 0
        for step, batch in enumerate(micro):
            x, mask = cached.batch(batch)
            logits = model(x, mask)
            losses = loss_fn(logits, torch.tensor(y[batch]))
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Training produced nonfinite loss")
            (losses.sum()/32).backward()
            accumulated += len(batch)
            accumulated_weight += float(weights[y[batch]].sum())
            loss_sum += float(losses.detach().sum())
            seen += len(batch)
            if accumulated >= 32 or step == len(micro)-1:
                # Correct accumulation scale when bucketed microbatches cross the target size.
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(32/accumulated_weight)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                accumulated_weight = 0.
                global_step += 1
            if step % 50 == 0:
                print(f"{name} epoch {epoch}/{epochs} rows {seen}/{len(ix)} loss {loss_sum/seen:.4f} elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
        record = {"epoch":epoch,"train_loss":loss_sum/seen,"elapsed_seconds":time.monotonic()-started,"optimizer_steps":global_step}
        if not final:
            probabilities = predict(model, cached, valid)
            probabilities = decision_correct(probabilities,weights,decision_power)
            metrics = evaluate(y[valid], probabilities, train, valid)
            record.update(metrics)
            np.save(out / "last_valid_probabilities.npy", probabilities)
            np.save(out / "last_valid_indices.npy", valid)
            print(f"{name} epoch {epoch} VALID MACRO F1 {metrics['macro_f1']:.6f}", flush=True)
            if metrics["macro_f1"] > best:
                best, best_epoch = metrics["macro_f1"], epoch
                torch.save(model.state_dict(), out / "best.pt")
                np.save(out / "valid_probabilities.npy", probabilities)
                np.save(out / "valid_indices.npy", valid)
                dump(out / "best_metrics.json", metrics)
                errors = train.iloc[valid].copy()
                errors["prediction"] = np.array(LABELS)[probabilities.argmax(1)]
                errors["confidence"] = probabilities.max(1)
                errors["token_length"] = cached.lengths[valid]
                errors[errors.label != errors.prediction].to_csv(out / "errors.csv", index=False)
        history.append(record)
        dump(out / "history.json", history)
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,
                    "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state()}, out / "recovery.pt")
        if not final and epoch-best_epoch >= 2:
            print("Early stopping after two epochs without improvement", flush=True)
            break
    if final:
        torch.save(model.state_dict(), out / "final.pt")
    if final:
        print("Completed full-data fit",name,"epochs",epochs,"optimizer steps",global_step,flush=True)
    else:
        print("Completed",name,"best",best,"epoch",best_epoch,flush=True)


def infer(name):
    train, test, _ = frames()
    out = ROOT / "artifacts/experiments" / name
    settings = json.loads((out / "config.json").read_text())
    assert settings["final"] and settings["train_rows"] == len(train)
    assert (out / "final.pt").exists()
    model = initialize_tail()
    model.load_state_dict(torch.load(out / "final.pt", map_location="cpu", weights_only=True))
    cached = CachedInputs()
    assert cached.metadata.get("train_from") == TRAIN_FROM
    p = predict(model, cached, np.arange(len(train), len(train)+len(test)))
    p = decision_correct(p,settings["weights"],settings.get("decision_weight_power",0.))
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    assert (p >= 0).all() and (p <= 1).all()
    np.save(out / "test_probabilities.npy", p)
    predictions = pd.DataFrame({"id":test.id,"label":np.array(LABELS)[p.argmax(1)]})
    sample = pd.read_csv(ROOT / "dataset/sample_submission.csv", dtype=str)
    result = sample[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")
    destination = ROOT / "submission/submission.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = list(destination.parent.iterdir())
    if any(entry.name != "submission.csv" for entry in entries):
        raise ValueError("Submission folder contains other user files; no files removed")
    staged = out / "submission_staged.csv"
    result.to_csv(staged, index=False, encoding="utf-8")
    staged.replace(destination)
    dump(out / "inference_manifest.json", {
        "trained_checkpoint":str((out / "final.pt").relative_to(ROOT)),
        "checkpoint_sha256":hashlib.sha256((out / "final.pt").read_bytes()).hexdigest(),
        "submission_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
        "test_sha256":hashlib.sha256((ROOT / "dataset/test.csv").read_bytes()).hexdigest(),
        "backbone":settings["cache"]["backbone"],
        "rows":len(result),"label_counts":result.label.value_counts().to_dict(),
        "ensemble":False,"label_order":LABELS,
    })
    print("Saved", destination, "labels", result.label.value_counts().to_dict(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cache","refine-cache","baseline","train","infer"])
    parser.add_argument("--name", default="tail2_fold0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-power", type=float, default=.35)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--init-c", type=float, default=.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--decision-power",type=float,default=0.)
    args = parser.parse_args()
    seed_all()
    if args.command == "cache":
        cache_activations(args.max_length)
    elif args.command == "refine-cache":
        refine_cache()
    elif args.command == "baseline":
        baseline()
    elif args.command == "train":
        train_model(args.name,args.fold,args.epochs,args.lr,args.weight_power,args.final,args.batch_size,args.init_c,args.resume,args.decision_power)
    else:
        infer(args.name)


if __name__ == "__main__":
    main()
