# train_sv_b.py (Fixed F1 Early Stop & Save)
# -*- coding: utf-8 -*-
import os, argparse, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入 Scheme B 的模块
from tc_utils import (
    set_seed, TraceDataset, collate_multi, vocab_sizes_from_meta,
    derive_keep_types, evaluate_detailed, print_per_class_reports,
    collect_per_class_reports, save_run_summary, save_ckpt
)
from tc_model import TraceClassifier

def main():
    ap = argparse.ArgumentParser("SV Scheme B Training (No Host Graph)")
    ap.add_argument("--data-root", default="dataset/tianchi/processed_0111")
    # 修改保存路径，方便管理
    ap.add_argument("--save-dir",  default="dataset/tianchi/processed_0111/f1earlystop")
    ap.add_argument("--save_pt",   default="dataset/tianchi/processed_0111/f1earlystop/best_model.pt")
    
    ap.add_argument("--type_min_support", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true", default=False)
    ap.add_argument("--early_stop_patience", type=int, default=10) # 建议稍微调大一点 patience
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    args = ap.parse_args()

    set_seed(args.seed)
    # 确保输出目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_pt), exist_ok=True)

    tr = os.path.join(args.data_root, "train.jsonl")
    va = os.path.join(args.data_root, "val.jsonl")
    te = os.path.join(args.data_root, "test.jsonl")
    
    api_sz, st_sz, node_sz, type_names, ctx_dim = vocab_sizes_from_meta(args.data_root)
    print(f"[Init] Vocab: API={api_sz}, Status={st_sz}, Node={node_sz}, Types={len(type_names)}, Ctx={ctx_dim}")

    ds_fit = TraceDataset(tr, task="multihead", fit_stats=True)
    stats  = ds_fit.stats
    keep_types = derive_keep_types(ds_fit.items, args.type_min_support)
    
    ds_tr = TraceDataset(tr, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    ds_va = TraceDataset(va, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    ds_te = TraceDataset(te, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    
    mk = lambda ds, shuf: DataLoader(ds, batch_size=args.batch, shuffle=shuf, collate_fn=collate_multi, num_workers=4)
    tr_loader, va_loader, te_loader = mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

    device = torch.device(args.device)
    model = TraceClassifier(
        api_sz, st_sz, node_sz, 
        n_types=len(type_names), 
        ctx_dim=ctx_dim
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        model.train(train)
        tot = 0.0
        iterator = tqdm(loader, leave=False) if not args.no_progress else loader
        for g, lab, _, _ in iterator:
            g = g.to(device)
            yb = lab["y_bin"].float().to(device)
            yc = lab["y_c3"].to(device)
            yt = lab["y_type"].to(device)
            mt = lab["m_type"].to(device)

            out = model(g)
            l1 = bce(out["logit_bin"], yb)
            # 在 Scheme B 中 y_c3 只有 0/1，这里实际上重复了 l1，但为了兼容性保留
            l2 = ce(out["logits_c3"], yc) 
            
            if mt.sum() > 0:
                l3 = (ce(out["logits_type"], yt) * mt).sum() / (mt.sum() + 1e-6)
            else:
                l3 = l1 * 0.0
            
            loss = 0.1 * l1 + 0.2 * l2 + 0.7 * l3

            if train:
                opt.zero_grad(); loss.backward(); 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            tot += float(loss.item())
        return tot / max(1, len(loader))

    # === [修复 1] F1 是越大越好，所以初始 best 要设为 0.0 (不是无穷大) ===
    best = 0.0 
    best_state = None
    no_improve = 0

    print("🚀 Start Training...")
    for ep in range(1, args.epochs + 1):
        trL = run_epoch(tr_loader, train=True)
        vaL = run_epoch(va_loader, train=False) # 这里的 vaL 仅作参考日志
        
        # 计算 F1
        metrics = evaluate_detailed(model, va_loader, device, type_names, keep_types=keep_types)
        current_f1 = metrics["type_f1"]
        
        print(f"[Epoch {ep:02d}] Loss: {trL:.4f} | Val F1: {current_f1:.4f} (Best: {best:.4f})")

        # === [修复 2] 正确的 F1 早停逻辑 ===
        if current_f1 > best + args.early_stop_min_delta:
            best = current_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"   ✨ New Best F1! ({best:.4f})")
        else:
            no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(f"🛑 [EarlyStop] No F1 improvement for {no_improve} epochs. Stopping.")
                break

    # 加载最佳权重进行测试
    if best_state is not None:
        print("🔙 Loading best model state for testing...")
        model.load_state_dict(best_state)
    else:
        print("⚠️ Warning: No best state found (maybe F1 never > 0?). Using last epoch state.")

    print("\n>>> Final Test <<<")
    metrics = evaluate_detailed(model, te_loader, device, type_names, keep_types=keep_types)
    print_per_class_reports(model, te_loader, device, type_names, keep_types=keep_types)

    # 保存结果
    reports = collect_per_class_reports(model, te_loader, device, type_names, keep_types=keep_types)
    
    save_run_summary(save_dir=args.save_dir, args=vars(args), data_root=args.data_root,
                     metrics_dict=metrics, reports_dict=reports,
                     stats=stats, type_names=type_names, keep_types=keep_types)
    
    # === [修复 3] 显式保存模型权重 ===
    save_ckpt(
        path=args.save_pt, 
        model_state=model.state_dict(), 
        stats=stats, 
        args_dict=vars(args), 
        type_names=type_names, 
        keep_types=keep_types
    )
    
    print(f"✅ Training Finished. Model saved to: {args.save_pt}")

if __name__ == "__main__":
    main()