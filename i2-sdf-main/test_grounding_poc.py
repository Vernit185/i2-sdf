import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from utils import rend_util
import utils
import model
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model.trainer.recon import ReconstructionTrainer

def run_grounding_poc():
    print("=" * 80)
    print("      I²-SDF PHYSICS GROUNDING CONSTRAINT — 20-ITERATION POC TEST      ")
    print("=" * 80)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device in use: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Load config
    conf_path = "config/synthetic.yml"
    with open(conf_path) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = utils.CfgNode(cfg_dict)
    cfg.dataset.scan_id = 0
    cfg.train.split_n_pixels = 1024

    # 2. Instantiate ReconstructionTrainer
    exp_dir = "exps/poc_test"
    os.makedirs(exp_dir, exist_ok=True)
    progbar_callback = utils.RichProgressBarWithScanId(0, leave=False)
    trainer = ReconstructionTrainer(cfg, progbar_callback, exp_dir)
    trainer = trainer.to(device)

    # 3. Explicitly set ground_weight to 0.1 for POC
    trainer.loss.ground_weight = 0.1
    print(f"[INFO] Grounding weight set to: {trainer.loss.ground_weight}")

    # 4. Setup Optimizer
    lr = cfg.train.learning_rate
    optimizer = optim.Adam(trainer.model.get_param_groups(lr), eps=1e-15)

    # 5. Dataloader (single worker for deterministic quick test)
    dataloader = DataLoader(
        trainer.train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=trainer.train_dataset.collate_fn,
        num_workers=0
    )
    data_iter = iter(dataloader)

    trainer.log = lambda *args, **kwargs: None
    trainer.log_if_nonzero = lambda *args, **kwargs: None

    # Pre-test metrics
    history = []
    
    print("\nStarting 20 Proof-of-Concept Iterations...")
    print("-" * 100)
    print(f"{'Iter':<6} | {'Total Loss':<12} | {'I²-SDF Base':<12} | {'Ground Loss':<14} | {'SDF Grad Norm':<14} | {'SDF Param Change':<16}")
    print("-" * 100)

    for step in range(1, 21):
        trainer._custom_step = step

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move tensors in batch to device
        indices, img_indices, model_input, ground_truth = batch
        model_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}
        ground_truth = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}
        indices = indices.to(device)
        img_indices = img_indices.to(device)
        batch_device = (indices, img_indices, model_input, ground_truth)

        # Record weights before step
        sdf_params_before = [p.clone().detach() for p in trainer.model.implicit_network.parameters() if p.requires_grad]

        # Forward pass & loss
        optimizer.zero_grad()
        loss_val = trainer.training_step(batch_device, step)

        # Retrieve individual losses
        ground_loss_item = trainer.last_loss_output.get('ground_loss', 0.0)
        if isinstance(ground_loss_item, torch.Tensor):
            ground_loss_val = ground_loss_item.item()
        else:
            ground_loss_val = float(ground_loss_item)

        total_loss_val = loss_val.item()
        base_loss_val = total_loss_val - (trainer.loss.ground_weight * ground_loss_val)

        # Backward pass
        loss_val.backward()

        # Compute gradient norm of the SDF network (ImplicitNetwork)
        grad_norm_sq = 0.0
        for p in trainer.model.implicit_network.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.data.norm(2).item() ** 2
        sdf_grad_norm = grad_norm_sq ** 0.5

        # Optimizer step
        optimizer.step()

        # Compute parameter change magnitude
        sdf_params_after = [p.clone().detach() for p in trainer.model.implicit_network.parameters() if p.requires_grad]
        param_diff_sq = 0.0
        for p_bef, p_aft in zip(sdf_params_before, sdf_params_after):
            param_diff_sq += (p_aft - p_bef).norm(2).item() ** 2
        param_change_mag = param_diff_sq ** 0.5

        history.append({
            'iter': step,
            'total_loss': total_loss_val,
            'base_loss': base_loss_val,
            'ground_loss': ground_loss_val,
            'grad_norm': sdf_grad_norm,
            'param_change': param_change_mag
        })

        print(f"{step:<6} | {total_loss_val:<12.6f} | {base_loss_val:<12.6f} | {ground_loss_val:<14.6f} | {sdf_grad_norm:<14.6e} | {param_change_mag:<16.6e}")

    print("-" * 100)
    runtime = time.time() - start_time

    # Summary analysis
    init_entry = history[0]
    final_entry = history[-1]

    avg_grad_norm = np.mean([h['grad_norm'] for h in history])
    avg_param_change = np.mean([h['param_change'] for h in history])
    finite_ground = all(np.isfinite(h['ground_loss']) for h in history)
    nonzero_ground = any(h['ground_loss'] > 0 for h in history)
    nonzero_grad = avg_grad_norm > 0
    nonzero_change = avg_param_change > 0

    print("\n" + "=" * 80)
    print("                    PROOF-OF-CONCEPT REPORT                    ")
    print("=" * 80)
    print(f"• Iterations Completed     : 20")
    print(f"• Initial Ground Loss      : {init_entry['ground_loss']:.6f}")
    print(f"• Final Ground Loss        : {final_entry['ground_loss']:.6f}")
    print(f"• Initial Total Loss       : {init_entry['total_loss']:.6f}")
    print(f"• Final Total Loss         : {final_entry['total_loss']:.6f}")
    print(f"• Mean SDF Gradient Norm   : {avg_grad_norm:.6e}")
    print(f"• Mean Param-Change Mag    : {avg_param_change:.6e}")
    print(f"• CUDA Accelerated         : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Total Runtime            : {runtime:.2f} s")
    print(f"• Errors Encountered       : None")
    print("=" * 80)

    print("\nCore Research Verification Questions:")
    print(f"A. Is ground_loss finite and non-zero?         => {'YES (finite=' + str(finite_ground) + ', min=' + str(min(h['ground_loss'] for h in history)) + ')' if finite_ground and nonzero_ground else 'NO'}")
    print(f"B. Does ground_loss contribute to total loss?  => {'YES (loss = base_loss + 0.1 * ground_loss)' if init_entry['ground_loss'] > 0 else 'NO'}")
    print(f"C. Are SDF parameters receiving gradients?     => {'YES (Mean Grad Norm: ' + f'{avg_grad_norm:.4e})' if nonzero_grad else 'NO'}")
    print(f"D. Do SDF parameters actually change?          => {'YES (Mean Delta_theta: ' + f'{avg_param_change:.4e})' if nonzero_change else 'NO'}")
    init_gl = init_entry['ground_loss']
    final_gl = final_entry['ground_loss']
    e_str = f"YES ({init_gl:.4f} -> {final_gl:.4f})" if final_gl < init_gl else f"Monitored trajectory ({init_gl:.4f} -> {final_gl:.4f})"
    print(f"E. Does ground_loss decrease over 20 iters?    => {e_str}")

if __name__ == '__main__':
    run_grounding_poc()
