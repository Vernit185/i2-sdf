import torch
import torch.nn.functional as F

def estimate_floor_reference(poses, pointcloud, percentile=0.01, device=None):
    """
    Heuristically estimates a floor reference height along a derived upward axis.
    This is an explicitly defined heuristic, not guaranteed ground-truth floor detection.
    
    Args:
        poses: (N, 4, 4) tensor of camera poses.
        pointcloud: (M, 3) tensor of 3D pointcloud points.
        percentile: float, the height percentile to consider as the floor reference.
        device: torch.device
        
    Returns:
        v_up: (3,) tensor, the estimated world-up direction vector.
        h_floor: scalar tensor, the estimated floor height along v_up, or -1.0 if unavailable.
    """
    if device is None:
        device = poses.device

    # Estimate up direction from cameras
    # Assuming pose[:, :3, 1] is the camera up vector in world space, the negative is often world up in typical SDF setups.
    cams_up = -poses[:, :3, 1].to(device)
    v_up = F.normalize(cams_up.mean(dim=0), dim=0)
    
    h_floor = torch.tensor(-1.0, device=device)
    if pointcloud is not None and isinstance(pointcloud, torch.Tensor) and pointcloud.numel() > 0:
        pc_cuda = pointcloud[:, :3].to(device)
        pc_heights = torch.matmul(pc_cuda, v_up)
        h_floor = torch.quantile(pc_heights, percentile)
        
    return v_up, h_floor

def generate_ground_probe_points(candidate_points, v_up, h_floor, num_samples_per_col=16, range_min=0.05, range_max=0.95):
    """
    Generates vertical probe points down towards the estimated floor.
    
    Args:
        candidate_points: (B, 3) tensor of candidate surface points.
        v_up: (3,) tensor, world-up vector.
        h_floor: scalar tensor, estimated floor height.
        num_samples_per_col: int, number of vertical samples per candidate.
        range_min: float, starting ratio of distance to floor (0=at point, 1=at floor).
        range_max: float, ending ratio of distance to floor.
        
    Returns:
        probe_points: (B * num_samples_per_col, 3) tensor of probe points.
    """
    B = candidate_points.size(0)
    device = candidate_points.device
    
    # Height of each candidate point above estimated floor along vertical axis
    cand_h = torch.matmul(candidate_points, v_up)
    height_above_floor = (cand_h - h_floor).clamp(min=1e-3)

    # Sample vertical points along downward column towards floor
    t = torch.linspace(range_min, range_max, num_samples_per_col, device=device).view(1, num_samples_per_col, 1)
    v_down = -v_up.view(1, 1, 3)
    ground_probe = candidate_points.unsqueeze(1) + t * (height_above_floor.view(B, 1, 1) * v_down)
    
    return ground_probe.reshape(-1, 3)

def compute_grounding_loss(ground_sdf, num_candidates=None, num_samples_per_col=None, beta=10.0):
    """
    Calculates a differentiable penalty for candidate geometry floating above the floor.
    Uses logsumexp for numerical stability.
    
    Args:
        ground_sdf: (B * num_samples_per_col, 1) or (B, num_samples_per_col) tensor of SDF values.
        num_candidates: B
        num_samples_per_col: K
        beta: float, sharpness of soft minimum.
        
    Returns:
        ground_loss: scalar tensor.
    """
    if ground_sdf.dim() == 2 and ground_sdf.shape[1] == 1:
        if num_candidates is not None and num_samples_per_col is not None:
            ground_sdf = ground_sdf.view(num_candidates, num_samples_per_col)
        else:
            return (F.relu(ground_sdf.view(-1)) ** 2).mean()

    # Numerically stable soft minimum along vertical column dimension using logsumexp
    soft_min = -(1.0 / beta) * torch.logsumexp(-beta * ground_sdf, dim=-1)
    ground_loss = (F.relu(soft_min) ** 2).mean()
    return ground_loss

def compute_grounding_metrics(ground_sdf, num_candidates=None, num_samples_per_col=None, beta=10.0, satisfaction_threshold=0.02):
    """
    Computes quantitative grounding metrics.
    
    Returns:
        metrics: dict of metrics.
    """
    if ground_sdf.dim() == 2 and ground_sdf.shape[1] == 1:
        if num_candidates is not None and num_samples_per_col is not None:
            ground_sdf = ground_sdf.view(num_candidates, num_samples_per_col)
        else:
            ground_sdf = ground_sdf.view(-1, 1)
            
    with torch.no_grad():
        soft_min = -(1.0 / beta) * torch.logsumexp(-beta * ground_sdf, dim=-1)
        violations = F.relu(soft_min)
        
        mean_violation = violations.mean().item()
        max_violation = violations.max().item()
        
        # A region satisfies grounding if its soft minimum SDF > 0 is less than or equal to threshold
        # (i.e. it hits a surface -> SDF <= 0 -> SoftMin <= 0, or very close to it)
        satisfied = (violations <= satisfaction_threshold)
        satisfaction_pct = satisfied.float().mean().item() * 100.0
        floating_count = (~satisfied).sum().item()
        
    return {
        'satisfaction_pct': satisfaction_pct,
        'mean_violation': mean_violation,
        'max_violation': max_violation,
        'floating_count': floating_count
    }
