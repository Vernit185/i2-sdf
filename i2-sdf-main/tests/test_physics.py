import unittest
import torch
import torch.nn as nn
import importlib.util
import sys

# Import physics directly to bypass model/__init__.py which loads all ML frameworks
spec = importlib.util.spec_from_file_location("physics", "model/physics.py")
physics = importlib.util.module_from_spec(spec)
sys.modules["physics"] = physics
spec.loader.exec_module(physics)
import yaml

class MockImplicitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)
        # initialize weights for deterministic test
        nn.init.constant_(self.fc.weight, 1.0)
        nn.init.constant_(self.fc.bias, 1.0)
        
    def forward(self, x):
        return self.fc(x)
        
    def get_sdf_vals(self, x):
        return self.forward(x)

class TestPhysicsAwareOptimization(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device("cpu")
        
    def test_floor_estimation(self):
        # 4 cameras pointing somewhat along Z, up vector is -Y in world space
        poses = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
        # set Y column to (0, -1, 0)
        poses[:, :3, 1] = torch.tensor([0.0, -1.0, 0.0])
        
        pointcloud = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, -0.5, 0.0],
            [-1.0, -2.0, 1.0], # lowest point along Y is -2.0
            [0.0, 2.0, 0.0]
        ])
        
        v_up, h_floor = physics.estimate_floor_reference(poses, pointcloud, percentile=0.01)
        
        # Up vector should be (0, 1, 0)
        self.assertTrue(torch.allclose(v_up, torch.tensor([0.0, 1.0, 0.0])))
        # Height of points along (0,1,0) are -1.0, -0.5, -2.0, 2.0
        # 1st percentile is very close to min (-2.0)
        self.assertTrue(h_floor.item() < -1.9)
        
    def test_probe_generation(self):
        v_up = torch.tensor([0.0, 1.0, 0.0])
        h_floor = torch.tensor(0.0)
        candidate = torch.tensor([[0.0, 1.0, 0.0]]) # 1 unit above floor
        
        probes = physics.generate_ground_probe_points(
            candidate, v_up, h_floor, 
            num_samples_per_col=3, range_min=0.0, range_max=1.0
        )
        # Should generate 3 points: [0, 1, 0], [0, 0.5, 0], [0, 0, 0]
        self.assertEqual(probes.shape, (3, 3))
        self.assertTrue(torch.allclose(probes[0], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(probes[2], torch.tensor([0.0, 0.0, 0.0])))
        
    def test_synthetic_grounding_loss(self):
        # SDF > 0 means empty space. SDF <= 0 means inside object.
        
        # Scenario 1: All points in column are empty space (SDF = 1.0)
        floating_sdf = torch.ones(2, 4)
        loss_float = physics.compute_grounding_loss(floating_sdf, 2, 4, beta=10.0)
        self.assertTrue(loss_float.item() > 0.0)
        
        # Scenario 2: At least one point in column is solid (SDF = -1.0)
        grounded_sdf = torch.ones(2, 4)
        grounded_sdf[:, -1] = -1.0
        loss_grounded = physics.compute_grounding_loss(grounded_sdf, 2, 4, beta=10.0)
        self.assertTrue(loss_grounded.item() < 1e-4) # Loss should be basically 0
        
    def test_strict_gradient_propagation(self):
        # Verify physics loss produces gradients correctly
        model = MockImplicitMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        ground_points = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0]
        ], requires_grad=False)
        
        # initial params
        weight_before = model.fc.weight.clone().detach()
        
        optimizer.zero_grad()
        
        # SDF Query
        sdf_vals = model.get_sdf_vals(ground_points)
        
        # Physics loss
        loss = physics.compute_grounding_loss(sdf_vals, num_candidates=1, num_samples_per_col=2)
        
        # Verify loss requires grad
        self.assertTrue(loss.requires_grad)
        
        # Backward
        loss.backward()
        
        # Gradients are not None and finite
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertTrue(torch.all(torch.isfinite(model.fc.weight.grad)))
        self.assertTrue(torch.norm(model.fc.weight.grad) > 0.0)
        
        # Update
        optimizer.step()
        
        # Verify params changed
        self.assertFalse(torch.allclose(model.fc.weight, weight_before))
        
        # Verify no NaNs
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        
    def test_config_baseline_identity(self):
        # Baseline ground_weight = 0.0 should not compute physics loss
        with open('config/synthetic.yml', 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        self.assertEqual(cfg_dict.get('loss', {}).get('ground_weight', 0.0), 0.0)
        
        with open('config/synthetic_physics.yml', 'r') as f:
            cfg_phys_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        self.assertEqual(cfg_phys_dict.get('loss', {}).get('ground_weight', 0.0), 0.1)

if __name__ == '__main__':
    unittest.main()
