"""Check diffusion boundaries, conditioning, and target/input separation."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from run_label_diffusion import Denoiser, Guidance, Schedule, sample_votes
from train_transformer import CLASSES, score, target_rows


class DiffusionTests(unittest.TestCase):
    def test_cosine_schedule_reaches_normal_prior(self):
        schedule = Schedule(100)
        self.assertLess(schedule.bar[-1].item(), 1e-5)
        self.assertTrue(torch.all(schedule.bar[1:] < schedule.bar[:-1]))
        self.assertEqual(schedule.posterior_variance[0].item(), 0)
        self.assertTrue(torch.all(schedule.posterior_variance >= 0))

    def test_final_reverse_step_with_oracle_noise_recovers_onehot(self):
        schedule = Schedule(100)
        truth = F.one_hot(torch.tensor([0, 2]), 3).float()
        prior = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        noise = torch.tensor([[1.0, -0.5, 0.3], [-1.0, 0.4, 0.7]])
        residual = schedule.noisy_residual(truth, prior, torch.zeros(2, dtype=torch.long), noise)
        recovered = schedule.reverse_mean(residual, prior, noise, 0) + prior
        torch.testing.assert_close(recovered, truth, atol=1e-4, rtol=1e-4)

    def test_sampling_needs_features_only_and_repeats_with_seed(self):
        torch.manual_seed(1)
        prior = Guidance(4, 3, 16)
        denoiser = Denoiser(4, 3, 16)
        features = torch.randn(5, 4)
        schedule = Schedule(8)
        a = sample_votes(denoiser, prior, schedule, features, 3, 4, 42)
        b = sample_votes(denoiser, prior, schedule, features, 3, 4, 42)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a.sum(1), 1)
        self.assertTrue(np.isfinite(a).all())

    def test_noise_loss_has_finite_gradients(self):
        model = Denoiser(4, 3, 16)
        schedule = Schedule(8)
        prior = torch.full((6, 3), 1 / 3)
        onehot = F.one_hot(torch.arange(6) % 3, 3).float()
        noise = torch.randn_like(prior)
        t = torch.arange(6)
        residual = schedule.noisy_residual(onehot, prior, t, noise)
        loss = F.mse_loss(model(torch.randn(6, 4), residual, prior, t), noise)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_action_target_never_enters_text(self):
        rows = [{"text": "Please send the file.", "decision": "audit", "label": 2}]
        mapped = target_rows(rows, "action", ["audit", "refuse"])
        self.assertEqual(mapped[0]["label"], 0)
        self.assertEqual(mapped[0]["text"], rows[0]["text"])
        self.assertEqual(rows[0]["label"], 2)
        result = score([0, 1], [0, 1], ["audit", "refuse", "engage_legitimate"])
        self.assertNotIn("binary_macro_f1", result)
        self.assertEqual(result["macro_f1_supported"], 1)
        self.assertLess(result["macro_f1"], 1)
        self.assertIn("binary_macro_f1", score([0, 1, 2], [0, 1, 2], CLASSES))


if __name__ == "__main__":
    unittest.main()
