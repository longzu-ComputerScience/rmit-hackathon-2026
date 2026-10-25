"""Check cached execution against direct approved-backbone execution."""
from __future__ import annotations

import json

import numpy as np
import torch

import competition as c


def main():
    c.seed_all()
    cached = c.CachedInputs()
    data = np.load(c.CACHE / "tokens.npz")
    pooled = np.load(c.CACHE / "pooled.npy", mmap_mode="r")
    # Test both short and truncated long inputs; different padding lengths too.
    train, test, _ = c.frames()
    n_train = len(train)
    indices = np.unique(np.array([int(np.argmin(cached.lengths)), int(np.argmax(cached.lengths)),
                                  n_train, n_train+57,
                                  n_train+int(np.argmax(cached.lengths[n_train:]))]))
    n = int(cached.lengths[indices].max())
    ids = torch.zeros((len(indices), n), dtype=torch.long)
    mask = torch.arange(n)[None] < torch.from_numpy(cached.lengths[indices])[:, None]
    for row,i in enumerate(indices):
        ids[row,:cached.lengths[i]] = torch.from_numpy(data["tokens"][cached.offsets[i]:cached.offsets[i+1]].astype(np.int64))
    model = c.AutoModel.from_pretrained(c.PRETRAINED, config=c.config(), local_files_only=True, attn_implementation="sdpa")
    model.eval()
    full, local, pos = c.masks(mask, model.config)
    with torch.inference_mode():
        hidden = model.embeddings(ids)
        for layer in model.layers[:c.FROZEN]:
            hidden = layer(hidden, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
        direct = hidden.half().float()
        for layer in model.layers[c.FROZEN:c.TRAIN_FROM]:
            direct = layer(direct, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
        stored, stored_mask = cached.batch(indices)
        valid = mask.unsqueeze(-1).expand_as(stored)
        max_boundary_error = float((direct[valid]-stored[valid]).abs().max())
        boundary_relative_rms = float((direct[valid]-stored[valid]).square().mean().sqrt()/stored[valid].square().mean().sqrt())
        print("Cache boundary numerical diagnostics:",{"max_absolute":max_boundary_error,"relative_rms":boundary_relative_rms},flush=True)
        # Unnormalized hidden-state magnitudes vary by token. Use relative RMS
        # for this float16 boundary; the stricter output-probability and label
        # checks below determine inference equivalence.
        assert boundary_relative_rms < np.finfo(np.float16).eps*4, boundary_relative_rms
        boundary = direct.clone()
        for layer in model.layers[c.TRAIN_FROM:]:
            direct = layer(direct, attention_mask=full, sliding_window_mask=local, position_ids=pos)[0]
        direct_pool = c.pool(model.final_norm(direct), mask).numpy()
        max_pool_error = float(np.max(np.abs(direct_pool-pooled[indices])))
        print("Pretrained pooled feature max error:",max_pool_error,flush=True)
        assert max_pool_error < .02, max_pool_error
        tail = c.initialize_tail().eval()
        final_path = c.ROOT / "artifacts/experiments/final_single_model/final.pt"
        encoder_change = None
        if final_path.is_file():
            trained = torch.load(final_path, weights_only=True, map_location="cpu")
            initial = tail.state_dict()
            encoder_change = float(sum((trained[k]-initial[k]).square().sum() for k in initial if k.startswith("layers.")))**.5
            assert encoder_change > 0, "Encoder weights did not change during fine-tuning"
            tail.load_state_dict(trained)
        p_cached = tail(stored, stored_mask).softmax(-1)
        direct_hidden = boundary
        p_direct = tail(direct_hidden, mask).softmax(-1)
        max_probability_error = float((p_cached-p_direct).abs().max())
        print("Trained probability max error:",max_probability_error,flush=True)
        assert max_probability_error < .003, max_probability_error
        assert torch.equal(p_cached.argmax(1), p_direct.argmax(1))
        saved = c.ROOT / "artifacts/experiments/final_single_model/test_probabilities.npy"
        if saved.is_file():
            saved_probabilities = np.load(saved)
            test_rows = indices >= n_train
            settings = json.loads((saved.parent / "config.json").read_text())
            corrected = c.decision_correct(p_cached.numpy(),settings["weights"],settings.get("decision_weight_power",0.))
            corrected_direct = c.decision_correct(p_direct.numpy(),settings["weights"],settings.get("decision_weight_power",0.))
            assert np.array_equal(corrected.argmax(1),corrected_direct.argmax(1))
            saved_error = float(np.max(np.abs(saved_probabilities[indices[test_rows]-n_train]-corrected[test_rows])))
            assert saved_error < .003, saved_error
            assert np.array_equal(saved_probabilities[indices[test_rows]-n_train].argmax(1),corrected_direct[test_rows].argmax(1))
        else:
            saved_error = None
    report = {"passed":True,"indices":indices.tolist(),"max_frozen_boundary_error":max_boundary_error,
              "frozen_boundary_relative_rms":boundary_relative_rms,
              "max_pooled_feature_error":max_pool_error,"max_probability_error":max_probability_error,
              "identical_predictions":True,"trained_checkpoint_checked":final_path.is_file(),
              "max_saved_test_probability_error":saved_error,"encoder_parameter_l2_change":encoder_change}
    c.dump(c.ROOT / "artifacts/forward_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
