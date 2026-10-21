"""MedSAM brain-mask refinement, shared by the pipeline and the GPU worker.

load() builds the predictor once; refine() runs it slice by slice with a box
prompt from the BET mask. The GPU worker keeps the predictor between jobs.
"""
import numpy as np
from scipy import ndimage


def load(checkpoint):
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry['vit_b'](checkpoint=None)
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    sam.load_state_dict(state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device).eval()
    return SamPredictor(sam)


def refine(pred, norm, bet_mask):
    """Refine each slice of the BET mask with a box prompt around it."""
    out = np.zeros_like(bet_mask)
    H, W = norm.shape[1], norm.shape[2]
    for i in range(norm.shape[0]):
        if bet_mask[i].sum() < 50:
            continue
        sl = (norm[i]*255).astype(np.uint8)
        rgb = np.stack([sl]*3, axis=-1)
        rows = np.any(bet_mask[i], axis=1); cols = np.any(bet_mask[i], axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
        box = np.array([max(0, cmin-3), max(0, rmin-3),
                        min(W-1, cmax+3), min(H-1, rmax+3)])
        pred.set_image(rgb)
        masks, _, _ = pred.predict(box=box[None, :], multimask_output=False)
        out[i] = masks[0]
    return out & ndimage.binary_dilation(bet_mask, iterations=2)
