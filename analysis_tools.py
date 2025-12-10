import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dataset import RandomJPEGCompression


def tta_predict(model, x):
    logits1 = model(x)
    logits2 = model(torch.flip(x, dims=[3]))
    return (logits1 + logits2) / 2.0


def predict_logits(model, x, use_tta):
    if isinstance(model, (list, tuple)):
        logits_sum = None
        for m in model:
            preds = tta_predict(m, x) if use_tta else m(x)
            logits_sum = preds if logits_sum is None else logits_sum + preds
        return logits_sum / float(len(model))
    return tta_predict(model, x) if use_tta else model(x)


def compute_patch_score_map(
    model,
    img,
    patch_size=128,
    stride=64,
    fake_class_index=1,
):
    model.eval()
    _, _, H, W = img.shape
    scores = []
    for y in range(0, H - patch_size + 1, stride):
        row_scores = []
        for x in range(0, W - patch_size + 1, stride):
            patch = img[:, :, y : y + patch_size, x : x + patch_size]
            with torch.no_grad():
                logits = model(patch)
                fake_logit = logits[:, fake_class_index].item()
            row_scores.append(fake_logit)
        scores.append(row_scores)
    return torch.tensor(scores)


def apply_perturbation(images, mode):
    if mode == "blur":
        return TF.gaussian_blur(images, kernel_size=3)
    if mode == "jpeg":
        jpeg = RandomJPEGCompression(quality_range=(40, 80), p=1.0)
        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()
        out = []
        for img in images:
            out.append(to_tensor(jpeg(to_pil(img.cpu()))))
        return torch.stack(out, dim=0).to(images.device)
    if mode == "noise":
        noise = torch.randn_like(images) * 0.02
        return (images + noise).clamp(0, 1)
    return images


@torch.no_grad()
def evaluate_under_perturbations(models, loader, device, use_tta=False):
    for m in models:
        m.eval()
    ref = models if len(models) > 1 else models[0]
    modes = ["clean", "blur", "jpeg", "noise"]
    results = {}
    for mode in modes:
        total = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            perturbed = apply_perturbation(
                images, None if mode == "clean" else mode
            )
            logits = predict_logits(ref, perturbed, use_tta)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        results[mode] = correct / float(total) if total > 0 else 0.0
    return results


@torch.no_grad()
def compare_models_under_perturbations(
    model_rgb,
    model_grad,
    loader,
    device,
    use_tta=False,
):
    acc_rgb = evaluate_under_perturbations([model_rgb], loader, device, use_tta)
    acc_grad = evaluate_under_perturbations([model_grad], loader, device, use_tta)
    delta = {k: acc_grad[k] - acc_rgb[k] for k in acc_rgb}
    return {"rgb": acc_rgb, "rgb_grad": acc_grad, "delta": delta}
