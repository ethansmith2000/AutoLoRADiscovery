import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def get_least_squares_solution(weight_matrix, orig_up, orig_down, **kwargs):
    rank = kwargs.get("rank")
    equal_norm_penalty = kwargs.get("equal_norm_penalty")
    steps = kwargs.get("steps")
    start_lr = kwargs.get("start_lr")
    end_lr = kwargs.get("end_lr")
    device = kwargs.get("device")
    beta1 = kwargs.get("beta1")
    beta2 = kwargs.get("beta2")
    weight_decay = kwargs.get("weight_decay")
    l = kwargs.get("l")
    l1_factor = kwargs.get("l1_factor")
    match_norm_penalty = kwargs.get("match_norm_penalty")
    error_threshold = kwargs.get("error_threshold")
    allowed_num_tries = kwargs.get("allowed_num_tries")

    with torch.no_grad():
        orig_rank = orig_up.shape[1]
        orig_up_norm = orig_up.norm()
        orig_down_norm = orig_down.norm()
        target_up_norm = (orig_up_norm * (rank / orig_rank) ** 0.5).to(device)
        target_down_norm = (orig_down_norm * (rank / orig_rank) ** 0.5).to(device)

    finished = False
    num_tries = 0
    while not finished:
        matrix = weight_matrix.float().to(device).requires_grad_(False)
        up = torch.randn(matrix.shape[0], rank, device=device)
        down = torch.randn(matrix.shape[1], rank, device=device)
        up = (up / up.norm() * target_up_norm).requires_grad_(True)
        down = (down / down.norm() * target_down_norm).requires_grad_(True)
        print("UP NORM", up.norm().item(), "DOWN NORM", down.norm().item())
        optimizer = torch.optim.AdamW(
            [up, down],
            lr=start_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        losses = []
        for i in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = up @ down.T
            lN_loss = (prediction - matrix).pow(l).mean() ** (1 / l)
            l1_loss = F.l1_loss(prediction, matrix)
            loss = lN_loss
            loss += l1_loss * l1_factor
            if equal_norm_penalty > 0:
                loss += equal_norm_penalty * F.mse_loss(up.norm().unsqueeze(0), down.norm().unsqueeze(0))
            if match_norm_penalty > 0:
                loss += match_norm_penalty * F.mse_loss(up.norm().unsqueeze(0), target_up_norm.unsqueeze(0))
                loss += match_norm_penalty * F.mse_loss(down.norm().unsqueeze(0), target_down_norm.unsqueeze(0))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            lr = start_lr * (1 - (i / steps)) + end_lr * (i / steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        with torch.no_grad():
            diff = torch.abs(prediction - matrix)
            pos = diff.argmax()
            max_diff = diff.max().item()
            max_rel_error = (max_diff / matrix.flatten()[pos].item() + 1e-6) * 100

        print(
            f"LOSS: {losses[-1]}, ABS MEAN {prediction.abs().mean():.7f} MAX DIFF: {max_diff:.7f}, MAX REL ERROR: {max_rel_error:.7f}%")
        print(
            f"UP NORM {up.norm().item():.7f}, TARGET UP NORM {target_up_norm.item():.7f}, DOWN NORM {down.norm().item():.7f}, TARGET DOWN NORM {target_down_norm.item():.7f}")
        if abs(max_rel_error) < error_threshold:
            finished = True
        else:
            if num_tries < allowed_num_tries:
                print("RETRYING")
            else:
                raise ValueError("FAILED")
            num_tries += 1

    return up.detach().cpu(), down.detach().cpu().T


def svd(full_mat, lora_up, lora_down, rank, **kwargs):
    device = kwargs.get("device")
    with torch.no_grad():
        full_mat = full_mat.to(device).float()
        u, s, v = torch.linalg.svd(full_mat)
        min_dim = min(u.shape[1], v.shape[0])
        s = s[:min_dim]
        u = u[:, :min_dim] @ (s ** 0.5).diag()
        v = (s ** 0.5).diag() @ v[:min_dim, :]

        new_lora_up = u[:, :rank]
        new_lora_down = v[:rank, :]

        diff = torch.abs(new_lora_up @ new_lora_down - full_mat)
        pos = diff.argmax()
        max_diff = diff.max().item()
        max_rel_error = (max_diff / full_mat.flatten()[pos].item() + 1e-6) * 100
        print(f"MAX DIFF: {max_diff:.7f}, MAX REL ERROR: {max_rel_error:.7f}%")

    return new_lora_up.cpu(), new_lora_down.cpu()


def simple_expand(full_mat, lora_up, lora_down, rank, **kwargs):
    new_lora_up = torch.zeros((lora_up.shape[0], rank), device=lora_up.device)
    new_lora_down = torch.zeros((rank, lora_down.shape[1]), device=lora_down.device)
    new_lora_up[:, :rank] = lora_up
    new_lora_down[:rank, :] = lora_down

    return new_lora_up, new_lora_down


def change_lora_rank(state_dict,
                     rank=64,
                     down_name="lora_down",
                     up_name="lora_up",
                     equal_norm_penalty=0.0,  # 0.0001,
                     match_norm_penalty=0.0001,  # 0.0001,
                     steps=200,
                     start_lr=0.05,
                     end_lr=0.001,
                     device="cuda",
                     beta1=0.9,
                     beta2=0.98,
                     weight_decay=0.001,
                     l=2,
                     l1_factor=0.0,
                     error_threshold=10,
                     allowed_num_tries=20,
                     method="auto"  # ["auto", "svd", "optimization", "zero_pad"]
                     ):
    """
    :param state_dict: a torch state_dict
    :param rank: desired rank
    :param down_name: naming convention for down component in state dict
    :param up_name: naming convention for up component in state dict
    :param equal_norm_penalty: penalty to enforce equal norm of up and down components
    :param match_norm_penalty: penalty to enforce the norm of up and down components to match the original norm, scaled by rank
    :param steps: number of optimization steps
    :param start_lr: start learning rate
    :param end_lr: end learning rate
    :param device: device
    :param beta1: adam beta1
    :param beta2: adam beta2
    :param weight_decay: weight decay
    :param l: l-norm, default to l2
    :param l1_factor: l1 factor for mixed loss
    :param error_threshold: error threshold in percent, if the worst observed error is > error_threshold, retry
    :param allowed_num_tries: number of retries before raising an error
    :param method: method to use, auto will choose between svd and optimization based on the size of the matrices
    :return: updated state_dict
    """

    lora_keys = [k for k in state_dict.keys() if down_name in k]
    weights_with_lora = [k.split(".lora")[0] for k in lora_keys]
    new_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
    cur_rank = state_dict[lora_keys[0]].shape[0]

    kwargs = dict(
        rank=rank, steps=steps,
        start_lr=start_lr, end_lr=end_lr,
        device=device, match_norm_penalty=match_norm_penalty,
        beta1=beta1, beta2=beta2,
        weight_decay=weight_decay,
        l=l,
        l1_factor=l1_factor,
        error_threshold=error_threshold,
        equal_norm_penalty=equal_norm_penalty,
        allowed_num_tries=allowed_num_tries,
    )

    for key in tqdm(weights_with_lora):
        with torch.no_grad():
            lora_down_key, lora_up_key = key + f".{down_name}.weight", key + f".{up_name}.weight"
            lora_down, lora_up = state_dict[lora_down_key], state_dict[lora_up_key]
            if "conv" in key:
                # out, rank, 1, 1
                lora_up = lora_up.reshape(lora_up.shape[0], -1)
                # rank, in, k, k
                k = lora_down.shape[-1]
                lora_down = lora_down.reshape(lora_down.shape[0], -1)
            full_mat = lora_up @ lora_down
            print(key, full_mat.shape)

        if method == "optimization":
            new_lora_up, new_lora_down = get_least_squares_solution(full_mat, lora_up, lora_down, **kwargs)
        elif method == "svd":
            new_lora_up, new_lora_down = svd(full_mat, lora_up, lora_down, **kwargs)
        elif method == "zero_pad":
            assert rank > cur_rank
            new_lora_up, new_lora_down = simple_expand(full_mat, lora_up, lora_down, **kwargs)
        elif method == "auto":
            if (not "cuda" in device and max([*lora_down.shape, *lora_up.shape]) <= 1024) or max(
                    [*lora_down.shape, *lora_up.shape]) <= 3072:
                new_lora_up, new_lora_down = svd(full_mat, lora_up, lora_down, **kwargs)
            else:
                new_lora_up, new_lora_down = get_least_squares_solution(full_mat, lora_up, lora_down, **kwargs)

        if "conv" in key:
            new_lora_up = new_lora_up[:, :, None, None]
            new_lora_down = new_lora_down.reshape(new_lora_down.shape[0], new_lora_down.shape[1] // int(k * k), k, k)

        new_state_dict[lora_down_key] = new_lora_down
        new_state_dict[lora_up_key] = new_lora_up

    state_dict.update(new_state_dict)

    return state_dict
