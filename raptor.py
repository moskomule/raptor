import math

import torch
from torch import Tensor


def _batch_cg_solve(
    b: Tensor,
    g: Tensor,
    c: float,
    steps: int,
    eps: float,
) -> Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_sq = r.pow(2).sum(dim=1, keepdim=True)

    for i in range(steps):
        Ap = p + c * (p @ g)
        pAp = (p * Ap).sum(dim=1, keepdim=True)
        alpha = r_sq / pAp.clamp_min(eps)
        x = x + alpha * p
        if i == steps - 1:
            break
        r = r - alpha * Ap
        r_sq_new = r.pow(2).sum(dim=1, keepdim=True)
        beta = r_sq_new / (r_sq + eps)
        p = r + beta * p
        r_sq = r_sq_new
    return x


def _qdwh_step_cg(
    x: Tensor,
    params: tuple[float, float, float],
    steps: int,
) -> Tensor:

    a, b, c = params
    gram = x.mT @ x

    return _batch_cg_solve(
        a * x + b * (x @ gram),
        g=gram,
        c=c,
        steps=steps,
        eps=1e-6,
    )


def _qdwh_step_cholesky(
    x: Tensor,
    params: tuple[float, float, float],
) -> Tensor:
    _, n = x.shape
    a, b, c = params
    z = torch.addmm(torch.eye(n, device=x.device, dtype=x.dtype), x.mT, x, beta=1, alpha=c)
    w = torch.linalg.cholesky_ex(z, upper=False).L
    y = torch.cholesky_solve(x.mT, w, upper=False).mT
    return (b / c) * x + (a - b / c) * y


def qdwh_coefficients(
    eps: float,
    num_iters: int,
) -> list[tuple[float, float, float]]:
    def h(l):
        d = math.cbrt(4 * (1 - l**2) / (l**4))
        return math.sqrt(1 + d) + 0.5 * math.sqrt(8 - 4 * d + 8 * (2 - l**2) / (l**2 * math.sqrt(1 + d)))

    def _yield_coeffs():
        l = eps
        for i in range(num_iters):
            a = h(l)
            b = (a - 1) ** 2 / 4
            c = a + b - 1
            l = l * (a + b * l**2) / (1 + c * l**2)
            yield (a, b, c)

    return list(_yield_coeffs())


def rational_polar(
    input: Tensor,
    coefficients: list[tuple[float, float, float]],
    eps: float = 1e-3,
    safety_factor: float = 1e-2,
    cg_steps: int = 3,
    backend: str = "cholesky",
) -> Tensor:
    assert input.dim() == 2, "Input must be a 2D tensor."

    ortho_input = input.to(torch.float32)

    if input.size(0) < input.size(1):
        ortho_input = ortho_input.mT

    ortho_input.div_(ortho_input.norm() * (1 + safety_factor) + eps)

    for i, params in enumerate(coefficients):
        if backend == "cholesky":
            ortho_input = _qdwh_step_cholesky(ortho_input, params)
        elif backend == "cg":
            ortho_input = _qdwh_step_cg(ortho_input, params, steps=cg_steps)
        else:
            raise NotImplementedError

    if input.size(0) < input.size(1):
        ortho_input = ortho_input.mT

    return ortho_input.to(input.dtype)


def qdwh(
    input: Tensor,
    num_iters: int = 5,
    eps: float = 1e-3,
    compute_dtype: torch.dtype = torch.float32,
    safety_factor: float = 1e-2,
) -> Tensor:
    coefficients = qdwh_coefficients(eps, num_iters)
    return rational_polar(
        input=input,
        coefficients=coefficients,
        eps=eps,
        safety_factor=safety_factor,
        backend="cg",
    )


def raptor(
    input: Tensor,
    num_iters: int = 5,
    eps: float = 1e-3,
    compute_dtype: torch.dtype = torch.float32,
    safety_factor: float = 1e-2,
    cg_steps: int = 3,
) -> Tensor:
    coefficients = qdwh_coefficients(eps, num_iters)
    return rational_polar(
        input=input,
        coefficients=coefficients,
        eps=eps,
        safety_factor=safety_factor,
        cg_steps=cg_steps,
        backend="cg",
    )
