import torch
import math
import torch.nn.functional as F

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


def compute_edge_gradients(latents, edge_strength=1.0):
    """
    エッジ検出用のSobelフィルタを適用して勾配を計算
    """
    # Sobelフィルタ定義
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=latents.dtype, device=latents.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=latents.dtype, device=latents.device).unsqueeze(0).unsqueeze(0)
    
    B, C, T, H, W = latents.shape
    
    # 各チャンネルとフレームに対してエッジ検出を適用
    edge_gradients = torch.zeros_like(latents)
    
    for b in range(B):
        for c in range(C):
            for t in range(T):
                # 現在のフレームを取得
                frame = latents[b, c, t].unsqueeze(0).unsqueeze(0)
                
                # Sobelフィルタを適用
                grad_x = F.conv2d(frame, sobel_x, padding=1)
                grad_y = F.conv2d(frame, sobel_y, padding=1)
                
                # 勾配の大きさを計算
                gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                edge_gradients[b, c, t] = gradient_magnitude.squeeze() * edge_strength
    
    return edge_gradients


def apply_edge_enhancement_guidance(latents, edge_strength=1.0, guidance_scale=1.0):
    """
    エッジ強調ガイダンスを適用
    """
    # エッジ勾配を計算
    edge_gradients = compute_edge_gradients(latents, edge_strength)
    
    # 勾配を正規化
    edge_gradients = edge_gradients / (edge_gradients.abs().max() + 1e-8)
    
    # ガイダンススケールを適用
    enhanced_latents = latents + guidance_scale * edge_gradients
    
    return enhanced_latents, edge_gradients

def sample_hunyuan(
        transformer,
        sampler='unipc',
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        width=512,
        height=512,
        latents=None,
        denoise_strength=1.0,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.0,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        negative_kwargs=None,
        callback=None,
        edge_enhancement_strength=0.0,
        edge_guidance_scale=1.0,
        **kwargs,
):
    device = device or transformer.device

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    # エッジ強調の前処理（no_gradの影響を受けないように先頭で処理）
    print(f"[DEBUG] Edge enhancement check: strength={edge_enhancement_strength}")
    edge_enhanced_latents = None
    if edge_enhancement_strength > 0.0 and latents is not None:
        print(f"[DEBUG] Pre-processing edge enhancement...")
        edge_enhanced_latents, edge_gradients = apply_edge_enhancement_guidance(
            latents, edge_enhancement_strength, edge_guidance_scale
        )
        print(f"[DEBUG] Edge gradients - max: {edge_gradients.max().item():.4f}, min: {edge_gradients.min().item():.4f}")
        print(f"[DEBUG] Latents difference - max: {(edge_enhanced_latents - latents).abs().max().item():.4f}")
        print(f"[DEBUG] Edge enhancement pre-processed: strength={edge_enhancement_strength}, scale={edge_guidance_scale}")

    with torch.no_grad():
        if denoise_strength < 1.0 and latents is not None:
            noise = torch.randn_like(latents)

            # flux_muを使用してノイズレベルを計算
            seq_length = latents.shape[2] * latents.shape[3] * latents.shape[4] // 4
            mu = calculate_flux_mu(seq_length, exp_max=7.0)
            sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)
            
            # denoise_strengthに基づいて適切なノイズスケールを計算
            init_step = min(int(num_inference_steps * denoise_strength), num_inference_steps)
            start_step = max(num_inference_steps - init_step, 0)
            noise_scale = sigmas[start_step]  # 開始ステップのノイズスケールを使用
            
            print("[DEBUG] mu:", mu)
            print("[DEBUG] sigmas:", sigmas)
            print("[DEBUG] start_step:", start_step)
            print("[DEBUG] noise_scale:", noise_scale)
            
            original_latents = latents.clone()
            latents = latents * (1 - noise_scale) + noise_scale * noise
            print("latents shape:", latents.shape)
        else:
            latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)

        # エッジ強調されたlatentsがあれば使用
        if edge_enhanced_latents is not None:
            print(f"[DEBUG] Using edge-enhanced latents")
            latents = edge_enhanced_latents

        B, C, T, H, W = latents.shape
        seq_length = T * H * W // 4

        if shift is None:
            mu = calculate_flux_mu(seq_length, exp_max=7.0)
        else:
            mu = math.log(shift)

        sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)

        k_model = fm_wrapper(transformer)

        if initial_latent is not None:
            sigmas = sigmas * strength
            first_sigma = sigmas[0].to(device=device, dtype=torch.float32)
            initial_latent = initial_latent.to(device=device, dtype=torch.float32)
            latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

        if concat_latent is not None:
            concat_latent = concat_latent.to(latents)

        distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

        prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
        prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
        prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
        negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
        negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
        negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
        concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    if sampler == 'unipc':
        # start_stepから始まるようにsigmasを調整
        if denoise_strength < 1.0:
            sigmas = sigmas[start_step:]
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    return results
