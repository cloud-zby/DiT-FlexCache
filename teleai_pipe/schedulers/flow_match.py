import torch


def merge_tensors_with_threshold(A, B, T):
    """
    将A中>=T的元素和B中<T的元素拼接为新Tensor C，并统计来源数量

    参数:
        A (torch.Tensor): 从大到小排序的一维张量
        B (torch.Tensor): 从大到小排序的一维张量
        T (float): 阈值

    返回:
        C (torch.Tensor): 合并后的新张量
        count_from_A (int): 来自A的元素数量
        count_from_B (int): 来自B的元素数量
    """
    A_selected = A[A >= T]
    B_selected = B[B < T]
    C = torch.cat((A_selected, B_selected))
    return C


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        shift=None,
    ):
        self.sigmas, self.timesteps = self.generate_timesteps(
            num_inference_steps, denoising_strength, shift
        )

    def set_stage_timesteps(
        self,
        num_inference_steps_1=40,
        num_inference_steps_2=4,
        boundary=0.9,
        denoising_strength=1.0,
        shift_1=None,
        shift_2=None,
    ):
        sigmas_1, timesteps_1 = self.generate_timesteps(
            num_inference_steps_1, denoising_strength, shift_1
        )
        sigmas_2, timesteps_2 = self.generate_timesteps(
            num_inference_steps_2, denoising_strength, shift_2
        )
        self.sigmas = merge_tensors_with_threshold(sigmas_1, sigmas_2, boundary)
        self.timesteps = merge_tensors_with_threshold(
            timesteps_1, timesteps_2, boundary * self.num_train_timesteps
        )
        print(f"sigmas: {self.sigmas}")
        print(f"timesteps: {self.timesteps}")

    def generate_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        shift=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
        timesteps = sigmas * self.num_train_timesteps
        return sigmas, timesteps

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs()
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
