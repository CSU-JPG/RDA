import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from diffusers.models import AutoencoderKL
from transformers import AutoModel, AutoTokenizer

import tok.lumina2_model as models
from tok.transport import Sampler, create_transport
from tok.ta_tok import TextAlignedTokenizer
from tok.utils import ScalingLayer


class LuminaAutoEncoder:
    def __init__(self, cfg, device="cuda:0", dtype=torch.bfloat16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        # Load tokenizer & text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"])
        self.tokenizer.padding_side = "right"
        self.text_encoder = AutoModel.from_pretrained(
            cfg["text_encoder"], torch_dtype=dtype, device_map="cuda"
        ).eval()
        self.cap_feat_dim = self.text_encoder.config.hidden_size

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            cfg["vae_path"], subfolder="vae", torch_dtype=dtype
        ).to(device)
        self.vae.requires_grad_(False)

        # Load text-aligned tokenizer
        self.ta_tok = TextAlignedTokenizer.from_checkpoint(
            cfg["ta_tok_path"], load_teacher=False, input_type='rec'
        )
        self.ta_tok.scale_layer = ScalingLayer(mean=[0., 0., 0.], std=[1.0, 1.0, 1.0])
        self.ta_tok.eval().to(device=device, dtype=dtype)

        # Load main model
        train_args = torch.load(os.path.join(cfg["ckpt"], "model_args.pth"))
        self.model = models.__dict__[train_args.model](
            in_channels=16,
            cond_in_channels=1152,
            qk_norm=train_args.qk_norm,
            cap_feat_dim=self.cap_feat_dim,
        )
        self.model.eval().to(device, dtype=dtype)

        ckpt = torch.load(
            os.path.join(
                cfg["ckpt"],
                f"consolidated{'_ema' if cfg['ema'] else ''}.00-of-{cfg['num_gpus']:02d}.pth",
            )
        )
        self.model.load_state_dict(ckpt, strict=True)

        # Sampler
        self.transport = create_transport("Linear", "velocity", None, None, None)
        self.sampler = Sampler(self.transport)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.cap_feats, self.cap_mask = self.encode_prompt([cfg["prompt"]])

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt_batch,
                padding=True,
                pad_to_multiple_of=8,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_masks = text_inputs.attention_mask

            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids.cuda(),
                attention_mask=prompt_masks.cuda(),
                output_hidden_states=True,
            ).hidden_states[-2]

        return prompt_embeds, prompt_masks

    def preprocess_cond(self, image_path=None, indices=None):
        if image_path is not None:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            cond_in = self.ta_tok(image[None].to(device=self.device, dtype=self.dtype))['encoded']
        elif indices is not None:
            assert indices.ndim == 2
            cond_in = self.ta_tok.decode_from_bottleneck(indices.to(self.device))
        
        B, L, D = cond_in.shape
        H = W = int(L**0.5)
        cond_in = cond_in.permute(0, 2, 1).reshape(B, D, H, W)
        cond_in = F.interpolate(cond_in, size=(128, 128), mode='bilinear')

        cond_ins_all = [[cond_in[0].float()]]

        return cond_ins_all

    def infer(self, input_image=None, indices=None):
        w, h = self.cfg['resolution'], self.cfg['resolution']
        latent_w, latent_h = w // 8, h // 8

        z = torch.randn([1, 16, latent_w, latent_h], device=self.device).to(self.dtype)
        cond = self.preprocess_cond(input_image, indices)

        model_kwargs = dict(
            cap_feats=self.cap_feats,
            cap_mask=self.cap_mask.to(self.cap_feats.device),
            cond=cond,
            position_type=[["aligned"]],
        )

        sample_fn = self.sampler.sample_ode(
            sampling_method=self.cfg["solver"],
            num_steps=self.cfg["num_sampling_steps"],
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=self.cfg["time_shifting_factor"],
        )
        samples = sample_fn(z, self.model.forward, **model_kwargs)[-1]

        samples = samples[:1]
        samples = self.vae.decode(samples / self.vae.config.scaling_factor + self.vae.config.shift_factor)[0]
        samples = (samples + 1.0) / 2.0
        samples.clamp_(0.0, 1.0)

        img = to_pil_image(samples[0].float())
        return img


def main():
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = dict(
        ckpt=snapshot_download("csuhan/Tar-Lumina2"),
        ta_tok_path=hf_hub_download("csuhan/TA-Tok", "ta_tok.pth"),
        text_encoder="google/gemma-2-2b",
        vae_path="black-forest-labs/FLUX.1-dev",
        resolution=1024,
        num_sampling_steps=50,
        solver="euler",
        num_gpus=1,
        ema=True,
        time_shifting_factor=6.0,
        prompt="You are an assistant designed to generate superior images with the highest degree of image alignment based on the SigLIP2 features of an original image. <Prompt Start> ",
    )

    torch.set_grad_enabled(False)
    torch.random.manual_seed(20)

    model = LuminaAutoEncoder(cfg)

    input_image = "asset/dog_cat.jpg"
    save_path = "demos/dog_cat_rec.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.autocast("cuda", torch.bfloat16):
        out_img = model.infer(input_image)
        out_img.save(save_path)


if __name__ == "__main__":
    main()
