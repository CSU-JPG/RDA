import torch

from inference_ar_model.Tar.t2i_inference_rda_prompt import (
    TarRDAInference,
    T2IConfig,
    RDAConfig,
    resolve_file,
    resolve_rda_model,
)

torch.manual_seed(0)

prompt = "A cozy and bright coffee shop signboard with the text “Morning Brew Café — Freshly Roasted Everyday”. Soft beige and light brown colors, sunlight streaming through the window, relaxed vibe."

ar_path = resolve_file(None, "ar_dtok_lp_512px.pth", "csuhan/TA-Tok")
encoder_path = resolve_file(None, "ta_tok.pth", "csuhan/TA-Tok")
vq_ckpt = "pretrained_model/vq_ds16_t2i.pt"
rda_ckpt, rda_config = resolve_rda_model("CSU-JPG/RDA_llamagen")

model = TarRDAInference(
    T2IConfig(
        model_path="csuhan/Tar-7B",
        ar_path=str(ar_path),
        encoder_path=str(encoder_path),
        decoder_path=str(vq_ckpt),
    ),
    RDAConfig(
        checkpoint_path=rda_ckpt
    ),
)

with torch.no_grad():
    ar_codes = model.generate_ar_codes(prompt)

    vq_image, vq_ids, quant_embeddings = model.decode_vq_image(ar_codes)
    rda_residual_image = model.decode_rda_residual(vq_ids, quant_embeddings)
    prediction_image = vq_image + rda_residual_image

outputs = model.make_output(vq_image, rda_residual_image, prediction_image)
output_dir = "outputs/tar_rda_demo"
model.save_output(outputs, output_dir, prompt)