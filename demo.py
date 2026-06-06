import torch
from PIL import Image

from tokenizer.tokenizer_image.rda_model import RDATokenizer


model = RDATokenizer.from_pretrained(
    "CSU-JPG/RDA_llamagen",
    vq_ckpt="pretrained_model/vq_ds16_t2i.pt",
).to("cuda")

vq = model.vq_model
rda = model.resvq_model

image = Image.open("examples/test.png").convert("RGB")
inputs = model.transform(image).unsqueeze(0).to("cuda")

with torch.no_grad():
    vq_image, _, vq_info, quant_embeddings = vq(inputs, return_quant=True)
    vq_latent = vq.post_quant_conv(quant_embeddings)
    vq_ids = vq_info[2].reshape(vq_image.shape[0], -1)

    residual_image = inputs - vq_image
    rda_residual_image, _, _ = rda(residual_image, vq_ids, vq_latent)
    prediction_image = vq_image + rda_residual_image

outputs = model.make_output(inputs, residual_image, vq_image, rda_residual_image, prediction_image)
model.save_output(outputs, "outputs/demo")
