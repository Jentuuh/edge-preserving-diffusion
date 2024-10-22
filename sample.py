import torch
import random
import numpy as np
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from pathlib import Path
import os

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from utils.vanilla_ddpm_model import ddpm_sampler
from utils.edge_preserving_model import edge_preserving_sampler, exponential_transition_scheme, linear_transition_scheme, cosine_transition_scheme
from utils.tools import save_png_separate_imgs, create_model, model_fn_wrapper, find_latest_sample_file

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("checkpointdir", None, "Checkpoint path.")
flags.DEFINE_string("savedir", None, "Save directory.")
flags.DEFINE_string("method", None, "Should be 'ours' or 'ddpm'.")
flags.DEFINE_integer("n_samples", None, "Amount of samples to generate.")
flags.DEFINE_integer("batch_size", None, "Size of batch used for each run of the backward process.")

flags.mark_flags_as_required(["checkpointdir", "savedir", "config", "n_samples", "batch_size", "method"])


def main(argv):
    assert FLAGS.method == "ours" or FLAGS.method =="ddpm" or FLAGS.method == "ours_linear", "--method flag should be 'ours', 'ours_linear' or 'ddpm'."
    sample(FLAGS.config, FLAGS.method, FLAGS.checkpointdir, FLAGS.savedir, FLAGS.n_samples, FLAGS.batch_size)

def sample(config, METHOD, CHECKPOINT_DIR, SAVE_DIR, NUM_SAMPLES, BATCH_SIZE):
    device = config.device

    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    latest_sample_nr = find_latest_sample_file(save_path)
    start_iteration = (latest_sample_nr + 1) // BATCH_SIZE
    print(f"Latest sample nr found is {latest_sample_nr}. Starting from iteration {start_iteration}.")

    if latest_sample_nr >= NUM_SAMPLES:
        return
    
    # Initialize model
    model = create_model(config, config.data.num_channels, config.data.num_channels)
    model_fn_eval = model_fn_wrapper(model, train=False)

    if METHOD == "ours":
        sampling_module = edge_preserving_sampler(config, model_fn_eval, linear_transition_scheme, device)
    elif METHOD == "ddpm":
        sampling_module = ddpm_sampler(config, model_fn_eval, device)

    # Load latest checkpoint (if exists and desired)
    if os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint {CHECKPOINT_DIR} found. Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_DIR)
        model.load_state_dict(checkpoint["model"])
    else: 
        print(f"Checkpoint at {CHECKPOINT_DIR} not found. Exiting.")
        os._exit(os.EX_DATAERR)

    # ==============
    # Sampling loop
    # ==============
    ITERATIONS = (NUM_SAMPLES // BATCH_SIZE) + 1
    for i in range(start_iteration, ITERATIONS):

        x_T = torch.randn(BATCH_SIZE, config.data.num_channels, config.data.image_size, config.data.image_size).to(device)
        with torch.no_grad():
            print(f'Generating sample batch {i}...')
            sample, full_path, full_path_noise = sampling_module(x_T)
            sample = (sample + 1.0) * 0.5
            save_png_separate_imgs(save_path, sample.cpu(), i, BATCH_SIZE)

if __name__ == "__main__":
    app.run(main)
