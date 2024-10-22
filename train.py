import torch
import torchvision
from torchvision import transforms
from torch.optim import Adam
from pathlib import Path
import os

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from utils.tools import save_png, save_video, create_model, find_latest_checkpoint_file, model_fn_wrapper
from utils.edge_preserving_model import edge_preserving_diffusion_trainer, edge_preserving_sampler, exponential_transition_scheme, linear_transition_scheme, cosine_transition_scheme
from utils.vanilla_ddpm_model import ddpm_diffusion_trainer, ddpm_sampler

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("datadir", None, "Dataset directory.")
flags.DEFINE_string("method", None, "Should be 'ours' or 'ddpm'.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
    assert FLAGS.method == "ours" or FLAGS.method =="ddpm", "--method flag should be 'ours' or 'ddpm'."
    train(FLAGS.config, FLAGS.method, FLAGS.datadir, FLAGS.workdir)

def train(config, METHOD, DATA_PATH, WORK_DIR):
    work_path = Path(WORK_DIR)
    work_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = ""
    if os.path.exists(work_path.joinpath("checkpoints/")):
        checkpoint_file = find_latest_checkpoint_file(work_path.joinpath("checkpoints/"))

    device = config.device

    if config.data.num_channels == 1:
        transform = transforms.Compose([transforms.Resize((config.data.image_size,config.data.image_size)), transforms.RandomHorizontalFlip(0.5), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((config.data.image_size,config.data.image_size)), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Initialize model
    model = create_model(config, config.data.num_channels, config.data.num_channels)
    model_fn_train = model_fn_wrapper(model)
    model_fn_eval = model_fn_wrapper(model, train=False)

    if METHOD == "ours":
        training_module = edge_preserving_diffusion_trainer(config, model_fn_train, cosine_transition_scheme, device)
        sampling_module = edge_preserving_sampler(config, model_fn_eval, cosine_transition_scheme, device)
    elif METHOD == "ddpm":
        training_module = ddpm_diffusion_trainer(config, model_fn_train, device)
        sampling_module = ddpm_sampler(config, model_fn_eval, device)

    # Prior x_T, this is passed to backwards sampler to generate a sample from target distribution x_0
    x_T = training_module.sample_prior_x_T(train_dataset, 16, device, True, work_path)

    # Training loop
    optimizer = Adam(model.parameters(), lr=config.optim.lr)
    nb_iter = 0
    last_epoch = 0
    loss = None

    # Load latest checkpoint (if exists and desired)    
    if os.path.exists(work_path.joinpath(f"checkpoints/{checkpoint_file}")):
        if config.resume:
            print(f"Checkpoint ({checkpoint_file}) found. Loading checkpoint")
            checkpoint = torch.load(work_path.joinpath(f"checkpoints/{checkpoint_file}"))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            nb_iter = checkpoint['iteration_n']
            last_epoch = checkpoint['epoch_n']
    else:
        print("No checkpoint found, starting new training.")

    print('Starting training...')
    for current_epoch in range(last_epoch, config.training.num_epochs):
        for i, data in enumerate(dataloader):
            print(f"Training iteration {nb_iter}, epoch {current_epoch} / {config.training.num_epochs}... Loss: {loss}")

            x0 = (data[0].to(device)*2)-1
            loss = training_module(x0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_iter += 1

            if nb_iter % config.training.sample_freq == 0:
                with torch.no_grad():
                    print(f'Saving sample at {nb_iter} iterations...')
                    sample, full_path, full_path_noise = sampling_module(x_T)
                    sample = (sample + 1.0) * 0.5
                    full_path = (full_path + 1.0) * 0.5
                    
                    work_path.joinpath(f'samples/').mkdir(parents=True, exist_ok=True)
                    save_png(work_path.joinpath(f'samples/'), sample.cpu(), f"sample_{str(nb_iter).zfill(8)}.png")
                    save_video(work_path.joinpath(f'samples/'), full_path.cpu(), "sampling_process.mp4")
                    save_video(work_path.joinpath(f'samples/'), full_path_noise.cpu(), "sampling_process_noise_masks.mp4")
                    
            if nb_iter % config.training.checkpoint_freq == 0:   
                    print(f'Saving intermediate checkpoint at {nb_iter} iterations...')
                    work_path.joinpath(f'intermediate_checkpoints/').mkdir(parents=True, exist_ok=True)
                    state = {}
                    state["model"] = model.state_dict()
                    state["optimizer"] = optimizer.state_dict()
                    state['epoch_n'] = current_epoch
                    state['iteration_n'] = nb_iter
                    torch.save(state, work_path.joinpath(f'intermediate_checkpoints/{config.data.dataset}_{nb_iter}.pth'))

if __name__ == "__main__":
    app.run(main)
