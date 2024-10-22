import torch
import os
import logging
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image 
from utils.unet import UNetModel
import torch.nn.functional as F
import re
from sklearn.neighbors import NearestNeighbors

def load_model(ckpt_dir, model, device):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if not os.path.exists(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return None
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        return loaded_state['model']

def save_video(save_dir, samples, filename):
    """ Saves a video from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the video"""
    padding = 0
    nrow = int(np.sqrt(samples[0].shape[0]))
    imgs = []
    for idx in range(len(samples)):
        sample = samples[idx].cpu().detach().numpy()
        sample = np.clip(sample * 255, 0, 255)
        image_grid = make_grid(torch.Tensor(sample), nrow, padding=padding).numpy(
        ).transpose(1, 2, 0).astype(np.uint8)
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        imgs.append(image_grid)
    #video_size = tuple(reversed(tuple(5*s for s in imgs[0].shape[:2])))
    video_size = tuple(reversed(tuple(s for s in imgs[0].shape[:2])))
    writer = cv2.VideoWriter(os.path.join(save_dir, filename), cv2.VideoWriter_fourcc(*'mp4v'),
                             30, video_size)
    for i in range(len(imgs)):
        image = cv2.resize(imgs[i], video_size, fx=0,
                           fy=0, interpolation=cv2.INTER_CUBIC)
        writer.write(image)
    writer.release()


def save_gif(save_dir, samples, name="process.gif"):
    """ Saves a gif from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the gif"""
    nrow = int(np.sqrt(samples[0].shape[0]))
    imgs = []
    for idx in range(len(samples)):
        s = samples[idx].cpu().detach().numpy()[:36]
        s = np.clip(s * 255, 0, 255).astype(np.uint8)
        image_grid = make_grid(torch.Tensor(s), nrow, padding=2)
        im = Image.fromarray(image_grid.permute(
            1, 2, 0).to('cpu', torch.uint8).numpy())
        imgs.append(im)
    imgs[0].save(os.path.join(save_dir, name), save_all=True,
                 append_images=imgs[1:], duration=0.5, loop=0)


def save_tensor(save_dir, data, name):
    """ Saves a Pytorch Tensor to save_dir with the given name."""
    with open(os.path.join(save_dir, name), "wb") as fout:
        np.save(fout, data.cpu().numpy())


def save_number(save_dir, data, name):
    """ Saves the number in argument 'data' as a text file and a .np file."""
    with open(os.path.join(save_dir, name), "w") as fout:
        fout.write(str(data))
    with open(os.path.join(save_dir, name) + ".np", "wb") as fout:
        np.save(fout, data)


def save_tensor_list(save_dir, data_list, name):
    """Saves a list of Pytorch tensors to save_dir with name 'name'"""
    with open(os.path.join(save_dir, name), "wb") as fout:
        np.save(fout, np.array([d.cpu().detach().numpy() for d in data_list]))

def save_png_separate_imgs(save_dir, data, batch_nr, batch_size):
    for i in range(data.shape[0]):
        name = f"{str(batch_nr * batch_size + i).zfill(8)}.png"
        with open(os.path.join(save_dir, name), "wb") as fout:
            save_image(data[i, :, :, :], fout)

def save_png(save_dir, data, name, nrow=None):
    """Save tensor 'data' as a PNG"""
    if nrow == None:
        nrow = int(np.sqrt(data.shape[0]))
    image_grid = make_grid(data, nrow, padding=2)
    with open(os.path.join(save_dir, name), "wb") as fout:
        save_image(image_grid, fout)

def torch_batch_to_np_imgs(torch_tensor):
    np_batch_array = torch_tensor.numpy().astype(np.float32)
    np_batch_array = np_batch_array.transpose(0, 2, 3, 1)
    return np_batch_array

def numpy_imgs_to_torch_batch(np_imgs):
    result = []
    for img in np_imgs:
        torch_img = img.transpose(2, 0, 1)
        result.append(torch_img)
    return torch.from_numpy(np.array(result))

def fft_of_torch_batch(torch_tensor):

    # Convert to grayscale
    torch_tensor = torch_tensor[:, :3, :, :].mean(dim=1).unsqueeze(1)

    ft = np.fft.ifftshift(torch_tensor.cpu())
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    ft_torch_batch = torch.from_numpy(ft)

    return torch.log(torch.abs(ft_torch_batch))/8

def create_model(config, n_in_channels=3, n_out_channels=3, device_ids=None):
    """Create the model."""
    model = UNetModel(config, n_in_channels, n_out_channels)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model

def model_fn_wrapper(model, train=True):
    def eval_model(x_t, t):
        if train:
            model.train()
            return model(x_t, t)
        else:
            model.eval()
            return model(x_t, t)    
    return eval_model

def extract_file_nr(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def find_latest_checkpoint_file(workdir_path):
    files = []
    for file in os.listdir(workdir_path):
        if file.endswith(".pth"):
            files.append(file)
        elif file.endswith(".ckpt"):
            files.append(file)

    last_checkpoint = max(files, key=extract_file_nr)
    return last_checkpoint

def find_latest_sample_file(samples_path):
    file_nrs = []
    for file in os.listdir(samples_path):
        print(file)
        if file.endswith(".png"):
            file_nr = re.findall(r'\d+', file)[0]
            file_nrs.append(file_nr)
    if len(file_nrs) != 0:
        return int(max(file_nrs))
    else:
        return 0

def ts_to_time_indices(ts, dt):
    return torch.round(torch.div(ts, dt)).int()

def compute_laplacian(x):
    """
    Computes discrete Laplacian of an image.
    """
    laplacian_kernel = torch.Tensor([
                            [   [0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]
                            ], 
                            [   [0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]
                            ],
                            
                            [   [0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]
                            ],
                            ]).to(x.device)
    laplacian_kernel = laplacian_kernel.view((1,x.shape[1],3,3))

    return torch.abs(F.conv2d(x.double(), laplacian_kernel.double(), padding='same'))

def compute_gradient(x):
    """
    Computes gradient magnitude image.
    """
    x_grad_kernel = torch.Tensor([
                            [   [1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]
                            ] for i in range(x.shape[1])]).to(x.device)
    x_grad_kernel = x_grad_kernel.view((1,x.shape[1],3,3))

    y_grad_kernel = torch.Tensor([
                            [   [1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]
                            ] for i in range(x.shape[1])]).to(x.device)
    y_grad_kernel = y_grad_kernel.view((1,x.shape[1],3,3))

    x_grad = F.conv2d(x.double(), x_grad_kernel.double(), padding='same')
    y_grad = F.conv2d(x.double(), y_grad_kernel.double(), padding='same')

    gradient_sqrd_norm = (x_grad**2 + y_grad **2)
    return gradient_sqrd_norm / torch.max(gradient_sqrd_norm)

def fft_of_torch_batch(torch_tensor):
    # Convert to grayscale
    torch_tensor = torch_tensor[:, :3, :, :].mean(dim=1).unsqueeze(1)

    ft = np.fft.ifftshift(torch_tensor.cpu())
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    ft_torch_batch = torch.from_numpy(ft)

    return torch.log(torch.abs(ft_torch_batch))

def gaussian_filter(kernel_size, sigma=1, muu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1.0/(2.0 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))

    return gauss

def diff_of_gaussians(sigma_1=1.0, sigma_2=1.6):
    gaussian_kernel1 = gaussian_filter(128, sigma=sigma_1)
    gaussian_kernel2 = gaussian_filter(128, sigma=sigma_2)
    return np.clip(gaussian_kernel2 - gaussian_kernel1, 0.0, 1.0)

def filter_img(spatial_img, fourier_kernel):
    ft = np.fft.ifftshift(spatial_img)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    filtered = fourier_kernel * ft 
    filtered = np.fft.ifftshift(filtered)
    filtered = np.fft.ifft2(filtered)
    filtered = np.fft.fftshift(filtered)
    return filtered.real

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def compute_nearest_neighbours(query_img_batch, dataset_imgs, n_neighbors=5, save_path=None):

    # Flatten the images (convert each image to a 1D vector)
    # and create a NearestNeighbors instance
    image_vectors = [img.flatten() for img in dataset_imgs]
    image_vectors = np.array(image_vectors)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(image_vectors)

    neighbors_batch = torch.empty(( query_img_batch.shape[0], n_neighbors, query_img_batch.shape[1], query_img_batch.shape[2], query_img_batch.shape[3]))
    for k, query_img in enumerate(torch_batch_to_np_imgs(query_img_batch.cpu())):
        # Make sure to scale back to RGB256 (We process tensors in range [0-1])
        query_img_vector = np.array([query_img.flatten()]) * 255.0

        # Choose an image to find its nearest neighbors
        distances, indices = nbrs.kneighbors(query_img_vector)
        neighbors = []
        for i, index in enumerate(indices[0]):
            if save_path is not None:
                neighbor_batch = numpy_imgs_to_torch_batch(np.array([image_vectors[index].reshape((128,128,3))])/ 255.0).float()
                save_png(save_path, neighbor_batch, f"neighbor_img_{k}_{i}.png")

            neighbors.append(image_vectors[index].reshape((128,128,3)))
        
        neighbors_batch[k] = (numpy_imgs_to_torch_batch(np.array(neighbors)))

    # neighbors_batch = np.transpose(np.array(neighbors_batch), (1, 0, 2, 3, 4))
    color_channel_order = [2, 1, 0]
    return torch.permute(neighbors_batch, (1,0,2,3,4))[:, :, color_channel_order, :, :]
        