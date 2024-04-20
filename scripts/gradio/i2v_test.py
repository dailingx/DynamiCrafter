import os
import time
from omegaconf import OmegaConf
import torch
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling, get_latent_z
from utils.utils import instantiate_from_config
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything


class Image2Video():
    def __init__(self,result_dir='./tmp/',gpu_num=1,resolution='256_256') -> None:
        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1])) #hw
        self.download_model()
        
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        ckpt_path='checkpoints/dynamicrafter_'+resolution.split('_')[1]+'_v1/model.ckpt'
        config_file='configs/inference_'+resolution.split('_')[1]+'_v1.0.yaml'
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            # model = model.cuda(gpu_id)
            assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8

    def get_image(self, image, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123):
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
            ])
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        start = time.time()
        gpu_id=0
        if steps > 60:
            steps = 60 
        model = self.model_list[gpu_id]
        model = model.cuda()
        batch_size=1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_emb = model.get_learned_conditioning([prompt])

            # img cond
            # 将一个NumPy数组image转换为PyTorch张量。改变张量的维度顺序，从(高, 宽, 通道数)变为(通道数, 高, 宽)，这是因为PyTorch处理图像数据时通常期望通道在前。
            # 将张量的数据类型转换为浮点数。将张量移动到模型所在的设备上（例如CPU或GPU）。
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
            # 将图像张量的像素值从[0, 255]归一化到[-1, 1]区间，这是一种常见的图像预处理步骤。
            img_tensor = (img_tensor / 255. - 0.5) * 2

            # 应用一个预定义的transform函数或对象到图像张量上，可能进行了缩放、裁剪等操作，以适配模型的输入要求。
            image_tensor_resized = transform(img_tensor) #3,h,w
            # 张量的第0维（批次维度）增加一个维度，将张量的形状从(通道数, 高, 宽)变为(1, 通道数, 高, 宽)，这样才能作为模型的一个批次输入。
            videos = image_tensor_resized.unsqueeze(0) # bchw

            # 在张量的第2维（时间维度）增加一个维度，然后调用get_latent_z函数或方法，该函数可能是为了获取视频或图像的潜在表示
            z = get_latent_z(model, videos.unsqueeze(2)) #bc,1,hw

            # 使用repeat函数复制z张量中的时间维度t，以匹配所需的帧数frames。这可能是为了调整潜在表示的时间维度，以适应后续处理。
            img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

            # 再次在图像张量的批次维度上增加一个维度，并通过模型的embedder来获取其嵌入表示。
            cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
            # 将图像嵌入cond_images通过另一个模型或模型的一部分image_proj_model进行处理，可能是为了进一步投影或编码图像信息。
            img_emb = model.image_proj_model(cond_images)

            # 将文本嵌入text_emb和图像嵌入img_emb沿着第1维（特征维度）拼接起来，构成一个联合嵌入表示。
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            # 创建一个包含某些值（fs）的张量，并指定类型为长整型，同时将该张量移动到模型所在的设备上。
            fs = torch.tensor([fs], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
            
            ## inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
            ## b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str=prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'

        save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")
        model = model.cpu()
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")
    
    def download_model(self):
        REPO_ID = 'Doubiiu/DynamiCrafter_'+str(self.resolution[1]) if self.resolution[1]!=256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        if not os.path.exists('./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/'):
            os.makedirs('./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/', filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/', local_dir_use_symlinks=False)
    
if __name__ == '__main__':
    i2v = Image2Video()
    video_path = i2v.get_image('prompts/art.png','man fishing in a boat at sunset')
    print('done', video_path)