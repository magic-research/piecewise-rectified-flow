import os
import json
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageStat
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torchvision import transforms as T


### >>>>>>>> >>>>>>>> text related >>>>>>>> >>>>>>>> ###

class TokenizerWrapper():
    def __init__(self, tokenizer, is_train, proportion_empty_prompts, use_generic_prompts=False):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.proportion_empty_prompts = proportion_empty_prompts
        self.use_generic_prompts = use_generic_prompts

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        captions = []
        for caption in prompts:
            if random.random() < self.proportion_empty_prompts:
                captions.append("")
            else:
                if self.use_generic_prompts:
                    captions.append("best quality, high quality")
                elif isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if self.is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column should contain either strings or lists of strings."
                    )
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return inputs.input_ids



### >>>>>>>> >>>>>>>> image related >>>>>>>> >>>>>>>> ###

MONOCHROMATIC_MAX_VARIANCE = 0.3

def is_monochromatic_image(pil_img):
    v = ImageStat.Stat(pil_img.convert('RGB')).var
    return sum(v)<MONOCHROMATIC_MAX_VARIANCE

def isnumeric(text):
    return (''.join(filter(str.isalnum, text))).isnumeric()



class TextPromptDataset(IterableDataset):
    '''
      The dataset for (text embedding, noise, generated latent) triplets.
    '''
    def __init__(self, 
                data_root, 
                tokenizer = None,
                transform = None,
                rank = 0,
                world_size = 1,
                shuffle = True,
    ):
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.img_root = os.path.join(data_root, 'JPEGImages')
        self.data_list = []
        
        print("#### Loading filename list...")
        json_root = os.path.join(data_root, 'list')
        json_list = [p for p in os.listdir(json_root) if p.startswith("shard") and p.endswith('.json')]
        
        # duplicate several shards to make sure each process has the same number of shards
        assert len(json_list) > world_size
        duplicate = world_size - len(json_list)%world_size if len(json_list)%world_size>0 else 0
        json_list = json_list + json_list[:duplicate]
        json_list = json_list[rank::world_size]
        
        for json_file in tqdm(json_list):
            shard_name = os.path.basename(json_file).split('.')[0]
            with open(os.path.join(json_root, json_file)) as f:
                key_text_pairs = json.load(f)
                
            for pair in key_text_pairs:
                self.data_list.append( [shard_name] + pair )

        print("#### All filename loaded...")
        
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.data_list)
    
    
    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:  # single-process data loading, return the full iterator
            data_list = self.data_list
        else:
            len_data = len(self.data_list) - len(self.data_list) % worker_info.num_workers
            data_list = self.data_list[:len_data][worker_info.id :: worker_info.num_workers]
            # print(worker_info.num_workers, worker_info.id, len(data_list)/len(self.data_list))
            
        if self.shuffle:
            random.shuffle(data_list) 
            
        while True:    
            for idx in range(len(data_list)):
                # try:
                shard_name = data_list[idx][0]
                data = {}
                
                img_file = data_list[idx][1]
                img = Image.open(os.path.join(self.img_root, shard_name, img_file+'.jpg')).convert("RGB")
                
                if is_monochromatic_image(img):
                    continue
                
                if self.transform is not None:
                    img = self.transform(img)
                    
                data['pixel_values'] = img
                
                text = data_list[idx][2]
                if self.tokenizer is not None:
                    if isinstance(self.tokenizer, list):
                        assert len(self.tokenizer)==2
                        data['input_ids'] = self.tokenizer[0](text)[0]
                        data['input_ids_2'] = self.tokenizer[1](text)[0]
                    else:
                        data['input_ids'] = self.tokenizer(text)[0]
                else:
                    data['input_ids'] = text
                
                yield data
                
                # except Exception as e:
                #     raise(e)

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        if self.tokenizer is not None:
            if isinstance(self.tokenizer, list):
                assert len(self.tokenizer)==2
                input_ids = torch.stack([example["input_ids"] for example in examples])
                input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
                return {"pixel_values": pixel_values, "input_ids": input_ids, "input_ids_2": input_ids_2,}
            else:
                input_ids = torch.stack([example["input_ids"] for example in examples])
                return {"pixel_values": pixel_values, "input_ids": input_ids,}
        else:
            input_ids = [example["input_ids"] for example in examples]
            return {"pixel_values": pixel_values, "input_ids": input_ids,}
    
    
def make_train_dataset(
        train_data_path, 
        size = 512,
        tokenizer=None, 
        cfg_drop_ratio=0,
        rank=0, 
        world_size=1,
        shuffle=True,
    ):
    
    _image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(size),
            T.CenterCrop((size,size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if tokenizer is not None:
        if isinstance(tokenizer, list):
            assert len(tokenizer)==2
            tokenizer_1 = TokenizerWrapper(
                tokenizer[0], 
                is_train=True, 
                proportion_empty_prompts=cfg_drop_ratio,
                use_generic_prompts=False,
            )
            tokenizer_2 = TokenizerWrapper(
                tokenizer[1], 
                is_train=True, 
                proportion_empty_prompts=cfg_drop_ratio,
                use_generic_prompts=False,
            )
            tokenizer = [tokenizer_1, tokenizer_2]
            
        else:
            tokenizer = TokenizerWrapper(
                tokenizer, 
                is_train=True, 
                proportion_empty_prompts=cfg_drop_ratio,
                use_generic_prompts=False,
            )

        
    train_dataset = TextPromptDataset(
        data_root=train_data_path,
        transform=_image_transform,
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        shuffle=shuffle,
    )
    return train_dataset
    
    
    
    
    
    
    
    
    

### >>>>>>>> >>>>>>>> Test >>>>>>>> >>>>>>>> ###
if __name__ == "__main__":
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "/mnt/bn/ic-research-aigc-editing/fast-diffusion-models/assets/public_models/StableDiffusion/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )
    train_dataset = make_train_dataset(tokenizer=tokenizer, rank=0, world_size=10)
    
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, num_workers=0, 
        collate_fn=train_dataset.collect_fn if hasattr(train_dataset, 'collect_fn') else None,
    )
    for batch in loader:
        pixel_values = batch["pixel_values"]
        prompt_ids = batch['input_ids']
        from einops import rearrange
        pixel_values = rearrange(pixel_values, 'b c h w -> b h w c')
        
        for i in range(pixel_values.shape[0]):
            import pdb; pdb.set_trace()
            Image.fromarray(((pixel_values[i] + 1 )/2 * 255 ).numpy().astype(np.uint8)).save('tmp.png')
            input_id = prompt_ids[i]
            text = tokenizer.decode(input_id).split('<|startoftext|>')[-1].split('<|endoftext|>')[0]
            print(text)
        pass