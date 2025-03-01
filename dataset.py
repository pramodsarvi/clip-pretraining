

from torch.utils.data import Dataset


class MyntraDataset(Dataset):
    def __init__(self, data_frame, captions, target_size=28):
         
        self.data_frame = data_frame[data_frame['subCategory'].str.lower() != 'innerwear']
        self.target_size = target_size  # Desired size for the square image
        self.transform = T.Compose([
            T.ToTensor()  # Convert image to tensor
        ])
 
        self.captions = captions
 
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        while True:
            sample = self.data_frame.iloc[idx]
            img_path = os.path.join(
                "fashion/myntradataset/images", f"{sample['id']}.jpg"
            )
 
            try:
                # Attempt to open the image
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, IOError):
                # If the image is not found, skip this sample by incrementing the index
                idx = (idx + 1) % len(
                    self.data_frame
                )  # Loop back to the start if we reach the end
                continue  # Retry with the next index
 
            # Resize the image to maintain aspect ratio
            image = self.resize_and_pad(image, self.target_size)
 
            # Apply transformations (convert to tensor)
            image = self.transform(image)
 
            # Retrieve the subCategory label and its corresponding caption
            label = sample["subCategory"].lower()
            label = {"lips": "lipstick", "eyes": "eyelash", "nails": "nail polish"}.get(
                label, label
            )
 
            label_idx = next(
                idx for idx, class_name in self.captions.items() if class_name == label
            )
            # # print(label_idx)
            # # print(self.captions[label_idx])
 
            # # Tokenize the caption using the tokenizer function
            cap, mask = tokenizer(self.captions[label_idx])
 
            # Make sure the mask is a tensor
            mask = torch.tensor(mask)
 
            # If the mask is a single dimension, make sure it is expanded correctly
            if len(mask.size()) == 1:
                mask = mask.unsqueeze(0)
 
            return {"image": image, "caption": cap, "mask": mask, "id": img_path}
        

    def resize_and_pad(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        image = image.resize((new_width, new_height))

        pad_width = (target_size - new_width) // 2
        pad_height = (target_size - new_height) // 2

        padding = (
            pad_width,
            pad_height,
            target_size - new_width - pad_width,
            target_size - new_height - pad_height,
        )
        image = ImageOps.expand(image, padding, fill=(0, 0, 0))

        return image
    
