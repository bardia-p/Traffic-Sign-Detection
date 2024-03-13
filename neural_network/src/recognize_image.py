import albumentations
import numpy as np
import torch

from neural_network.src.model import Net
import os.path


class Recog:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    # load the model checkpoint
    my_path = os.path.abspath(os.path.dirname(__file__))

    checkpoint = torch.load(os.path.join(my_path, '../outputs/model.pth'))
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    aug = albumentations.Compose([
        # 48x48 resizing is required
        albumentations.Resize(48, 48, always_apply=True),

    ])

    def recog_image(self, image):
        """
        Returns a tuple containing the top three most likely signs
        present from the detected sign
        :param image: numpy coloured image
        :return:      list of tuples with order representing likelyhood,
                        first index of tuple representing match strength,
                        and second index of typle representing sign index
        """
        self.model.eval()
        with torch.no_grad():
            image = image / 255.
            image = self.aug(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float).to(self.device)
            image = image.unsqueeze(0)
            outputs = self.model(image)
            values, indexes = torch.topk(outputs.data, 3)

        return list(zip(values.flatten().tolist(), indexes.flatten().tolist()))
