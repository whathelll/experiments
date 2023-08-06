# this file will explore the building of an U-Net encoder / decoder for the MNIST dataset
# the goal is to build a model that can take a 28x28 image and output a 28x28 image
# the model will also take a text input with a text encoder that will be trained contrastively with the image encoder similar to CLIP
# the model will be trained with a contrastive loss function
# the decoder will be trained with a reconstruction loss function

# import libraries

# %%
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %%
torch.cuda.is_available()

# %%
# load the MNIST dataset
train_data = datasets.MNIST(
    root="/workspaces/fast.ai/data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
test_data = datasets.MNIST(
    root="/workspaces/fast.ai/data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

# %%
# view sample data

image = train_data[0][0]
label = train_data[0][1]
print(label)
# shuffle image dimensions to be (28, 28, 1)
image = image.permute(1, 2, 0)
plt.imshow(image)

# print(image.min())

# %%
# define the loss function


def predictive_loss(x, y):
    return F.cross_entropy(x, y)


# %%
# import libraries
train_dataloader = DataLoader(train_data, batch_size=256, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, pin_memory=True, shuffle=True)

# %%
# define the model


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = nn.Parameter(torch.Tensor([0.5]))
        self.embed_length = 64
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Flatten(),
            nn.Linear(10 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, self.embed_length),
            nn.ReLU(),
        )
        self._predict_image_layer = nn.Linear(self.embed_length, 10)
        self.text_embedding = nn.Embedding(num_embeddings=10, embedding_dim=10)
        self.text_encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=128),
            nn.Linear(128, self.embed_length),
            nn.ReLU(),
        )
        self.image_decoder = nn.Sequential(
            nn.Linear(self.embed_length, 512),
            nn.ReLU(),
            nn.Linear(512, 10 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (10, 14, 14)),
            nn.ConvTranspose2d(10, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def encode_image(self, x):
        return self.image_encoder(x)

    def predict_image(self, x):
        x = self.encode_image(x)
        x = self._predict_image_layer(x)
        return x

    def generate_image(self, y):
        vector = self.encode_text(y)
        image = self.image_decoder(vector)
        return image

    def reconstruct_image(self, x):
        vector = self.encode_image(x)
        image = self.image_decoder(vector)
        return image

    def encode_text(self, y):
        # Apply the embedding layer to the input tensor
        embedded_tensor = self.text_embedding(y)
        result = self.text_encoder(embedded_tensor)
        return result


model = Model().cuda()

# %%
# test generation
print(train_data[0][0].shape)
with torch.no_grad():
    label = torch.Tensor([6]).int().cuda()
    image = model.generate_image(label)[0]
    print(image.shape)
    plt.imshow(image[0].cpu())


# %%


def CLIP_Loss(x, y, temperature):
    # image_encoder - ResNet or Vision Transformer
    # text_encoder - CBOW or Text Transformer
    # I[n, h, w, c] - minibatch of aligned images
    # T[n, l] - minibatch of aligned texts
    # W_i[d_i, d_e] - learned proj of image to embed
    # W_t[d_t, d_e] - learned proj of text to embed
    # t - learned temperature parameter
    # extract feature representations of each modality
    # I_f = image_encoder(I) #[n, d_i]
    # T_f = text_encoder(T) #[n, d_t]
    # joint multimodal embedding [n, d_e]
    # I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
    # T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
    # scaled pairwise cosine similarities [n, n]
    # logits = np.dot(I_e, T_e.T) * np.exp(t)
    # symmetric loss function
    # labels = np.arange(n)
    # loss_i = cross_entropy_loss(logits, labels, axis=0)
    # loss_t = cross_entropy_loss(logits, labels, axis=1)
    # loss = (loss_i + loss_t)/2

    # normalize the embeddings
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    # compute the logits
    logits = x @ y.T / torch.exp(temperature)
    # compute the labels
    labels = torch.arange(logits.shape[0]).cuda()
    # compute the loss
    loss_x = F.cross_entropy(logits, labels)
    loss_y = F.cross_entropy(logits.T, labels)
    return (loss_x + loss_y) / 2


# %%
# create a class to do the training from above
class Trainer:
    def __init__(self, model: Model) -> None:
        self.model = model

    def _train_loop(self, dataloader, train_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()
            loss = train_fn(X, y, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test_loop(self, dataloader, train_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.cuda()
                y = y.cuda()
                loss = train_fn(X, y, model)
                test_loss += loss.item()

        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    def require_grad(self, module, requires_grad=True):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def train(self, train_fn, optimizer, epochs=1):
        for t in range(epochs):
            print(f"Epoch {t+1} -------------------------------")
            self._train_loop(train_dataloader, train_fn, optimizer)
            self._test_loop(test_dataloader, train_fn)
        print("Done!")

    def train_clip(self, train_dataloader, test_dataloader, epochs=4):
        self.require_grad(self.model.image_encoder, requires_grad=True)
        self.require_grad(self.model.text_encoder, requires_grad=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        def train_fn(X, y, model):
            x_v = model.encode_image(X)
            y_v = model.encode_text(y)
            loss = CLIP_Loss(x_v, y_v, model.temperature)
            return loss

        self.train(train_fn, optimizer, epochs=epochs)

    def train_predict_image(self, epochs=1):
        self.require_grad(self.model.image_encoder, requires_grad=False)

        def train_fn(X, y, model):
            pred = model.predict_image(X)
            loss = F.cross_entropy(pred, y)
            return loss

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.train(train_fn, optimizer, epochs=epochs)

    def train_reconstruct_image(self, epochs=1):
        self.require_grad(self.model.image_encoder, requires_grad=False)
        self.require_grad(self.model.text_encoder, requires_grad=False)

        def train_fn(X, y, model: Model):
            pred = model.reconstruct_image(X)
            loss = F.l1_loss(pred, X)
            return loss

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.train(train_fn, optimizer, epochs=epochs)


# %%
# train CLIP
model = Model().cuda()
trainer = Trainer(model)
trainer.train_clip(train_dataloader, test_dataloader, epochs=1)
print(model.temperature)
trainer.train_predict_image()

# %%
# test image prediction
with torch.no_grad():
    index = random.randint(0, len(train_data))
    print(index)
    image = train_data[index][0].cuda()
    label = train_data[index][1]
    label = torch.Tensor([label]).int().cuda()
    print(label)
    plt.imshow(image.cpu().squeeze(0))
    x = model.encode_image(image.unsqueeze(0))
    y = model.encode_text(label)
    prediction = model._predict_image_layer(x)
    prediction = F.softmax(prediction, dim=1)
    print("prediction from image:", torch.argmax(prediction, dim=1).squeeze().tolist())
    prediction = model._predict_image_layer(y)
    prediction = F.softmax(prediction, dim=1)
    print("prediction from label:", torch.argmax(prediction, dim=1).squeeze().tolist())

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    print("x", torch.round(x, decimals=2))
    print("y", torch.round(y, decimals=2))
    print("similarity:", (x @ y.T).item())

# %%
# trainer = Trainer(model)
trainer.train_reconstruct_image(epochs=4)

# %%
# pass text to generate embedding and try to predict the label
with torch.no_grad():
    label = torch.arange(10).cuda()
    images = model.generate_image(label)
    # plt show all 10 images
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].cpu().squeeze(0))
# %%
