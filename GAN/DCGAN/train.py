import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard  import SummaryWriter
from model import discriminator,generator,initialize_weights
import torchvision

#hyperparamters

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=28
image_size=64
channels=1
learing_rate=2e-4
z_dim=100
epochs=5
features_disc=64
features_gen=64

transforms=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(channels)],[0.5 for _ in range(channels)]
    ),
])


dataset=datasets.MNIST(root="dataset/",train=True,download=True,transform=transforms)
 
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

gen=generator(z_dim,channels,features_gen).to(device)
disc=discriminator(channels,features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

optimizer_gen=optim.Adam(gen.parameters(),lr=learing_rate,betas=(0.5,0.999))
optimizer_disc=optim.Adam(disc.parameters(),lr=learing_rate,betas=(0.5,0.999))

criterion=nn.BCELoss()
fixed_noise=torch.randn(batch_size,z_dim,1,1).to(device)
writer_real=SummaryWriter(f"logs/real")
writer_fake=SummaryWriter(f"logs/fake")
step=0

gen.train()
disc.train()

for epoch in range(epochs):
    for batch_idx,(real,_) in enumerate(dataloader):
        real=real.to(device)
        noise=torch.randn(batch_size,z_dim,1,1).to(device)
        fake=gen(noise)
        ## trian discriminator
        disc_real=disc(real).reshape(-1)
        lossdisc_real=criterion(disc_real,torch.ones_like(disc_real))

        disc_fake=disc(fake).reshape(-1)
        lossdisc_fake=criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_disc=(lossdisc_real+lossdisc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        optimizer_disc.step()

        #train generator 

        output=disc(fake).reshape(-1)
        loss_gen=criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

