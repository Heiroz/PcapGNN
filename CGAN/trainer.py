import torch
import torch.nn as nn
import torch.optim as optim
from decode import decode_tensor
class CGANTrainer:
    def __init__(self, generator, discriminator, data_loader, noisy_dim, 
                 num_epochs=50, lr=0.0002, beta1=0.5, beta2=0.999):
        self.generator = generator
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.z_dim = noisy_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (condition_vector, real_data) in enumerate(self.data_loader):
                real_data = real_data.to(self.device)
                condition_vector = condition_vector.to(self.device)
                self.discriminator.zero_grad()
                # Extract num_generated and reshape real_data
                bs, num_generated, size = real_data.shape
                labels_real = torch.ones(real_data.size(0) * real_data.size(1), 1, device=self.device)
                # Generate fake data and reshape
                fake_data = []
                for _ in range(num_generated):
                    z = torch.randn(bs, self.z_dim, device=self.device)
                    sample = self.generator(z, condition_vector)
                    fake_data.append(sample)
                fake_data = torch.stack(fake_data, dim=1)
                fake_data = fake_data.squeeze(dim=2)
                labels_fake = torch.zeros(fake_data.size(0) * fake_data.size(1), 1, device=self.device)

                # Calculate discriminator loss
                expanded_condition_vector = condition_vector.unsqueeze(1).repeat(1, num_generated, 1)
                real_data = real_data.reshape(-1, real_data.shape[2])
                fake_data = fake_data.reshape(-1, fake_data.shape[2])
                expanded_condition_vector = expanded_condition_vector.reshape(-1, expanded_condition_vector.shape[2])

                output_real = self.discriminator(real_data, expanded_condition_vector)
                # print(output_real)

                output_fake = self.discriminator(fake_data.detach(), expanded_condition_vector)
                loss_D_real = self.criterion(output_real, labels_real)
                loss_D_fake = self.criterion(output_fake, labels_fake)
                loss_D = loss_D_real + loss_D_fake

                loss_D.backward()
                self.optimizer_D.step()

                self.generator.zero_grad()

                # Re-generate fake data for generator update
                fake_data = []
                for _ in range(num_generated):
                    z = torch.randn(bs, self.z_dim, device=self.device)
                    sample = self.generator(z, condition_vector)
                    fake_data.append(sample)
                fake_data = torch.stack(fake_data, dim=1)
                fake_data = fake_data.reshape(-1, self.generator.output_size)
                output_fake = self.discriminator(fake_data, expanded_condition_vector)
                loss_G = self.criterion(output_fake, labels_real)

                loss_G.backward()
                self.optimizer_G.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.data_loader)}], "
                        f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
                    # raw_data = decode_tensor(real_data, 32)
                    # generated_data = decode_tensor(fake_data, 32)
                    # print(f"raw_data: {raw_data}")
                    # print(f"generated_data: {generated_data}")

            torch.save(self.generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(self.discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

        print("Training finished.")
