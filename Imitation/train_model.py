import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import typing, collections
typing.OrderedDict = collections.OrderedDict
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

class CarlaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """Carla dataset sınıfı"""
        json_path = os.path.join(data_dir, 'data_log.json')
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.data_dir = data_dir
        
        # Geçersiz verileri filtrele
        self.data = [item for item in self.data if os.path.exists(item['image_path'])]
        
        print(f"Dataset yüklendi: {len(self.data)} sample")
        
        # Transform tanımla
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Görüntüyü yükle
            image = Image.open(item['image_path']).convert('RGB')
            image = self.transform(image)
            
            # Kontrol verilerini hazırla
            steering = np.clip(item['steering'], -1.0, 1.0)
            throttle = np.clip(item['throttle'], 0.0, 1.0)
            brake = np.clip(item['brake'], 0.0, 1.0)
            
            controls = torch.tensor([steering, throttle, brake], dtype=torch.float32)
            
            return image, controls
            
        except Exception as e:
            print(f"Veri yükleme hatası (idx: {idx}): {e}")
            # Hata durumunda sıfır tensor döndür
            image = torch.zeros(3, 224, 224)
            controls = torch.zeros(3)
            return image, controls

class DrivingModel(nn.Module):
    def __init__(self, num_outputs=3):
        """ResNet18 tabanlı sürüş modeli"""
        super(DrivingModel, self).__init__()
        
        from torchvision.models import resnet18
        
        # Önceden eğitilmiş ResNet18 kullan
        self.backbone = resnet18(pretrained=True)
        
        # Son katmanı değiştir
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
        
        # Çıkış aktivasyon fonksiyonları
        self.steering_activation = nn.Tanh()  # [-1, 1]
        self.throttle_brake_activation = nn.Sigmoid()  # [0, 1]
        
    def forward(self, x):
        x = self.backbone(x)
        
        # Ayrı aktivasyon fonksiyonları uygula
        steering = self.steering_activation(x[:, 0:1])
        throttle = self.throttle_brake_activation(x[:, 1:2])
        brake = self.throttle_brake_activation(x[:, 2:3])
        
        return torch.cat([steering, throttle, brake], dim=1)

class SimpleCNN(nn.Module):
    def __init__(self):
        """Basit CNN modeli"""
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # İlk blok
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # İkinci blok
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Üçüncü blok
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
        
        # Aktivasyon fonksiyonları
        self.steering_activation = nn.Tanh()
        self.throttle_brake_activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        # Ayrı aktivasyon fonksiyonları
        steering = self.steering_activation(x[:, 0:1])
        throttle = self.throttle_brake_activation(x[:, 1:2])
        brake = self.throttle_brake_activation(x[:, 2:3])
        
        return torch.cat([steering, throttle, brake], dim=1)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir='models'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss fonksiyonu - ağırlıklı MSE
        self.criterion = nn.MSELoss()
        
        # Optimizer ve scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Metrikleri sakla
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, controls) in enumerate(self.train_loader):
            images, controls = images.to(self.device), controls.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Ağırlıklı loss hesapla (steering daha önemli)
            steering_loss = self.criterion(outputs[:, 0], controls[:, 0])
            throttle_loss = self.criterion(outputs[:, 1], controls[:, 1])
            brake_loss = self.criterion(outputs[:, 2], controls[:, 2])
            
            loss = 3.0 * steering_loss + 1.0 * throttle_loss + 1.0 * brake_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress göster
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{num_batches} - Loss: {loss.item():.6f} '
                      f'(Steering: {steering_loss.item():.6f}, '
                      f'Throttle: {throttle_loss.item():.6f}, '
                      f'Brake: {brake_loss.item():.6f})')
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        steering_errors = []
        
        with torch.no_grad():
            for images, controls in self.val_loader:
                images, controls = images.to(self.device), controls.to(self.device)
                outputs = self.model(images)
                
                # Loss hesapla
                steering_loss = self.criterion(outputs[:, 0], controls[:, 0])
                throttle_loss = self.criterion(outputs[:, 1], controls[:, 1])
                brake_loss = self.criterion(outputs[:, 2], controls[:, 2])
                
                loss = 3.0 * steering_loss + 1.0 * throttle_loss + 1.0 * brake_loss
                total_loss += loss.item()
                
                # Steering error'ları topla
                steering_error = torch.abs(outputs[:, 0] - controls[:, 0])
                steering_errors.extend(steering_error.cpu().numpy())
                
        avg_steering_error = np.mean(steering_errors)
        return total_loss / len(self.val_loader), avg_steering_error
    
    def train(self, epochs):
        """Ana eğitim döngüsü"""
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        max_patience = 10
        
        print(f"Eğitim başladı - Toplam {epochs} epoch")
        print(f"Device: {self.device}")
        print(f"Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f'\n{"="*50}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'{"="*50}')
            
            # Eğitim
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, steering_error = self.validate()
            
            # Metrikleri kaydet
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            print(f'\nEpoch {epoch+1} Sonuçları:')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
            print(f'Avg Steering Error: {steering_error:.6f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.8f}')
            
            # En iyi modeli kaydet
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                model_path = os.path.join(self.save_dir, 'best_driving_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'steering_error': steering_error
                }, model_path)
                
                print(f'✓ En iyi model kaydedildi! (Val Loss: {val_loss:.6f})')
            else:
                patience_counter += 1
                print(f'Patience: {patience_counter}/{max_patience}')
                
                if patience_counter >= max_patience:
                    print(f'\nEarly stopping! En iyi epoch: {best_epoch+1}')
                    break
            
            # Her 10 epoch'ta checkpoint kaydet
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, checkpoint_path)
        
        print(f'\nEğitim tamamlandı!')
        print(f'En iyi validation loss: {best_val_loss:.6f} (Epoch {best_epoch+1})')
        
        # Grafikleri çiz
        self.plot_training_history()
    
    def plot_training_history(self):
        """Eğitim grafiklerini çiz"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Detail')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Carla Model Eğitimi')
    parser.add_argument('--data_dir', required=True, help='Dataset klasörü')
    parser.add_argument('--model', default='resnet', choices=['resnet', 'simple'], 
                       help='Model tipi')
    parser.add_argument('--epochs', default=50, type=int, help='Epoch sayısı')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--save_dir', default='models', help='Model kayıt klasörü')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    print("Dataset yükleniyor...")
    dataset = CarlaDataset(args.data_dir)
    
    if len(dataset) < 100:
        print(f"UYARI: Dataset çok küçük ({len(dataset)} sample). En az 1000+ sample önerilir.")
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    if args.model == 'resnet':
        model = DrivingModel()
        print("ResNet18 tabanlı model kullanılıyor")
    else:
        model = SimpleCNN()
        print("Basit CNN model kullanılıyor")
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, device, args.save_dir)
    
    # Eğitimi başlat
    trainer.train(args.epochs)

if __name__ == "__main__":
    main()