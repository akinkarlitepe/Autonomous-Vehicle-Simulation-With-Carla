import carla
import torch
import torch.nn as nn
import typing, collections
typing.OrderedDict = collections.OrderedDict
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import time
import argparse
import os
import pygame
import threading
import queue

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

class CarlaModelTester:
    def __init__(self, model_path, model_type='resnet', host='127.0.0.1', port=2000, enable_pygame=False):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Modeli yükle
        if model_type == 'resnet':
            self.model = DrivingModel()
        else:
            self.model = SimpleCNN()
            
        self.load_model(model_path)
        
        # Görüntü ön işleme
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.current_image = None
        self.vehicle = None
        self.camera = None
        self.spectator_camera = None
        
        # Pygame display
        self.enable_pygame = enable_pygame
        self.display = None
        self.clock = None
        self.image_queue = queue.Queue(maxsize=2)
        
        # İstatistikler
        self.frame_count = 0
        self.predictions = []
        
    def load_model(self, model_path):
        """Modeli yükler"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Checkpoint formatını kontrol et
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model yüklendi (Epoch: {checkpoint.get('epoch', 'N/A')}, "
                      f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
            else:
                # Eski format
                self.model.load_state_dict(checkpoint)
                print("Model yüklendi (eski format)")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Model yükleme hatası: {e}")
    
    def init_pygame_display(self):
        """Pygame display'ini başlatır"""
        if not self.enable_pygame:
            return
            
        pygame.init()
        self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Carla Model Test - Camera View')
        self.clock = pygame.time.Clock()
        print("Pygame display açıldı")
        
    def setup_vehicle_and_sensors(self, vehicle_filter='vehicle.tesla.model3'):
        """Araç ve sensörleri kurur"""
        try:
            # Mevcut weather'ı al
            weather = self.world.get_weather()
            print(f"Hava durumu: Cloudiness={weather.cloudiness}, "
                  f"Precipitation={weather.precipitation}, Sun={weather.sun_altitude_angle}")
            
            # Araç spawn et
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(vehicle_filter)[0]
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Rastgele spawn point seç
            spawn_point = np.random.choice(spawn_points)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            print(f"Araç spawn edildi: {spawn_point.location}")
            print(f"Araç tipi: {vehicle_bp.id}")
            
            # Kamera sensörü ekle
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0)
            )
            self.camera = self.world.spawn_actor(
                camera_bp, 
                camera_transform, 
                attach_to=self.vehicle
            )
            self.camera.listen(self.process_image)
            print("Kamera sensörü eklendi")
            
            # Spectator kamerayı aracı takip etmesi için ayarla
            self.setup_spectator_follow()
            
            # Pygame display'ini başlat
            if self.enable_pygame:
                self.init_pygame_display()
            
            # Başlangıçta biraz bekle
            time.sleep(2)
            
        except Exception as e:
            print(f"Kurulum hatası: {e}")
            self.cleanup()
            raise
    
    def setup_spectator_follow(self):
        """Spectator kamerayı aracı takip etmesi için ayarla"""
        try:
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            
            # Araçtan biraz uzakta ve yukarıda konumlandır
            spectator_transform = carla.Transform(
                carla.Location(
                    x=vehicle_transform.location.x - 8,
                    y=vehicle_transform.location.y,
                    z=vehicle_transform.location.z + 4
                ),
                carla.Rotation(
                    pitch=-15,
                    yaw=vehicle_transform.rotation.yaw
                )
            )
            spectator.set_transform(spectator_transform)
            print("Spectator kamera aracı takip etmeye ayarlandı")
            
        except Exception as e:
            print(f"Spectator ayarlama hatası: {e}")
    
    def update_spectator_position(self):
        """Spectator kamerayı aracın pozisyonuna göre günceller"""
        try:
            if self.vehicle is None:
                return
                
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            
            # Araçtan biraz arkada ve yukarıda takip et
            offset_x = -8 * np.cos(np.radians(vehicle_transform.rotation.yaw))
            offset_y = -8 * np.sin(np.radians(vehicle_transform.rotation.yaw))
            
            spectator_transform = carla.Transform(
                carla.Location(
                    x=vehicle_transform.location.x + offset_x,
                    y=vehicle_transform.location.y + offset_y,
                    z=vehicle_transform.location.z + 4
                ),
                carla.Rotation(
                    pitch=-15,
                    yaw=vehicle_transform.rotation.yaw
                )
            )
            spectator.set_transform(spectator_transform)
            
        except Exception as e:
            pass  # Sessizce geç, çok kritik değil
        
    def process_image(self, image):
        """Gelen görüntüyü işler"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # RGBA'dan RGB'ye
            self.current_image = array
            
            # Pygame için görüntüyü ekle
            if self.enable_pygame and self.image_queue.qsize() < 2:
                self.image_queue.put(array.copy())
            
        except Exception as e:
            print(f"Görüntü işleme hatası: {e}")
            self.current_image = None
    
    def update_pygame_display(self):
        """Pygame display'ini günceller"""
        if not self.enable_pygame or self.display is None:
            return
            
        # Pygame olaylarını kontrol et
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
        # Görüntüyü göster
        if not self.image_queue.empty():
            try:
                image = self.image_queue.get_nowait()
                # OpenCV'den Pygame'e format dönüşümü
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = np.rot90(image)
                image = np.flipud(image)
                image_surface = pygame.surfarray.make_surface(image)
                self.display.blit(image_surface, (0, 0))
                
                # Kontrol bilgilerini ekrana yazdır
                if hasattr(self, 'last_control'):
                    self.draw_control_info()
                
                pygame.display.flip()
                self.clock.tick(30)  # 30 FPS limit
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Pygame güncelleme hatası: {e}")
                
        return True
    
    def draw_control_info(self):
        """Kontrol bilgilerini pygame ekranına çizer"""
        if not hasattr(self, 'last_control'):
            return
            
        font = pygame.font.Font(None, 36)
        
        # Bilgi metinleri
        texts = [
            f"Steering: {self.last_control.steer:.3f}",
            f"Throttle: {self.last_control.throttle:.3f}",
            f"Brake: {self.last_control.brake:.3f}",
            f"Speed: {getattr(self, 'current_speed', 0):.1f} km/h",
            f"Frame: {self.frame_count}"
        ]
        
        # Metinleri çiz
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (255, 255, 255))
            # Siyah arka plan
            text_rect = text_surface.get_rect()
            pygame.draw.rect(self.display, (0, 0, 0), 
                           (10, 10 + i * 40, text_rect.width + 10, text_rect.height + 5))
            self.display.blit(text_surface, (15, 12 + i * 40))
        
    def predict_control(self):
        """Mevcut görüntüden kontrol tahmininde bulunur"""
        if self.current_image is None:
            return carla.VehicleControl()
            
        try:
            # Görüntüyü modele uygun hale getir
            image = Image.fromarray(self.current_image)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                prediction = self.model(image_tensor)
                steering, throttle, brake = prediction[0].cpu().numpy()
            
            # Değerleri sınırla
            steering = np.clip(steering, -1.0, 1.0)
            throttle = np.clip(throttle, 0.0, 1.0)
            brake = np.clip(brake, 0.0, 1.0)
            
            throttle = 0.5  
            brake    = 0.0

            # Throttle ve brake'i normalize et (ikisi birden aktif olmasın)
            if throttle > 0.0:
                brake = 0.0
            
            # Kontrol objesi oluştur
            control = carla.VehicleControl()
            control.steer = float(steering)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = False
            control.reverse = False
            
            # Son kontrolü sakla (pygame için)
            self.last_control = control
            
            # İstatistikleri kaydet
            self.predictions.append({
                'steering': steering,
                'throttle': throttle,
                'brake': brake
            })
            
            return control
            
        except Exception as e:
            print(f"Tahmin hatası: {e}")
            return carla.VehicleControl()
        
    def test_drive(self, duration_seconds=120, max_speed_kmh=20):
        """Modeli test eder"""
        print(f"Model testi başladı...")
        print(f"Süre: {duration_seconds} saniye")
        print(f"Maksimum hız: {max_speed_kmh} km/h")
        if self.enable_pygame:
            print("Pygame penceresi açık - görüntüyü takip edebilirsiniz")
        print("Ctrl+C ile durdurmak için...")
        
        start_time = time.time()
        last_stats_time = start_time
        last_spectator_update = start_time
        low_speed_counter = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                # Kontrol tahmininde bulun
                control = self.predict_control()
                
                # Hız kontrolü
                velocity = self.vehicle.get_velocity()
                speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                self.current_speed = speed_kmh  # Pygame için
                
                if speed_kmh > max_speed_kmh:
                    control.throttle = 0.0
                    control.brake = min(1.0, (speed_kmh - max_speed_kmh) / 10.0)
                
                # Kontrolü uygula
                self.vehicle.apply_control(control)
                
                # Spectator kamerayı güncelle (her 0.1 saniyede bir)
                current_time = time.time()
                if current_time - last_spectator_update >= 0.1:
                    self.update_spectator_position()
                    last_spectator_update = current_time
                
                # Pygame display'ini güncelle
                if self.enable_pygame:
                    if not self.update_pygame_display():
                        print("Pygame penceresi kapatıldı, test sonlandırılıyor...")
                        break
                
                # Her 5 saniyede bir istatistik göster
                if current_time - last_stats_time >= 5.0:
                    self.show_stats(speed_kmh, control)
                    last_stats_time = current_time
                
                # Araç çok yavaşsa veya durmuşsa restart
                #if speed_kmh < 1.0 and self.frame_count > 100:
                    #print("Araç durdu, pozisyon yenileniyor...")
                    #self.respawn_vehicle()

                    if speed_kmh < 1.0:
                        low_speed_counter += 1
                    else:
                        low_speed_counter = 0

                    if low_speed_counter > 40:   # örneğin 40 frame (~2 saniye) durunca
                        print("Uzun süre durdu, respawn ediliyor...")
                        self.respawn_vehicle()
                        low_speed_counter = 0
                
                self.world.tick()
                self.frame_count += 1
                time.sleep(0.05)  # 20 FPS
                
        except KeyboardInterrupt:
            print("\nTest kullanıcı tarafından durduruldu.")
        
        print(f"\nTest tamamlandı! Toplam {self.frame_count} frame işlendi.")
        self.show_final_stats()
        
    def show_stats(self, speed_kmh, control):
        """Anlık istatistikleri gösterir"""
        location = self.vehicle.get_transform().location
        print(f"Frame: {self.frame_count:5d} | "
              f"Speed: {speed_kmh:5.1f} km/h | "
              f"Steer: {control.steer:6.3f} | "
              f"Throttle: {control.throttle:5.3f} | "
              f"Brake: {control.brake:5.3f} | "
              f"Pos: ({location.x:.1f}, {location.y:.1f})")
    
    def show_final_stats(self):
        """Final istatistikleri gösterir"""
        if not self.predictions:
            return
            
        predictions = np.array([[p['steering'], p['throttle'], p['brake']] 
                               for p in self.predictions])
        
        print("\n" + "="*60)
        print("TEST İSTATİSTİKLERİ")
        print("="*60)
        print(f"Toplam frame: {len(predictions)}")
        print(f"Ortalama steering: {np.mean(predictions[:, 0]):.4f}")
        print(f"Steering std: {np.std(predictions[:, 0]):.4f}")
        print(f"Ortalama throttle: {np.mean(predictions[:, 1]):.4f}")
        print(f"Ortalama brake: {np.mean(predictions[:, 2]):.4f}")
        print(f"Max steering: {np.max(predictions[:, 0]):.4f}")
        print(f"Min steering: {np.min(predictions[:, 0]):.4f}")
        
    def respawn_vehicle(self):
        """Aracı yeniden konumlandırır"""
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            new_spawn_point = np.random.choice(spawn_points)
            self.vehicle.set_transform(new_spawn_point)
            self.setup_spectator_follow()  # Spectator'ı yeniden ayarla
            time.sleep(1)
        except Exception as e:
            print(f"Respawn hatası: {e}")
        
    def record_test(self, duration_seconds=60, save_dir='test_recordings'):
        """Test sırasında görüntü ve tahminleri kaydeder"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Test kaydı başladı - {duration_seconds} saniye")
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                if self.current_image is not None:
                    # Görüntüyü kaydet
                    image_path = os.path.join(save_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(image_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                    
                    # Kontrol tahmininde bulun
                    control = self.predict_control()
                    self.vehicle.apply_control(control)
                    
                    # Spectator'ı güncelle
                    self.update_spectator_position()
                    
                    # Pygame display'ini güncelle
                    if self.enable_pygame:
                        if not self.update_pygame_display():
                            break
                    
                    # Metadata kaydet
                    velocity = self.vehicle.get_velocity()
                    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    
                    metadata = {
                        'frame': frame_count,
                        'timestamp': time.time() - start_time,
                        'steering': float(control.steer),
                        'throttle': float(control.throttle),
                        'brake': float(control.brake),
                        'speed': float(speed)
                    }
                    
                    with open(os.path.join(save_dir, f"frame_{frame_count:06d}.json"), 'w') as f:
                        import json
                        json.dump(metadata, f)
                    
                    frame_count += 1
                
                self.world.tick()
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Kayıt durduruldu.")
        
        print(f"Toplam {frame_count} frame kaydedildi: {save_dir}")
        
    def cleanup(self):
        """Kaynakları temizler"""
        print("Temizlik yapılıyor...")
        
        # Pygame'i kapat
        if self.enable_pygame and self.display is not None:
            pygame.quit()
            
        if self.camera is not None:
            try:
                self.camera.destroy()
            except:
                pass
                
        if self.spectator_camera is not None:
            try:
                self.spectator_camera.destroy()
            except:
                pass
                
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description='Carla Model Test')
    parser.add_argument('--model_path', required=True, help='Model dosya yolu')
    parser.add_argument('--model_type', default='resnet', choices=['resnet', 'simple'],
                       help='Model tipi')
    parser.add_argument('--host', default='127.0.0.1', help='Carla host')
    parser.add_argument('--port', default=2000, type=int, help='Carla port')
    parser.add_argument('--duration', default=120, type=int, help='Test süresi (saniye)')
    parser.add_argument('--max_speed', default=50, type=int, help='Maksimum hız (km/h)')
    parser.add_argument('--vehicle', default='vehicle.tesla.model3', help='Araç tipi')
    parser.add_argument('--record', action='store_true', help='Test sırasında kayıt yap')
    parser.add_argument('--weather', choices=['clear', 'rain', 'fog'], default='clear',
                       help='Hava durumu')
    parser.add_argument('--enable_pygame', action='store_true', 
                       help='Pygame ile görüntü gösterimini etkinleştir')
    parser.add_argument('--no_spectator', action='store_true',
                       help='Spectator kamera takibini devre dışı bırak')
    
    args = parser.parse_args()
    
    tester = CarlaModelTester(
        model_path=args.model_path,
        model_type=args.model_type,
        host=args.host,
        port=args.port,
        enable_pygame=args.enable_pygame
    )
    
    try:
        # Hava durumunu ayarla
        weather_preset = carla.WeatherParameters.ClearNoon
        if args.weather == 'rain':
            weather_preset = carla.WeatherParameters.HardRainNoon
        elif args.weather == 'fog':
            weather_preset = carla.WeatherParameters.CloudyNoon
            weather_preset.fog_density = 50.0
            
        tester.world.set_weather(weather_preset)
        print(f"Hava durumu ayarlandı: {args.weather}")
        
        # Test kurulumu
        tester.setup_vehicle_and_sensors(vehicle_filter=args.vehicle)
        
        # Test veya kayıt
        if args.record:
            tester.record_test(duration_seconds=args.duration)
        else:
            tester.test_drive(
                duration_seconds=args.duration,
                max_speed_kmh=args.max_speed
            )
            
    except Exception as e:
        print(f"Test hatası: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()