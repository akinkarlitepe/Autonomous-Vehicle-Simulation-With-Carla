import carla
import cv2
import numpy as np
import json
import os
import time
import argparse
from datetime import datetime

class DataCollector:
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Veri klasörü oluştur
        self.data_dir = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(f"{self.data_dir}/images", exist_ok=True)
        
        self.data_log = []
        self.frame_count = 0
        self.vehicle = None
        self.camera = None
        
    def setup_vehicle_and_sensors(self):
        """Araç ve sensörleri kurur"""
        try:
            # Araç spawn et
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0]
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Araç spawn edildi: {spawn_point.location}")
            
            # Autopilot'u etkinleştir
            self.vehicle.set_autopilot(True)
            print("Autopilot etkinleştirildi")
            
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
            
        except Exception as e:
            print(f"Kurulum hatası: {e}")
            self.cleanup()
            raise
        
    def process_image(self, image):
        """Gelen görüntüyü işler ve kontrol verilerini kaydeder"""
        try:
            # Görüntüyü numpy array'e çevir
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # RGBA'dan RGB'ye
            
            # Görüntüyü kaydet
            image_path = f"{self.data_dir}/images/frame_{self.frame_count:06d}.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            
            # Aracın kontrollerini al
            control = self.vehicle.get_control()
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Veriyi kaydet
            data_point = {
                'frame': self.frame_count,
                'image_path': image_path,
                'steering': float(control.steer),
                'throttle': float(control.throttle),
                'brake': float(control.brake),
                'speed': float(speed),
                'timestamp': image.timestamp
            }
            
            self.data_log.append(data_point)
            self.frame_count += 1
            
            # Her 100 frame'de bir bilgi yazdır
            if self.frame_count % 100 == 0:
                print(f"Frame {self.frame_count}: Speed={speed:.1f}km/h, "
                      f"Steering={control.steer:.3f}, Throttle={control.throttle:.3f}, "
                      f"Brake={control.brake:.3f}")
                
        except Exception as e:
            print(f"Görüntü işleme hatası: {e}")
        
    def collect_data(self, duration_seconds=300):
        """Belirtilen süre boyunca veri toplar"""
        print(f"Veri toplama başladı... {duration_seconds} saniye boyunca devam edecek.")
        print(f"Veriler '{self.data_dir}' klasörüne kaydediliyor.")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                self.world.tick()
                time.sleep(0.05)  # 20 FPS
                
                # Araç durmuşsa yeniden başlat
                velocity = self.vehicle.get_velocity()
                speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                if speed < 1.0 and self.frame_count > 100:  # İlk 100 frame'i atla
                    print("Araç durdu, yeniden spawn ediliyor...")
                    self.respawn_vehicle()
                    
        except KeyboardInterrupt:
            print("\nVeri toplama kullanıcı tarafından durduruldu.")
        
        # Verileri JSON dosyasına kaydet
        json_path = f"{self.data_dir}/data_log.json"
        with open(json_path, 'w') as f:
            json.dump(self.data_log, f, indent=2)
            
        print(f"\nVeri toplama tamamlandı!")
        print(f"Toplam {len(self.data_log)} frame kaydedildi.")
        print(f"Veriler: {json_path}")
        
    def respawn_vehicle(self):
        """Aracı yeniden spawn eder"""
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            new_spawn_point = np.random.choice(spawn_points)
            self.vehicle.set_transform(new_spawn_point)
            self.vehicle.set_autopilot(True)
            time.sleep(2)  # Aracın stabilize olması için bekle
        except Exception as e:
            print(f"Respawn hatası: {e}")
        
    def cleanup(self):
        """Kaynakları temizler"""
        print("Temizlik yapılıyor...")
        if self.camera is not None:
            try:
                self.camera.destroy()
            except:
                pass
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description='Carla Veri Toplama')
    parser.add_argument('--host', default='127.0.0.1', help='Carla host')
    parser.add_argument('--port', default=2000, type=int, help='Carla port')
    parser.add_argument('--duration', default=300, type=int, 
                       help='Veri toplama süresi (saniye)')
    
    args = parser.parse_args()
    
    collector = DataCollector(host=args.host, port=args.port)
    
    try:
        collector.setup_vehicle_and_sensors()
        collector.collect_data(duration_seconds=args.duration)
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()