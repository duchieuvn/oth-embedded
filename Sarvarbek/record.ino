#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>


void setup() {
  Serial.begin(115200);
  while (!Serial);  // Wait for serial connection

  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1);
  }
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  BLE.scan();
}

String my_watch = "c2:ab:11:12:63:ac";
String iphone_name = "hieu_iphone";
int rssi_watch = 999;
int rssi_iphone = 999;
float acc_x,acc_y,acc_z;
unsigned long lastBLEScan = 0;
const unsigned long BLE_SCAN_INTERVAL = 1000;  // milliseconds


bool found_watch = false;
bool found_iphone = false;

void loop() {
  // Scan for BLE each 1000ms
  if (millis() - lastBLEScan >= BLE_SCAN_INTERVAL) {
    BLEDevice device = BLE.available();
    found_watch = false; 
    found_iphone = false;  

    while (device) {
      String address = device.address();
      String name = device.localName();

      if (!found_watch && address == my_watch) {
          rssi_watch = device.rssi();
          found_watch = true;
      }
      if (!found_iphone && name == iphone_name) {
          rssi_iphone = device.rssi();
          found_iphone = true;
      }
      if (found_watch && found_iphone) {
        break;
      }
      
      device = BLE.available();
    }
    lastBLEScan = millis();
  }

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(acc_x, acc_y, acc_z);

    char buffer[96];

    snprintf(buffer, sizeof(buffer), "%lu,%.2f,%.2f,%.2f,%d,%d",
         millis(), acc_x, acc_y, acc_z, rssi_watch, rssi_iphone);
    Serial.println(buffer);
  }
}
