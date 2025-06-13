#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

String my_watch = "c2:ab:11:12:63:ac";
String iphone_name = "hieu_iphone";
int rssi_watch = 777;
int rssi_iphone = 777;
float acc_x,acc_y,acc_z,gcc_x,gcc_y,gcc_z,mcc_x,mcc_y,mcc_z;

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

void loop() {
  BLEDevice device = BLE.available();  // Get next available device

  while (device) {
    String address = device.address();
    String name = device.localName();

    if (address == my_watch) {
      rssi_watch = device.rssi();
    }

    if (name == iphone_name) {
      rssi_iphone = device.rssi();
    }

    device = BLE.available();  // Get next device
  }

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(acc_x,acc_y,acc_z);
    IMU.readGyroscope(gcc_x, gcc_y, gcc_z);
    IMU.readMagneticField(mcc_x, mcc_y, mcc_z);
    String address = device.address();
    String name = device.localName();
    Serial.print(name);

    Serial.print(millis());
    Serial.print(" | ");
    Serial.print(acc_x);
    Serial.print(" | ");
    Serial.print(acc_y);
    Serial.print(" | ");
    Serial.print(acc_z);
    Serial.print(" | ");
    Serial.print(gcc_x);
    Serial.print(" | ");
    Serial.print(gcc_y);
    Serial.print(" | ");
    Serial.print(gcc_z);
    Serial.print(" | ");
    Serial.print(mcc_x);
    Serial.print(" | ");
    Serial.print(mcc_y);
    Serial.print(" | ");
    Serial.print(mcc_z);
    Serial.print(" | ");
    Serial.print(rssi_watch);
    Serial.print(" | "); 
    Serial.print(rssi_iphone);
    Serial.println();
    
  }
}

