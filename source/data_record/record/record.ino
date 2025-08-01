#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

const int BATCH_SIZE = 10;
String dataBuffer = "";

void setup() {
  Serial.begin(115200);
  while (!Serial);

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
float acc_x, acc_y, acc_z;
float gyro_x, gyro_y, gyro_z;

unsigned long lastBLEScan = 0;
const unsigned long BLE_SCAN_INTERVAL = 1000;  // milliseconds

bool found_watch = false;
bool found_iphone = false;

int batchCount = 0;

void loop() {
  // Read and buffer IMU data
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(acc_x, acc_y, acc_z);
    IMU.readGyroscope(gyro_x, gyro_y, gyro_z);

    char line[128];
    snprintf(line, sizeof(line), "%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
             millis(), acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, rssi_watch, rssi_iphone);

    dataBuffer += line;
    batchCount++;

    if (batchCount >= BATCH_SIZE) {
      Serial.print(dataBuffer);    
      dataBuffer = "";           
      batchCount = 0;
    }
  }
}
