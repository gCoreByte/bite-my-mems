#include <Arduino.h>
#include "HX711.h"
//https://github.com/RobTillaart/HX711
#include <EEPROM.h>
#include <driver/i2s.h>
#include <driver/adc.h>
#include <freertos/semphr.h>

const int BAUD_RATE = 460800;
const uint16_t MAGIC_SLOW = 0xABCD;
const uint16_t MAGIC_MEMS = 0xABCE;

// HX711 circuit wiring
HX711 scale;
const int LOADCELL_DOUT_PIN = 19;
const int LOADCELL_SCK_PIN = 18;

float loadcell_scale;

// FSR pins (ADC1)
const int fsrPins[] = {36, 39};
const int fsrPinsSize = sizeof(fsrPins) / sizeof(fsrPins[0]);
const adc1_channel_t FSR_CHANNELS[] = {ADC1_CHANNEL_0, ADC1_CHANNEL_3}; // GPIO 36, 39

// MEMS mic (ADC1_CHANNEL_6 = GPIO 34)
const int MEMS_SAMPLE_RATE = 22050;
const int MEMS_DMA_BUF_LEN = 256;
const int MEMS_DMA_BUF_COUNT = 4;

// Serial mutex (both cores write to Serial)
SemaphoreHandle_t serialMutex;

//This is for sending data every sendCooldown, until then data is saved in a buffer
const unsigned long SEND_COOLDOWN = 1000;
unsigned long nextSendTime = SEND_COOLDOWN;
unsigned long start_time = 0; //Used for subtracting from timestamp

bool inCalibration = false;
const int BUFFER_SIZE = 20;
int bufferIndex = 0;

struct Reading {
  int loadcell;
  int fsr[fsrPinsSize];
  int timestamp;
};

Reading buffer[BUFFER_SIZE];

uint8_t xorChecksum(const uint8_t* data, size_t len) {
  uint8_t cs = 0;
  for (size_t i = 0; i < len; i++) cs ^= data[i];
  return cs;
}

// MEMS task running on Core 0
// Packet: header(8) + samples(count * 2 bytes, uint16 LE) + checksum(1)
void memsTask(void* param) {
  uint16_t i2sRaw[MEMS_DMA_BUF_LEN];
  uint8_t packet[8 + MEMS_DMA_BUF_LEN * 2 + 1];

  while (true) {
    size_t bytesRead = 0;
    esp_err_t err = i2s_read(I2S_NUM_0, i2sRaw, sizeof(i2sRaw), &bytesRead, portMAX_DELAY);
    if (err != ESP_OK || bytesRead == 0) continue;

    int sampleCount = bytesRead / sizeof(uint16_t);

    uint16_t count = (uint16_t)sampleCount;
    uint32_t ts = millis();
    memcpy(packet + 0, &MAGIC_MEMS, 2);
    memcpy(packet + 2, &count, 2);
    memcpy(packet + 4, &ts, 4);

    // Store 12-bit ADC values as uint16 LE (2 bytes per sample)
    for (int i = 0; i < sampleCount; i++) {
      uint16_t val = i2sRaw[i] & 0x0FFF;
      packet[8 + i * 2] = val & 0xFF;
      packet[8 + i * 2 + 1] = (val >> 8) & 0xFF;
    }

    int payloadSize = sampleCount * 2;
    packet[8 + payloadSize] = xorChecksum(packet + 8, payloadSize);

    xSemaphoreTake(serialMutex, portMAX_DELAY);
    Serial.write(packet, 8 + payloadSize + 1);
    xSemaphoreGive(serialMutex);
  }
}

void collectData(){
  if (!scale.is_ready()) {
    return;  // HX711 not ready — don't block, just skip this loop iteration
  }
  buffer[bufferIndex].loadcell = scale.read();

  // Skip FSR reads for now — i2s_adc_disable/enable may disrupt MEMS
  for (int i = 0; i < fsrPinsSize; i++) {
    buffer[bufferIndex].fsr[i] = 0;
  }

  buffer[bufferIndex].timestamp = millis() - start_time;
  bufferIndex++;
}

void printReadings() {
  if (bufferIndex == 0) return;

  uint16_t sendTime = millis() - start_time;

  xSemaphoreTake(serialMutex, portMAX_DELAY);
  Serial.write((uint8_t*)&MAGIC_SLOW, 2);
  Serial.write((uint8_t*)&bufferIndex, 2);
  Serial.write((uint8_t*)&sendTime, 2);
  Serial.write((uint8_t*)buffer, sizeof(Reading) * bufferIndex);
  xSemaphoreGive(serialMutex);
}

void setup() {
  Serial.begin(BAUD_RATE);

  // ADC config for all ADC1 channels
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_0);  // 0-1.1V range for better mic resolution
  for (int i = 0; i < fsrPinsSize; i++) {
    adc1_config_channel_atten(FSR_CHANNELS[i], ADC_ATTEN_DB_11);
  }

  // I2S ADC DMA for MEMS mic
  i2s_config_t i2sConfig = {};
  i2sConfig.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
  i2sConfig.sample_rate = MEMS_SAMPLE_RATE;
  i2sConfig.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  i2sConfig.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  i2sConfig.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  i2sConfig.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  i2sConfig.dma_buf_count = MEMS_DMA_BUF_COUNT;
  i2sConfig.dma_buf_len = MEMS_DMA_BUF_LEN;
  i2sConfig.use_apll = false;

  i2s_driver_install(I2S_NUM_0, &i2sConfig, 0, NULL);
  i2s_set_adc_mode(ADC_UNIT_1, ADC1_CHANNEL_6);
  i2s_adc_enable(I2S_NUM_0);

  // Mutex + MEMS task BEFORE HX711
  serialMutex = xSemaphoreCreateMutex();
  xTaskCreatePinnedToCore(memsTask, "MEMS", 8192, NULL, 5, NULL, 0);

  // HX711
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  // scale.set_rate_80SPS();   // disabled — non-default config, suspect cause of HX711 hang
  // scale.set_raw_mode();      // disabled — non-default config, suspect cause of HX711 hang

  Serial.println("HX711_INIT_DONE");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis > nextSendTime || bufferIndex == BUFFER_SIZE){
    nextSendTime = currentMillis + SEND_COOLDOWN;
    printReadings();
    start_time = currentMillis;
    bufferIndex = 0;
  }
  collectData();
}
