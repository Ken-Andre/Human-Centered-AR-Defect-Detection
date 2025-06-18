/*
 * ESP32 REAL SENSOR MODULE - AR/IoT Assembly Demo
 *
 * - Mesure en temps réel : température (DHT22 ou DHT11) et vibration analogique (SW-420 ou tout accéléromètre analogique)
 * - Envoi périodique (1-2s) vers API AR centrale via HTTP POST : http://<server_ip>:5000/sensors/<serial_number>
 * - Attribution du Serial Number par variable en haut de script
 *
 * Matériel : ESP32 Dev Module, capteur DHT, module vibration/accéléro analogique
 * Dépendances Arduino : WiFi.h, HTTPClient.h, DHT.h (Adafruit DHT sensor library)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"

// ==== CONFIGURATION UTILISATEUR ====
const char* WIFI_SSID     = "TON_SSID_WIFI";          // <--- À REMPLACER
const char* WIFI_PASS     = "TON_MOT_DE_PASSE_WIFI";  // <--- À REMPLACER

const char* SERVER_URL    = "http://192.168.1.50:5000";  // <--- Adresse API (par défaut, modifie si besoin)
const char* SERIAL_NUMBER = "SN-IM2025001";              // <--- Serial attribué à ce module

// ==== BROCHES CAPTEURS ====
#define DHTPIN        4        // GPIO pour DHT (par défaut GPIO4)
#define DHTTYPE       DHT22    // DHT22 (ou DHT11 si capteur bleu)
#define VIBRATION_PIN 36       // GPIO pour vibration analogique (ADC1_CH0, GPIO36 sur ESP32)

DHT dht(DHTPIN, DHTTYPE);

// ==== TEMPO ENVOI (ms) ====
unsigned long lastSend = 0;
unsigned long sendInterval = 1000 + random(0, 1000);  // entre 1000 et 2000 ms

// =============================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Connexion WiFi
  Serial.printf("[ESP32] Connexion à %s...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[ESP32] WiFi connecté !");
  Serial.print("[ESP32] Adresse IP : ");
  Serial.println(WiFi.localIP());

  dht.begin();
  analogReadResolution(12); // ADC 12 bits [0-4095]
  randomSeed(analogRead(0));
}

void loop() {
  if ((millis() - lastSend) > sendInterval) {
    sendSensorData();
    lastSend = millis();
    sendInterval = 1000 + random(0, 1000); // Prochain envoi dans 1-2s
  }
}

void sendSensorData() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ESP32] Erreur : WiFi non connecté");
    return;
  }

  // ---- Lecture Capteurs ----
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int vibrationRaw = analogRead(VIBRATION_PIN);
  float vibrationNorm = vibrationRaw / 4095.0; // normalisé entre 0 et 1

  // Vérif lecture capteur
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("[DHT] Erreur de lecture capteur DHT");
    return;
  }

  String postUrl = String(SERVER_URL) + "/sensors/" + SERIAL_NUMBER;
  String payload = "{";
  payload += "\"temperature\":" + String(temperature, 2) + ",";
  payload += "\"vibration\":" + String(vibrationNorm, 3) + ",";
  payload += "\"timestamp\":\"" + getISOTime() + "\"";
  payload += "}";

  HTTPClient http;
  http.begin(postUrl);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(payload);
  String response = http.getString();

  Serial.printf("[POST] %s\n", postUrl.c_str());
  Serial.printf("[DATA] %s\n", payload.c_str());
  Serial.printf("[HTTP] Code: %d\n", httpCode);
  Serial.printf("[RESP] %s\n\n", response.c_str());

  http.end();
}

// Timestamp ISO (simulé depuis uptime)
String getISOTime() {
  unsigned long nowMs = millis();
  unsigned long seconds = nowMs / 1000;
  unsigned long mins = seconds / 60;
  unsigned long hrs = mins / 60;

  char isoTime[25];
  sprintf(isoTime, "2025-06-18T%02lu:%02lu:%02lu", hrs % 24, mins % 60, seconds % 60);
  return String(isoTime);
}
