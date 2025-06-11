#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#define DHTPIN 4
#define DHTTYPE DHT22
#define VIBPIN 5

DHT dht(DHTPIN, DHTTYPE);
const char* ssid = "votre_wifi";
const char* password = "votre_mot_de_passe";
const char* server = "http://192.168.1.100:8080/SN-IM2025007";

void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(VIBPIN, INPUT);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connexion WiFi...");
  }
}

void loop() {
  float temp = dht.readTemperature();
  int vib = digitalRead(VIBPIN);
  HTTPClient http;
  http.begin(server);
  http.addHeader("Content-Type", "application/json");
  String payload = "{\"temp\":" + String(temp) + ",\"vib\":" + String(vib) + "}";
  http.POST(payload);
  http.end();
  delay(1000);
}