#include <ArduinoJson.h>
const int trigPins[4]={5,18,19,21}; const int echoPins[4]={17,16,4,2};
float readDistanceM(int trigPin,int echoPin){ digitalWrite(trigPin,LOW); delayMicroseconds(2); digitalWrite(trigPin,HIGH); delayMicroseconds(10); digitalWrite(trigPin,LOW); long duration=pulseIn(echoPin,HIGH,30000); if(duration<=0) return 9.9f; return (duration*0.000343f)/2.0f; }
void setup(){ Serial.begin(115200); for(int i=0;i<4;++i){ pinMode(trigPins[i],OUTPUT); pinMode(echoPins[i],INPUT);} }
void loop(){ StaticJsonDocument<256> doc; JsonArray arr=doc.createNestedArray("distances_m"); for(int i=0;i<4;++i){ arr.add(readDistanceM(trigPins[i], echoPins[i])); delay(5);} serializeJson(doc,Serial); Serial.println(); delay(50);}