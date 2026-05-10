#include <ArduinoJson.h>
#define RIGHT_LPWM 25
#define RIGHT_RPWM 26
#define LEFT_LPWM 32
#define LEFT_RPWM 33
#define STEER_LPWM 22
#define STEER_RPWM 23
#define PWM_PIN 16
#define START_SWITCH_PIN 4
#define PWM_FREQ 1000
#define PWM_RES 8
#define CH_RIGHT_L 0
#define CH_RIGHT_R 1
#define CH_LEFT_L 2
#define CH_LEFT_R 3
#define CH_STEER_L 4
#define CH_STEER_R 5
float center_angle=0.0f, filtered_angle=0.0f, steer_ratio=30.0f/45.0f, alpha=0.25f; unsigned long last_report_ms=0;
float readAngle(){ uint32_t hi=pulseIn(PWM_PIN,HIGH,50000), lo=pulseIn(PWM_PIN,LOW,50000); if(!hi||!lo) return -1000.0f; return ((float)hi/(float)(hi+lo))*360.0f; }
float calibrateCenter(){ float s=0.0f; int c=0; for(int i=0;i<120;++i){ float a=readAngle(); if(a>-999.0f){s+=a; c++;} delay(5);} return c>0?s/c:0.0f; }
void stopAll(){ ledcWrite(CH_RIGHT_L,0); ledcWrite(CH_RIGHT_R,0); ledcWrite(CH_LEFT_L,0); ledcWrite(CH_LEFT_R,0); ledcWrite(CH_STEER_L,0); ledcWrite(CH_STEER_R,0);} 
void applyDrive(int gear,int pwm){ pwm=constrain(pwm,0,255); if(gear>=0){ledcWrite(CH_RIGHT_L,0); ledcWrite(CH_RIGHT_R,pwm); ledcWrite(CH_LEFT_L,0); ledcWrite(CH_LEFT_R,pwm);} else {ledcWrite(CH_RIGHT_L,pwm); ledcWrite(CH_RIGHT_R,0); ledcWrite(CH_LEFT_L,pwm); ledcWrite(CH_LEFT_R,0);} }
void applySteerPwm(int pwm){ if(pwm>0){ledcWrite(CH_STEER_L,0); ledcWrite(CH_STEER_R,constrain(pwm,0,255));} else if(pwm<0){ledcWrite(CH_STEER_R,0); ledcWrite(CH_STEER_L,constrain(-pwm,0,255));} else {ledcWrite(CH_STEER_L,0); ledcWrite(CH_STEER_R,0);} }
int steerController(float target_deg){ float enc=readAngle(); if(enc<-999.0f) return 0; float sw=enc-center_angle; if(sw>180.0f) sw-=360.0f; if(sw<-180.0f) sw+=360.0f; float wheel=sw*steer_ratio; filtered_angle=alpha*wheel+(1.0f-alpha)*filtered_angle; return constrain((int)(4.5f*(target_deg-filtered_angle)),-220,220); }
void reportState(){ StaticJsonDocument<256> doc; doc["start_switch"]=digitalRead(START_SWITCH_PIN)==LOW; doc["steer_deg"]=filtered_angle; serializeJson(doc,Serial); Serial.println(); }
void setup(){ Serial.begin(115200); pinMode(PWM_PIN,INPUT); pinMode(START_SWITCH_PIN,INPUT_PULLUP); ledcSetup(CH_RIGHT_L,PWM_FREQ,PWM_RES); ledcSetup(CH_RIGHT_R,PWM_FREQ,PWM_RES); ledcSetup(CH_LEFT_L,PWM_FREQ,PWM_RES); ledcSetup(CH_LEFT_R,PWM_FREQ,PWM_RES); ledcSetup(CH_STEER_L,PWM_FREQ,PWM_RES); ledcSetup(CH_STEER_R,PWM_FREQ,PWM_RES); ledcAttachPin(RIGHT_LPWM,CH_RIGHT_L); ledcAttachPin(RIGHT_RPWM,CH_RIGHT_R); ledcAttachPin(LEFT_LPWM,CH_LEFT_L); ledcAttachPin(LEFT_RPWM,CH_LEFT_R); ledcAttachPin(STEER_LPWM,CH_STEER_L); ledcAttachPin(STEER_RPWM,CH_STEER_R); stopAll(); delay(1500); center_angle=calibrateCenter(); }
void loop(){ if(Serial.available()){ String line=Serial.readStringUntil('
'); StaticJsonDocument<256> doc; if(deserializeJson(doc,line)==DeserializationError::Ok){ const char* type=doc["type"]|""; if(strcmp(type,"stop")==0){ stopAll(); } else if(strcmp(type,"drive")==0){ int gear=doc["gear"]|1; float speed=doc["speed_mps"]|0.0; float steer=doc["steer_deg"]|0.0; int pwm=constrain((int)(speed*255.0/1.0),0,180); applyDrive(gear,pwm); applySteerPwm(steerController(steer)); } } } if(millis()-last_report_ms>100){ last_report_ms=millis(); reportState(); } }
