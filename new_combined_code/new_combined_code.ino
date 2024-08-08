#include <Arduino_LSM9DS1.h>
#include <BikeAlarm_inferencing.h>  // Include the new model header file

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f        
#define VIBRATION_SENSOR_PIN 12         // Pin where the vibration sensor is connected
#define SENSOR_PIN          A6
#define LED_PIN             13          // Pin for the built-in LED
#define SAMPLE_RATE         100         // Sampling rate in Hz
#define SAMPLE_PERIOD       1000 / SAMPLE_RATE
#define THRESHOLD           1000
#define MIN_COUNT           3
#define MAX_COUNT           15

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

bool lock = false;
int highCount = 0;
int lowCount = 0;
enum State { IDLE, HIGH1, LOW_PHASE, HIGH2 };
State currentState = IDLE;

unsigned long lastSwitchTime = 0;
const unsigned long unlockPeriod = 5000; // 10 seconds
const int motionLoops = 5;

/**
* @brief      Arduino setup function
*/
void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("Start collecting data...");

    // Initialize IMU
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
    } else {
        Serial.println("IMU initialized");
    }

    // Initialize pins
    pinMode(SPEAKER_PIN, OUTPUT);
    digitalWrite(SPEAKER_PIN, LOW);

    pinMode(VIBRATION_SENSOR_PIN, INPUT);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW); // LED initially off
}

/**
* @brief      Detect tap pattern to lock/unlock
*/
bool detectPattern() {
    int sensorValue = analogRead(SENSOR_PIN);
    Serial.println(sensorValue);

    switch (currentState) {
        case IDLE:
            if (sensorValue > THRESHOLD) {
                highCount++;
                if (highCount >= MIN_COUNT && highCount <= MAX_COUNT) {
                    currentState = HIGH1;
                }
            } else {
                highCount = 0;
            }
            break;

        case HIGH1:
            if (sensorValue < THRESHOLD) {
                lowCount++;
                if (lowCount >= MIN_COUNT && lowCount <= MAX_COUNT) {
                    currentState = LOW_PHASE;
                }
            } else {
                lowCount = 0;
                highCount++;
                if (highCount > MAX_COUNT) {
                    currentState = IDLE;
                    highCount = 0;
                }
            }
            break;

        case LOW_PHASE:
            if (sensorValue > THRESHOLD) {
                highCount++;
                if (highCount >= MIN_COUNT && highCount <= MAX_COUNT) {
                    currentState = HIGH2;
                }
            } else {
                highCount = 0;
                lowCount++;
                if (lowCount > MAX_COUNT) {
                    currentState = IDLE;
                    lowCount = 0;
                }
            }
            break;

        case HIGH2:
            lock = !lock;
            Serial.print("Lock status: ");
            Serial.println(lock ? "locked" : "unlocked");
            digitalWrite(LED_PIN, lock ? LOW : HIGH); // Turn off LED if locked, on if unlocked
            currentState = IDLE;
            highCount = 0;
            lowCount = 0;
            return true;
    }

    delay(SAMPLE_PERIOD);
    return false;
}

/**
* @brief      Detect motion and trigger alarm
*/
void detectMotion() {
    bool vibrationDetected = digitalRead(VIBRATION_SENSOR_PIN) == HIGH;

    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);

        for (int i = 0; i < 3; i++) {
            if (fabs(buffer[ix + i]) > MAX_ACCEPTED_RANGE) {
                buffer[ix + i] = ei_get_sign(buffer[ix + i]) * MAX_ACCEPTED_RANGE;
            }
        }

        buffer[ix + 0] *= CONVERT_G_TO_MS2;
        buffer[ix + 1] *= CONVERT_G_TO_MS2;
        buffer[ix + 2] *= CONVERT_G_TO_MS2;

        delayMicroseconds(next_tick - micros());
    }

    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        Serial.print("Failed to create signal from buffer\n");
        return;
    }

    ei_impulse_result_t result = { 0 };
    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        Serial.print("Failed to run classifier\n");
        return;
    }

    Serial.print("Predictions ");
    Serial.print("(DSP: ");
    Serial.print(result.timing.dsp);
    Serial.print(" ms., Classification: ");
    Serial.print(result.timing.classification);
    Serial.print(" ms., Anomaly: ");
    Serial.print(result.timing.anomaly);
    Serial.println(" ms.):");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        Serial.print("    ");
        Serial.print(result.classification[ix].label);
        Serial.print(": ");
        Serial.println(result.classification[ix].value, 5);
    }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
    Serial.print("    anomaly score: ");
    Serial.println(result.anomaly, 3);
#endif

    String detectedMovement = "";
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > 0.5) {
            detectedMovement = result.classification[ix].label;
            break;
        }
    }

    if (detectedMovement == "alarm") {
        playReggaetonBeat(10000);
    }

    delay(100);
}

void loop() {
    if (!lock) {
        // Detect lock pattern
        if (detectPattern()) {
            lastSwitchTime = millis(); // Reset the time after lock
        }
    } else {
        // Detect motion for 3 loops
        for (int i = 0; i < motionLoops; i++) {
            detectMotion();
        }

        // Allow 10 seconds for unlocking
        unsigned long startUnlockTime = millis();
        digitalWrite(LED_PIN, HIGH); // Turn on LED during unlocking period
        while (millis() - startUnlockTime < unlockPeriod) {
            if (detectPattern()) {
                lastSwitchTime = millis(); // Reset the time after unlock
                break;
            }
        }
        digitalWrite(LED_PIN, LOW); // Turn off LED after unlocking period
    }
}

float ei_get_sign(float number) {
    return (number >= 0.0) ? 1.0 : -1.0;
}

void playReggaetonBeat(unsigned long duration) {
    int bassDrumFrequency = 60;
    int snareDrumFrequency = 400;
    int hiHatFrequency = 1000;
    unsigned long startTime = millis();

    while (millis() - startTime < duration) {
        tone(SPEAKER_PIN, bassDrumFrequency, 200);
        delay(300);
        tone(SPEAKER_PIN, snareDrumFrequency, 150);
        delay(150);
        tone(SPEAKER_PIN, hiHatFrequency, 100);
        delay(200);
        tone(SPEAKER_PIN, bassDrumFrequency, 200);
        delay(300);
        tone(SPEAKER_PIN, hiHatFrequency, 100);
        delay(100);
        tone(SPEAKER_PIN, snareDrumFrequency, 150);
        delay(150);
    }
}