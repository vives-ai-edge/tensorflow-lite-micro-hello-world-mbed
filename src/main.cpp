#include "mbed.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

using namespace tflite;

EventQueue queue;

// Application constants and variables
const float Pi = 3.14159265359f;
const float xrange = 2.f * Pi;
const int InferencesPerCycle = 100;
int inference_count = 0;

// Structure that contains the generated tensorflow model
const Model* model = GetModel(g_model);

// Interpreter, input and output structures
MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Allocate memory for the tensors
const int ModelArenaSize = 2468;
const int ExtraArenaSize = 560 + 16 + 100;
const int TensorArenaSize = ModelArenaSize + ExtraArenaSize;
uint8_t tensor_arena[TensorArenaSize];

// Led to indicate the resulted values
PwmOut led(LED1);


void setLed(float value) {
  led = abs(value);
}

void printValues(float x, float y) {
  printf("x_value: % 2.3f, y_value: % 02.3f\n", x, y);
}

float inference(float x) {
  input->data.f[0] = x;
  interpreter->Invoke();
  return output->data.f[0];
}

float generateNextPosition() {
    float position = (float) inference_count / InferencesPerCycle;
    float y_value =  position * xrange;
    inference_count = (inference_count + 1) % InferencesPerCycle;
    return y_value;
}

void run_once() {
    float x_value = generateNextPosition();

    float y_value = inference(x_value);

    setLed(y_value);
    printValues(x_value, y_value);
}

int main(void) {
  static AllOpsResolver resolver;
  static MicroInterpreter static_interpreter(model, resolver, tensor_arena, TensorArenaSize, NULL);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  queue.call_every(10ms, run_once);
  queue.dispatch_forever();
}


