#include "mbed.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "model.h"

using namespace tflite;
using namespace std::chrono;

EventQueue queue;

// Application constants and variables
const float Pi = 3.14159265359f;
const float xrange = 2.f * Pi;
const int InferencesPerCycle = 100;
int inference_count = 0;

// Structure that contains the generated tensorflow model
const Model* model = GetModel(g_model);

// Allocate memory for the tensors
const int ModelArenaSize = 4468;
const int ExtraArenaSize = 560 + 16 + 100;
const int TensorArenaSize = ModelArenaSize + ExtraArenaSize;
uint8_t tensor_arena[TensorArenaSize];

// Led to indicate the resulted values
PwmOut nucleo_led(LED1);
DigitalOut sensortile_led((PinName)0x6C);

struct Context {
  TfLiteTensor* input;
  RecordingMicroInterpreter* interpreter;
  TfLiteTensor* output;
};

void setLed(float value) {
  nucleo_led = (value + 1) / 2;
  sensortile_led = value < 0.0f ? 0 : 1;
}

void printValues(float x, float y) {
  printf("x_value: % 2.3f, y_value: % 02.3f\n", x, y);
}

float generateNextXValue() {
    float position = (float) inference_count / InferencesPerCycle;
    float x_value =  position * xrange;
    inference_count = (inference_count + 1) % InferencesPerCycle;
    return x_value;
}

float inference(float x, Context& context) {
  float input_scale = context.input->params.scale;
  int input_zero_point = context.input->params.zero_point;

  float output_scale = context.output->params.scale;
  int output_zero_point = context.output->params.zero_point;

  context.input->data.int8[0] = x / input_scale + input_zero_point;
  context.interpreter->Invoke();
  return (context.output->data.int8[0] - output_zero_point) * output_scale;
}

void run_once(Context& context) {
    float x_value = generateNextXValue();

    Timer t;

    printf("[INFERENCE] Start\n");
    t.start();

    float y_value = inference(x_value, context);

    t.stop();
    printf("[INFERENCE] End\n");
    printf("The time taken was %llu microseconds\n", duration_cast<microseconds>(t.elapsed_time()).count());

    setLed(y_value);
    printValues(x_value, y_value);
}

int main(void) {
  printf("Starting TensorFlow Lite Micro model...\n");

  MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  // constexpr int kNumResourceVariables = 24;
  // RecordingMicroAllocator* allocator(RecordingMicroAllocator::Create(tensor_arena, TensorArenaSize));
  // MicroResourceVariables* resource_variables = MicroResourceVariables::Create(allocator, kNumResourceVariables);

  MicroProfiler profiler;
  RecordingMicroInterpreter interpreter(model, resolver, tensor_arena, TensorArenaSize, nullptr, nullptr);
  interpreter.AllocateTensors();

  // profiler.LogTicksPerTagCsv();

  interpreter.GetMicroAllocator().PrintAllocations();

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  Context context = {input, &interpreter, output};

  // queue.call_every(10ms, run_once, context);
  queue.call(run_once, context);
  queue.dispatch_forever();
}


