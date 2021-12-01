/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mbed.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

using namespace tflite;


// This constant represents the range of x values our model was trained on,
// which is from 0 to (2 * Pi). We approximate Pi to avoid requiring additional
// libraries.
const float Pi = 3.14159265359f;
const float kXrange = 2.f * Pi;

// This constant determines the number of inferences to perform across the range
// of x values defined above. Since each inference takes time, the higher this
// number, the more time it will take to run through the entire range. The value
// of this constant can be tuned so that one full cycle takes a desired amount
// of time. Since different devices take different amounts of time to perform
// inference, this value should be defined per-device.
const int InferencesPerCycle = 100;

PwmOut led(LED1);


const Model* model = nullptr;
MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Minimum arena size, at the time of writing. After allocating tensors
// you can retrieve this value by invoking interpreter.arena_used_bytes().
const int ModelArenaSize = 2468;
// Extra headroom for model + alignment + future interpreter changes.
const int ExtraArenaSize = 560 + 16 + 100;
const int TensorArenaSize = ModelArenaSize + ExtraArenaSize;
uint8_t tensor_arena[TensorArenaSize];


void setLed(float value) {
  led = abs(value);
}

// Output the results.
// Log the current X and Y values

void printValues(float x, float y) {
  printf("x_value: % 2.3f, y_value: % 02.3f\n", x, y);
}

float infeer(float x) {
  // Place our calculated x value in the model's input tensor
  input->data.f[0] = x;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    printf("Invoke failed on x_val: %f\n", x);
    return -1.0;
  }

  interpreter->Invoke();

  // Read the predicted y value from the model's output tensor
  return output->data.f[0];
}




int main(void) {
  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  // static MicroErrorReporter micro_error_reporter;
  // error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {

    printf("Model provided is schema version %d not equal "
          "to supported version %d.",
          (int) model->version(), (int) TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TensorArenaSize, NULL);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  while (true) {
    // Calculate an x value to feed into the model. We compare the current
    // inference_count to the number of inferences per cycle to determine
    // our position within the range of possible x values the model was
    // trained on, and use this to calculate a value.
    float position = (float) inference_count / InferencesPerCycle;
    float x_val = position * kXrange;

    float y_val = infeer(x_val);

    setLed(y_val);
    printValues(x_val, y_val);

    // Increment the inference_counter, and reset it if we have reached
    // the total number per cycle
    inference_count = (inference_count + 1) % InferencesPerCycle;
    // if (inference_count >= InferencesPerCycle) inference_count = 0;
  }
}


