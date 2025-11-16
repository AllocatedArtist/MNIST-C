#ifndef LAYER_H_
#define LAYER_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <float.h>

#define LAYER_COUNT 4

#define SL_SIZE 300
#define TL_SIZE 100
#define DIGIT_COUNT 10

#define INPUT_LAYER 0
#define SECOND_LAYER 1
#define THIRD_LAYER 2
#define FINAL_LAYER 3

typedef struct {
  float *activations_;

  float *biases_;
  float *weights_;

  float *inputs_;

  float *dl_;

  float *weight_gradients_;
  float *bias_gradients_;

  int neuron_count_;
  int input_count_;
} layer_t;

void layer_init(layer_t *layer, int input_size, int neuron_size) {
  layer->inputs_ = malloc(sizeof(float) * neuron_size);
  layer->activations_ = malloc(sizeof(float) * neuron_size);

  layer->weights_ = NULL;
  layer->weight_gradients_ = NULL;
  layer->biases_ = NULL;
  layer->bias_gradients_ = NULL;
  layer->dl_ = NULL;

  if (input_size > 0) {
    layer->weights_ = malloc(sizeof(float) * input_size * neuron_size);
    layer->weight_gradients_ = malloc(sizeof(float) * input_size * neuron_size);

    layer->biases_ = malloc(sizeof(float) * neuron_size);
    layer->bias_gradients_ = malloc(sizeof(float) * neuron_size);

    layer->dl_ = malloc(sizeof(float) * neuron_size);
  }

  layer->neuron_count_ = neuron_size;
  layer->input_count_ = input_size;
}

void forward_pass_validate(const layer_t *input, const layer_t *output,
                           float z) {
  if (fabs(z) == INFINITY || isnan(z)) {
    printf("Input: %d neurons, Output: %d neurons\n", input->neuron_count_,
           output->neuron_count_);
    printf("Input:\n");
    for (int i = 0; i < input->neuron_count_; ++i) {
      printf("%f\n", input->activations_[i]);
    }
    printf("Weights:\n");
    for (int i = 0; i < output->neuron_count_ * input->neuron_count_; ++i) {
      printf("%f\n", output->weights_[i]);
    }
    printf("Biases:\n");
    for (int i = 0; i < output->neuron_count_; ++i) {
      printf("%f\n", output->biases_[i]);
    }
  }

  assert(fabs(z) != INFINITY || !isnan(z) && "Invalid dot product");
}

void layer_forward_pass(const layer_t *input, layer_t *output) {
  int input_size = input->neuron_count_;
  int output_size = output->neuron_count_;

  for (int y = 0; y < output_size; ++y) {
    float dp = output->biases_[y];

    for (int x = 0; x < input_size; ++x) {
      float act = input->activations_[x];
      float weight = output->weights_[y * input_size + x];

      dp += act * weight;
    }

    forward_pass_validate(input, output, dp);

    output->inputs_[y] = dp;
  }
}

float sigmoid(float input) { return 1.0f / (1 + expf(-input)); }

void layer_sigmoid(layer_t *output) {
  int output_size = output->neuron_count_;
  for (int i = 0; i < output_size; ++i) {
    float input = output->inputs_[i];
    output->activations_[i] = sigmoid(input);
  }
}

float output_cost(layer_t *output, int expected) {
  assert(output->neuron_count_ == DIGIT_COUNT);

  float loss = 0.0f;

  int expected_val[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  expected_val[expected] = 1;

  float sum = 0.0f;
  for (int i = 0; i < DIGIT_COUNT; ++i) {
    sum += powf((output->activations_[i] - expected_val[i]), 2.0f);
  }

  return sum;
}

float cross_entropy(const layer_t *output, int expected) {
  assert(output->neuron_count_ == DIGIT_COUNT);

  return -log(fmaxf(output->activations_[expected], FLT_EPSILON));
}

float cost_derivative(float predicted, float expected) {
  return 2 * (predicted - expected);
}

float softmax(const layer_t *output_layer, float current_input) {
  float max_in = output_layer->inputs_[0];

  for (int i = 1; i < output_layer->neuron_count_; ++i) {
    float input = output_layer->inputs_[i];
    if (max_in < input) {
      max_in = input;
    }
  }

  float bottom = 0.0f;
  for (int i = 0; i < output_layer->neuron_count_; ++i) {
    float input = output_layer->inputs_[i];
    bottom += exp(input - max_in);
  }

  assert(bottom != 0.0f && "divide by zero");

  float top = exp(current_input - max_in);

  return top / bottom;
}

void layer_softmax(layer_t *output) {
  for (int i = 0; i < output->neuron_count_; ++i) {
    output->activations_[i] = softmax(output, output->inputs_[i]);
  }
}

float sigmoid_derivative(float activation) {
  return activation * (1 - activation);
}

void layer_output_backwards(layer_t *output, int expected) {
  assert(expected >= 0 && expected < 10 && "Invalid expectation.");
  int expected_val[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  expected_val[expected] = 1.0f;

  for (int i = 0; i < output->neuron_count_; ++i) {
    // derivative of cross entropy multiplied by derivative of softmax
    // yields simple subtraction
    output->dl_[i] = output->activations_[i] - expected_val[i];
  }
}

void layer_compute_hidden(layer_t *current, layer_t *next) {
  int next_size = next->neuron_count_;
  int current_size = current->neuron_count_;

  for (int x = 0; x < current_size; ++x) {
    float dl = 0.0f;
    for (int y = 0; y < next_size; ++y) {
      // derivative of next layer
      float dc_dn = next->dl_[y];
      float weight = next->weights_[y * current_size + x];
      dl += weight * dc_dn;
    }
    // backwards sigmoid
    float da_dx = sigmoid_derivative(current->activations_[x]);
    dl *= da_dx;
    current->dl_[x] = dl;
  }
}

void layer_compute_gradients(const layer_t *input_layer, layer_t *current) {

  int current_layer_size = current->neuron_count_;
  int input_size = input_layer->neuron_count_;

  for (int y = 0; y < current_layer_size; ++y) {
    float dl = current->dl_[y];
    for (int x = 0; x < input_size; ++x) {
      float act = input_layer->activations_[x];
      current->weight_gradients_[y * input_size + x] += act * dl;
    }
    current->bias_gradients_[y] += dl;
  }
}

void layer_update(layer_t *current, float batch_size, float learning_rate) {

  float cf = learning_rate / batch_size;

  int input_count = current->input_count_;

  for (int i = 0; i < current->neuron_count_; ++i) {
    current->biases_[i] -= cf * current->bias_gradients_[i];
    for (int j = 0; j < input_count; ++j) {
      int current_index = j + i * current->input_count_;
      current->weights_[current_index] -=
          cf * current->weight_gradients_[current_index];
    }
  }
}

#endif
