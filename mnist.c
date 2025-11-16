#ifndef MNIST_H_
#define MNIST_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PIXEL_COUNT 28 * 28

typedef struct {
  uint8_t *data_;
  int data_size_;
} image_t;

typedef struct {
  image_t image_;
  int label_;
} mnist_input;

typedef struct {
  mnist_input *testing_data_;
  int testing_data_size_;
} mnist_t;

typedef enum {
  UBYTE = 0x08,
  SBYTE = 0x09,
  SHORT = 0x0B,
  INT = 0x0C,
  FLOAT = 0x0D,
  DOUBLE = 0x0E
} mnist_magic_number;

void mnist_read_test_data(const char *path, mnist_t *mnist) {
  FILE *fp = fopen(path, "rb");
  assert(fp != NULL && "File not found");

  uint32_t magic_number;

  fread(&magic_number, sizeof(uint32_t), 1, fp);
  magic_number = __builtin_bswap32(magic_number);

  uint32_t type = (magic_number >> 8) & 0xFF;

  assert(type == UBYTE && "Expected unsigned byte");

  uint32_t dimensions_count = magic_number & 0xFF;

  uint32_t size = 1;
  for (int i = 0; i < dimensions_count; ++i) {
    uint32_t dimension_size = 0;
    fread(&dimension_size, sizeof(uint32_t), 1, fp);
    dimension_size = __builtin_bswap32(dimension_size);

    size *= dimension_size;
  }

  uint8_t *all_bytes = malloc(size);
  assert(all_bytes != NULL && "Unable to alloc memory");

  uint32_t bytes_read = fread(all_bytes, sizeof(uint8_t), size, fp);

  assert(bytes_read == size && "Unable to read all data");

  int img_count = size / (PIXEL_COUNT);

  mnist->testing_data_size_ = img_count;
  mnist->testing_data_ = malloc(sizeof(mnist_input) * img_count);

  for (int i = 0; i < img_count; ++i) {
    int offset = PIXEL_COUNT;
    image_t *current = &mnist->testing_data_[i].image_;

    current->data_ = malloc(PIXEL_COUNT);
    memcpy(current->data_, all_bytes + i * PIXEL_COUNT, PIXEL_COUNT);
    current->data_size_ = PIXEL_COUNT;
  }

  free(all_bytes);

  fclose(fp);
}

void mnist_read_label_data(const char *path, mnist_t *mnist) {
  FILE *fp = fopen(path, "rb");
  assert(fp != NULL && "File not found");

  uint32_t magic_number;

  fread(&magic_number, sizeof(uint32_t), 1, fp);
  magic_number = __builtin_bswap32(magic_number);

  uint32_t type = (magic_number >> 8) & 0xFF;

  assert(type == UBYTE && "Expected unsigned byte");

  uint32_t dimensions_count = magic_number & 0xFF;

  uint32_t size = 1;
  for (int i = 0; i < dimensions_count; ++i) {
    uint32_t dimension_size = 0;
    fread(&dimension_size, sizeof(uint32_t), 1, fp);
    dimension_size = __builtin_bswap32(dimension_size);

    size *= dimension_size;
  }

  uint8_t *all_bytes = malloc(size);

  fread(all_bytes, sizeof(uint8_t), size, fp);

  for (int i = 0; i < size; ++i) {
    mnist->testing_data_[i].label_ = all_bytes[i];
  }

  free(all_bytes);

  fclose(fp);
}

void mnist_free(mnist_t *mnist) {
  for (int i = 0; i < mnist->testing_data_size_; ++i) {
    free(mnist->testing_data_[i].image_.data_);
  }
  free(mnist->testing_data_);
}

void shuffle_validate(const mnist_input *shuffled, const mnist_t *unshuffled) {
  int size = unshuffled->testing_data_size_;
  for (int i = 0; i < size; ++i) {
    int detected = 0;
    for (int j = 0; j < size; ++j) {
      if (shuffled[i].image_.data_ ==
          unshuffled->testing_data_[j].image_.data_) {
        detected = 1;
        break;
      }
    }
    assert(detected == 1 && "Shuffle shouldn't lose data");
  }
}

void mnist_shuffle(mnist_t *mnist) {
  mnist_input *in = mnist->testing_data_;

  int size = mnist->testing_data_size_;
  int indices[mnist->testing_data_size_];

  for (int i = 0; i < size; ++i) {
    indices[i] = i;
  }

  for (int i = size - 1; i > 0; --i) {
    int j = rand() % (i + 1);
    int temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  mnist_input shuffled_array[size];
  for (int i = 0; i < size; ++i) {
    shuffled_array[i] = in[indices[i]];
  }

  for (int i = 0; i < size; ++i) {
    in[i] = shuffled_array[i];
  }
}

#endif
