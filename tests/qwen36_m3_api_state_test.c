/* Live API-state test for the async decode interface. Requires the packed
 * model directory and metallib, so it is a separate target and is not part
 * of the fixture-only `make test` suite. */

#include "qwen36_m3_decode.h"
#include "qwen36_m3_global_image.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned failures;

static void check(int condition, const char *label) {
    printf("check=%s status=%s\n", label, condition ? "PASS" : "FAIL");
    if (!condition) ++failures;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model-directory> <metallib>\n", argv[0]);
        return 2;
    }
    char error[512];
    qwen36_m3_decode_result result;
    const float *logits = NULL;
    size_t logit_count = 0;

    check(qwen36_m3_model_forward_submit(NULL, 0, 0, error,
                                         sizeof(error)) != 0,
          "submit_null_model_rejected");
    check(qwen36_m3_model_forward_wait(NULL, &result, &logits, &logit_count,
                                       error, sizeof(error)) != 0,
          "wait_null_model_rejected");

    qwen36_m3_model *model =
        qwen36_m3_model_open(argv[1], argv[2], 8, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model open failed: %s\n", error);
        return 2;
    }

    check(qwen36_m3_model_forward_wait(model, &result, &logits, &logit_count,
                                       error, sizeof(error)) != 0,
          "wait_without_submit_rejected");
    check(qwen36_m3_model_forward_submit(model, QWEN36_VOCAB_SIZE, 0, error,
                                         sizeof(error)) != 0,
          "submit_invalid_token_rejected");
    check(qwen36_m3_model_forward_submit(model, 19, 8, error,
                                         sizeof(error)) != 0,
          "submit_position_beyond_capacity_rejected");

    check(qwen36_m3_model_forward_submit(model, 19, 0, error,
                                         sizeof(error)) == 0,
          "first_submit_accepted");
    check(qwen36_m3_model_forward_submit(model, 19, 1, error,
                                         sizeof(error)) != 0,
          "second_submit_while_in_flight_rejected");
    int wait_status = qwen36_m3_model_forward_wait(
        model, &result, &logits, &logit_count, error, sizeof(error));
    check(wait_status == 0, "wait_after_submit_succeeds");
    check(wait_status == 0 && logits != NULL &&
              logit_count == QWEN36_VOCAB_SIZE,
          "wait_exposes_full_logits");
    check(wait_status == 0 && result.input_token == 19 &&
              result.position == 0 && result.duration_ms > 0.0,
          "wait_reports_submitted_token");
    check(qwen36_m3_model_forward_wait(model, &result, &logits, &logit_count,
                                       error, sizeof(error)) != 0,
          "second_wait_rejected");

    check(qwen36_m3_model_forward(model, 19, 1, &result, &logits,
                                  &logit_count, error, sizeof(error)) == 0,
          "synchronous_forward_still_works");

    check(qwen36_m3_model_forward_submit(model, 19, 2, error,
                                         sizeof(error)) == 0,
          "submit_before_reset_accepted");
    qwen36_m3_model_reset(model);
    check(qwen36_m3_model_forward_submit(model, 19, 0, error,
                                         sizeof(error)) == 0 &&
              qwen36_m3_model_forward_wait(model, &result, &logits,
                                           &logit_count, error,
                                           sizeof(error)) == 0,
          "reset_drains_pending_and_allows_new_forward");

    check(qwen36_m3_model_forward_submit(model, 19, 1, error,
                                         sizeof(error)) == 0,
          "submit_before_close_accepted");
    qwen36_m3_model_close(model);
    check(1, "close_drains_pending_without_crash");

    printf("VERDICT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
